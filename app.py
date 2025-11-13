import os
from typing import List, Dict, Any, Optional
import io
import re
import uuid
import base64
from datetime import datetime
from fastapi import File, UploadFile, Body
from pdfminer.high_level import extract_text as pdf_extract_text
from pdf2image import convert_from_bytes
import pytesseract
import html as _html

import requests
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from monkdb import client
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# Config
# -------------------------

MONKDB_URL = os.getenv("MONKDB_URL", "http://localhost:4200")
MONKDB_USERNAME = os.getenv("MONKDB_USERNAME", "testuser")
MONKDB_PASSWORD = os.getenv("MONKDB_PASSWORD", "testpassword")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# -------------------------
# App + Models
# -------------------------

app = FastAPI(
    title="Global Hybrid Search + Chatbot (MonkDB + Ollama/Mistral)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model once
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_conn():
    if MONKDB_PASSWORD:
        return client.connect(
            MONKDB_URL,
            username=MONKDB_USERNAME,
            password=MONKDB_PASSWORD,
        )
    return client.connect(
        MONKDB_URL,
        username=MONKDB_USERNAME,
    )


class SearchResult(BaseModel):
    id: str
    entity_type: str
    title: str
    body_snippet: str
    amount: float
    priority_score: float
    score_text: float
    score_vector: float
    score_hybrid: float


class ChatRequest(BaseModel):
    question: str
    k: int = 8
    alpha: float = 0.6
    entity_type: Optional[str] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SearchResult]


# -------------------------
# Helpers
# -------------------------

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def _clean_snippet(txt: str, max_len: int = 320) -> str:
    if not txt:
        return ""
    t = _html.unescape(txt)
    t = TAG_RE.sub("", t)
    t = WS_RE.sub(" ", t).strip()
    return (t[:max_len].rstrip() + "…") if len(t) > max_len else t


def normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_s = max(scores.values())
    min_s = min(scores.values())
    if max_s == min_s:
        return {k: 1.0 for k in scores}
    return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}


def run_hybrid_search(
    q: str,
    k: int = 10,
    alpha: float = 0.6,
    entity_type: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
) -> List[SearchResult]:
    """
    Core hybrid search:
      - Vector: knn_match(embedding, query_emb, k')
      - Text: MATCH(ft_all, query)
      - Filters: entity_type, min_amount, max_amount
      - Hybrid score = alpha * text + (1 - alpha) * vector
    """
    # 1) Embed query
    q_emb = embed_model.encode(q, normalize_embeddings=True).tolist()

    conn = get_conn()
    cur = conn.cursor()

    # 2) Vector search
    vec_where = ["knn_match(embedding, ?, ?)"]
    vec_params: List[Any] = [q_emb, k * 4]  # overfetch

    if entity_type:
        vec_where.append("entity_type = ?")
        vec_params.append(entity_type)
    if min_amount is not None:
        vec_where.append("amount >= ?")
        vec_params.append(min_amount)
    if max_amount is not None:
        vec_where.append("amount <= ?")
        vec_params.append(max_amount)

    vec_sql = """
        SELECT id, entity_type, title, body, amount, priority_score, _score
        FROM doc.app_search
        WHERE {}
        ORDER BY _score DESC
        LIMIT ?
        """.format(' AND '.join(vec_where))
    cur.execute(vec_sql, tuple(vec_params + [k * 4]))
    vec_rows = cur.fetchall()

    # 3) Full-text search
    ft_where = ["MATCH(ft_all, ?)"]
    ft_params: List[Any] = [q]

    if entity_type:
        ft_where.append("entity_type = ?")
        ft_params.append(entity_type)
    if min_amount is not None:
        ft_where.append("amount >= ?")
        ft_params.append(min_amount)
    if max_amount is not None:
        ft_where.append("amount <= ?")
        ft_params.append(max_amount)

    ft_sql = f"""
        SELECT id, entity_type, title, body,
               amount, priority_score, _score
        FROM doc.app_search
        WHERE {' AND '.join(ft_where)}
        ORDER BY _score DESC
        LIMIT ?
    """
    ft_params.append(k * 4)
    cur.execute(ft_sql, tuple(ft_params))
    ft_rows = cur.fetchall()

    conn.close()

    # 4) Merge & score
    items: Dict[str, Dict[str, Any]] = {}
    vec_scores: Dict[str, float] = {}
    text_scores: Dict[str, float] = {}

    for rid, etype, title, body, amount, prio, s in vec_rows:
        items.setdefault(
            rid,
            dict(
                id=rid,
                entity_type=etype,
                title=title,
                body=body or "",
                amount=float(amount) if amount is not None else 0.0,
                priority_score=float(prio) if prio is not None else 0.0,
            ),
        )
        vec_scores[rid] = float(s)

    for rid, etype, title, body, amount, prio, s in ft_rows:
        items.setdefault(
            rid,
            dict(
                id=rid,
                entity_type=etype,
                title=title,
                body=body or "",
                amount=float(amount) if amount is not None else 0.0,
                priority_score=float(prio) if prio is not None else 0.0,
            ),
        )
        text_scores[rid] = float(s)

    vec_norm = normalize(vec_scores)
    text_norm = normalize(text_scores)

    results: List[SearchResult] = []
    for rid, meta in items.items():
        sv = vec_norm.get(rid, 0.0)
        st = text_norm.get(rid, 0.0)
        hybrid = alpha * st + (1.0 - alpha) * sv

        results.append(
            SearchResult(
                id=meta["id"],
                entity_type=meta["entity_type"],
                title=_clean_snippet(meta["title"], 180),
                body_snippet=_clean_snippet(meta["body"], 5000),
                amount=meta["amount"],
                priority_score=meta["priority_score"],
                score_text=st,
                score_vector=sv,
                score_hybrid=hybrid,
            )
        )

    results.sort(key=lambda r: r.score_hybrid, reverse=True)
    return results[:k]


def call_ollama_chat(prompt: str) -> str:
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant for an enterprise portal. "
                    "Use ONLY the provided context. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500, detail=f"Ollama request failed: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama error: {resp.status_code} {resp.text}",
        )

    data = resp.json()
    try:
        return data["message"]["content"].strip()
    except Exception:
        raise HTTPException(
            status_code=500, detail="Unexpected Ollama response format")


def _now():
    return datetime.utcnow()


def _uuid():
    return uuid.uuid4().hex


def extract_text_from_pdf_bytes(pdf_bytes: bytes, do_ocr_fallback: bool = True) -> Dict[str, Any]:
    """
    Returns: { 'text_full': str, 'pages': [{'page_no':1,'text': '...'}], 'page_count': N, 'used_ocr': bool }
    """
    text = ""
    pages_info = []
    used_ocr = False

    # Try digital text first
    try:
        text = pdf_extract_text(io.BytesIO(pdf_bytes)) or ""
    except Exception:
        text = ""

    # If no text and OCR allowed → rasterize pages and OCR
    if not text.strip() and do_ocr_fallback:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        for idx, img in enumerate(images, start=1):
            pg_text = pytesseract.image_to_string(img)
            pages_info.append({"page_no": idx, "text": pg_text})
            text += f"\n\n{pg_text}"
        used_ocr = True
        page_count = len(images)
    else:
        # Split by simple page separators heuristic (keeps it lightweight)
        raw_pages = [p for p in text.split('\f') if p is not None]
        if len(raw_pages) <= 1:
            # fallback: naive chunk by ~3000 chars
            raw_pages = [text[i:i+3000]
                         for i in range(0, len(text), 3000)] or [text]
        for i, pg in enumerate(raw_pages, start=1):
            pages_info.append({"page_no": i, "text": pg})
        page_count = len(pages_info)

    return {
        "text_full": text.strip(),
        "pages": pages_info,
        "page_count": page_count,
        "used_ocr": used_ocr
    }


def yield_chunks(filename: str, pages: List[Dict[str, Any]], max_tokens: int = 200, overlap: int = 40):
    """
    Simple paragraph/length-aware chunker. Keeps order; tags with page_no.
    """
    for pg in pages:
        page_no = pg["page_no"]
        txt = (pg["text"] or "").strip()
        if not txt:
            continue
        # split on blank lines as crude layout proxy
        blocks = [b.strip() for b in re.split(r"\n\s*\n", txt) if b.strip()]
        buf = []
        size = 0
        for blk in blocks:
            tokens = max(1, len(blk.split()))
            if size + tokens <= max_tokens:
                buf.append(blk)
                size += tokens
            else:
                if buf:
                    content = "\n\n".join(buf)
                    yield page_no, content
                # overlap by tail of previous
                tail = " ".join((" ".join(buf)).split()
                                [-overlap:]) if buf else ""
                buf = [tail, blk] if tail else [blk]
                size = len(" ".join(buf).split())
        if buf:
            content = "\n\n".join(buf)
            yield page_no, content


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://[^\s)]+")
DATE_RE = re.compile(
    r"(?:(?:\d{1,2}[/.-]){2}\d{2,4})|(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})", re.I)
TITLE_RE = re.compile(r"^\s*(.+?)\n", re.S)


def simple_extract_fields(text_full: str) -> List[Dict[str, Any]]:
    results = []

    def add(field, value, conf, evidence_snippet):
        results.append({
            "field_name": field,
            "field_value": value,
            "confidence": float(conf),
            "evidence": {"snippet": evidence_snippet[:240]}
        })

    # Title heuristic = first non-empty line
    m = TITLE_RE.search(text_full)
    if m:
        title = m.group(1).strip()
        add("title", title, 0.6, title)

    # Dates
    dates = DATE_RE.findall(text_full)
    for d in list(set(dates))[:5]:
        add("date", d, 0.5, d)

    # Emails
    emails = EMAIL_RE.findall(text_full)
    for e in list(set(emails))[:10]:
        add("email", e, 0.9, e)

    # URLs
    urls = URL_RE.findall(text_full)
    for u in list(set(urls))[:10]:
        add("url", u, 0.8, u)

    return results


def ensure_tables_exist():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS doc.documents (
      doc_id TEXT PRIMARY KEY, filename TEXT, mime_type TEXT, page_count INTEGER,
      source_uri TEXT, tags ARRAY(TEXT), text_full TEXT,
      created_at TIMESTAMP WITH TIME ZONE, status TEXT, error TEXT,
      INDEX ft_all USING FULLTEXT (text_full) WITH (analyzer='english')
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS doc.extractions (
      extract_id TEXT PRIMARY KEY, doc_id TEXT, schema_name TEXT,
      field_name TEXT, field_value TEXT, field_num DOUBLE, confidence DOUBLE,
      evidence OBJECT(DYNAMIC), is_approved BOOLEAN, corrected_value TEXT,
      created_at TIMESTAMP WITH TIME ZONE
    )
    """)
    conn.close()


def insert_document_row(doc_id: str, filename: str, mime: str, page_count: int, source_uri: str, text_full: str, status: str, error: str = None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO doc.documents (doc_id, filename, mime_type, page_count, source_uri, text_full, created_at, status, error)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT (doc_id) DO UPDATE SET
        filename = EXCLUDED.filename, mime_type = EXCLUDED.mime_type, page_count = EXCLUDED.page_count,
        source_uri = EXCLUDED.source_uri, text_full = EXCLUDED.text_full,
        status = EXCLUDED.status, error = EXCLUDED.error
    """, (doc_id, filename, mime, page_count, source_uri, text_full, _now(), status, error))
    conn.close()


def bulk_insert_chunks_into_app_search(doc_id: str, filename: str, chunks: List[Dict[str, Any]]):
    conn = get_conn()
    cur = conn.cursor()
    for ch in chunks:
        title = f"{filename} — p{ch['page_no']}"
        body = ch["content"]
        emb = embed_model.encode(
            title + "\n" + body, normalize_embeddings=True).tolist()
        rec_id = _uuid()
        cur.execute("""
            INSERT INTO doc.app_search (id, entity_type, title, body, amount, priority_score, created_at, embedding)
            VALUES (?, 'document', ?, ?, 0.0, 0.0, ?, ?)
        """, (rec_id, title, body, _now(), emb))
    conn.close()


def bulk_insert_extractions(doc_id: str, fields: List[Dict[str, Any]]):
    conn = get_conn()
    cur = conn.cursor()
    rows = []
    for f in fields:
        rows.append((
            _uuid(
            ), doc_id, "generic_doc.v1", f["field_name"], f["field_value"],
            None, float(f["confidence"]), f.get(
                "evidence", {}), False, None, _now()
        ))
    cur.executemany("""
      INSERT INTO doc.extractions
        (extract_id, doc_id, schema_name, field_name, field_value, field_num,
         confidence, evidence, is_approved, corrected_value, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.close()


class FetchRequest(BaseModel):
    url: str
    filename: Optional[str] = None


@app.post("/documents/fetch")
def documents_fetch(req: FetchRequest):
    ensure_tables_exist()
    try:
        r = requests.get(req.url, timeout=60)
        r.raise_for_status()
        pdf_bytes = r.content
    except requests.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to download PDF: {e}")

    filename = req.filename or req.url.split("/")[-1] or "document.pdf"
    doc_id = _uuid()

    insert_document_row(doc_id, filename, "application/pdf",
                        0, req.url, "", "processing")

    try:
        parsed = extract_text_from_pdf_bytes(pdf_bytes)
        text_full = parsed["text_full"]
        page_count = parsed["page_count"]

        # Chunk + index into app_search
        chunks = []
        for page_no, content in yield_chunks(filename, parsed["pages"]):
            chunks.append({"page_no": page_no, "content": content})
        if chunks:
            bulk_insert_chunks_into_app_search(doc_id, filename, chunks)

        # Write document row
        insert_document_row(doc_id, filename, "application/pdf",
                            page_count, req.url, text_full, "ready")

        # Simple field extraction + validation
        fields = simple_extract_fields(text_full)
        bulk_insert_extractions(doc_id, fields)

        return {"doc_id": doc_id, "filename": filename, "page_count": page_count, "chunks_indexed": len(chunks), "fields_extracted": len(fields)}
    except Exception as e:
        insert_document_row(doc_id, filename, "application/pdf",
                            0, req.url, "", "error", str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@app.post("/documents/upload")
async def documents_upload(file: UploadFile = File(...)):
    ensure_tables_exist()
    if file.content_type not in ("application/pdf",):
        raise HTTPException(
            status_code=400, detail="Only PDFs are supported in this demo")
    pdf_bytes = await file.read()
    filename = file.filename or "document.pdf"
    # reuse logic via in-memory fetch path
    tmp = requests.models.Response()
    # call the same core (copy of above minus requests.get)
    doc_id = _uuid()
    insert_document_row(doc_id, filename, "application/pdf",
                        0, "upload://local", "", "processing")
    try:
        parsed = extract_text_from_pdf_bytes(pdf_bytes)
        text_full = parsed["text_full"]
        page_count = parsed["page_count"]
        chunks = []
        for page_no, content in yield_chunks(filename, parsed["pages"]):
            chunks.append({"page_no": page_no, "content": content})
        if chunks:
            bulk_insert_chunks_into_app_search(doc_id, filename, chunks)
        insert_document_row(doc_id, filename, "application/pdf",
                            page_count, "upload://local", text_full, "ready")
        fields = simple_extract_fields(text_full)
        bulk_insert_extractions(doc_id, fields)
        return {"doc_id": doc_id, "filename": filename, "page_count": page_count, "chunks_indexed": len(chunks), "fields_extracted": len(fields)}
    except Exception as e:
        insert_document_row(doc_id, filename, "application/pdf",
                            0, "upload://local", "", "error", str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@app.get("/extractions/{doc_id}")
def get_extractions(doc_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
      SELECT extract_id, schema_name, field_name, field_value, confidence, evidence, is_approved, corrected_value, created_at
      FROM doc.extractions
      WHERE doc_id = ?
      ORDER BY confidence DESC, created_at DESC
    """, (doc_id,))
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "extract_id": r[0],
            "schema_name": r[1],
            "field_name": r[2],
            "field_value": r[3],
            "confidence": float(r[4]) if r[4] is not None else None,
            "evidence": r[5],
            "is_approved": bool(r[6]) if r[6] is not None else False,
            "corrected_value": r[7],
            "created_at": r[8].isoformat() if r[8] else None
        } for r in rows
    ]


class ReviewRequest(BaseModel):
    corrected_value: str


@app.post("/review/{extract_id}")
def review_update(extract_id: str, req: ReviewRequest):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
      UPDATE doc.extractions
      SET is_approved = TRUE, corrected_value = ?
      WHERE extract_id = ?
    """, (req.corrected_value, extract_id))
    conn.close()
    return {"ok": True, "extract_id": extract_id}

# -------------------------
# Public APIs
# -------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search/hybrid", response_model=List[SearchResult])
def search_hybrid(
    q: str = Query(..., description="Search text"),
    k: int = Query(10, ge=1, le=100),
    alpha: float = Query(0.6, ge=0.0, le=1.0),
    entity_type: Optional[str] = Query(
        None, description="Filter by entity type"),
    min_amount: Optional[float] = Query(None),
    max_amount: Optional[float] = Query(None),
):
    """
    Primary global search API.
    Combines BM25 + vector + numeric filters + entity_type.
    """
    return run_hybrid_search(q, k, alpha, entity_type, min_amount, max_amount)


@app.get("/search/global", response_model=List[SearchResult])
def search_global(q: str = Query(...), k: int = Query(10)):
    """
    Convenience alias for a global search (no filters, default alpha).
    """
    return run_hybrid_search(q=q, k=k, alpha=0.6)


@app.get("/search/by-entity", response_model=List[SearchResult])
def search_by_entity(
    q: str = Query(...),
    entity_type: str = Query(...),
    k: int = Query(10),
    alpha: float = Query(0.6),
):
    """
    Search within a specific entity_type (e.g. 'order', 'ticket').
    """
    return run_hybrid_search(q=q, k=k, alpha=alpha, entity_type=entity_type)


@app.get("/search/numeric", response_model=List[SearchResult])
def search_numeric(
    q: str = Query(...),
    min_amount: float = Query(...),
    max_amount: float = Query(...),
    k: int = Query(10),
    alpha: float = Query(0.6),
):
    """
    Search with mandatory numeric range condition.
    """
    return run_hybrid_search(
        q=q,
        k=k,
        alpha=alpha,
        min_amount=min_amount,
        max_amount=max_amount,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Chatbot over app_search:
    - Uses hybrid search to fetch relevant records.
    - Passes them as context to local Mistral via Ollama.
    """
    hits = run_hybrid_search(
        q=req.question,
        k=req.k,
        alpha=req.alpha,
        entity_type=req.entity_type,
        min_amount=req.min_amount,
        max_amount=req.max_amount,
    )

    if hits:
        context_blocks = []
        for h in hits:
            context_blocks.append(
                f"[{h.entity_type} | amount={h.amount} | prio={h.priority_score}] "
                f"{h.title}\n{h.body_snippet[:1000]}"
            )
        context = "\n\n".join(context_blocks)

    else:
        context = "No relevant records found."

    prompt = f"""
Context:
{context}

Question:
{req.question}
"""

    answer = call_ollama_chat(prompt)
    return ChatResponse(answer=answer, sources=hits)
