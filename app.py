import os
from typing import List, Dict, Any, Optional

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

    vec_sql = f"""
        SELECT id, entity_type, title, body,
               amount, priority_score, _score
        FROM doc.app_search
        WHERE {' AND '.join(vec_where)}
    """
    cur.execute(vec_sql, tuple(vec_params))
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
                title=meta["title"],
                body_snippet=meta["body"][:260],
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
                f"{h.title}\n{h.body_snippet}"
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
