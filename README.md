# Unified Search & Chat Demo (MonkDB + Hybrid Search + Mistral/Ollama)

This project demonstrates:

- A global search UI that queries `doc.app_search` in MonkDB.
- Full-text + vector hybrid ranking
- Numeric filtering (`min`/`max` amount)
- Entity-type filtering (orders/tickets/events/FAQs/etc.)
- A chatbot that answers questions based on the same search results, using Mistral via Ollama
- A simple HTML/CSS/JS frontend that talks to FastAPI

This README focuses on how to use the frontend end-to-end, but includes the minimal backend + data setup so everything actually works.

## 1. Architecture Overview

High level:

- MonkDB
    - Stores indexed records in doc.app_search
    - Provides:
        - `MATCH(ft_all, ?)` for full-text search
        - `knn_match(embedding, ?, k) for vector search
- FastAPI backend
    - Exposes REST APIs:
        - `GET /search/hybrid`
        - `GET /search/global`
        - `GET /search/by-entity`
        - `GET /search/numeric`
        - `POST /chat`
        - `GET /health`
- Mistral via Ollama
    - Local LLM used for the /chat endpoint
- Frontend (`index.html`)
    - Single-page app (no React)
    - Two panels:
        - Global Search
        - Chatbot

All calls go from the browser → FastAPI → MonkDB (+ Ollama for chat).

## 2. Prerequisites

Before using the frontend, make sure these are in place.

### 2.1 MonkDB & table

MonkDB running and this table created:

```sql
CREATE TABLE IF NOT EXISTS doc.app_search (
    id             TEXT PRIMARY KEY,
    entity_type    TEXT,
    title          TEXT,
    body           TEXT,
    amount         DOUBLE,
    priority_score DOUBLE,
    created_at     TIMESTAMP WITH TIME ZONE,
    embedding      FLOAT_VECTOR(384),

    INDEX ft_all USING FULLTEXT (title, body)
        WITH (analyzer = 'english')
);
```

### 2.2 Demo data (AG News → app_search)

Use the `load_data.py` script (from earlier steps) to populate demo records.

Environment:

```txt
export MONKDB_URL="http://localhost:4200"
export MONKDB_USERNAME="your_username"
export MONKDB_PASSWORD="your_password"
```

Run:

```bash
python3 load_data.py
```

Verify:

```sql
SELECT count(*) FROM doc.app_search;
```

You should see thousands of rows.

### 2.3 Ollama + Mistral

Install and run Ollama:

```bash
ollama pull mistral
ollama serve
```

Optional env:

```text
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="mistral"
```

### 2.4 FastAPI backend

Start the FastAPI app (`app.py` containing all endpoints):

```bash
uvicorn app:app --reload --port 8000
```

Quick check:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

If that works, the APIs are ready for the frontend.

## 3. Serving the Frontend

### 3.1 Directory layout

Example:

```bash
.
├─ app.py                 # FastAPI backend
├─ load_data.py       # demo data loader
└─ static/
   └─ index.html          # frontend
```
Use the provided `index.html` in static/. Run the below command to serve the HTML file by *cd'ing* to static directory.

```bash
python3 -m http.server 5500
```

## 4. Using the Global Search UI

When you open the page, the left panel is Global Search.

It’s a thin wrapper over `GET /search/hybrid`.

### 4.1 Controls

- Search box
    - Placeholder: “Search across entities…”
    - Required.
    - Drives the `q` parameter.
    - Example: `stock market`, `payment delay`, `artificial intelligence`.
- Search button
    - Executes the query using the current filters.
    - Disabled while a request is in progress.
    - Entity type dropdown
    - Values:
        - All types → no filter
        - Orders
        - Tickets
        - Events
        - FAQs
    - When set, sends entity_type to the API.
    - Example:
        - Choose Orders → &entity_type=order
        - Choose Tickets → &entity_type=ticket
- Min amount / Max amount
    - Optional numeric filters on amount.
    - If you enter:
        - Min only: `&min_amount=<value>`
        - Max only: `&max_amount=<value>`
        - Both: `&min_amount=<min>&max_amount=<max>`
    - Only results whose amount falls in that range are returned.
    - These map directly to numeric conditions in MonkDB.
- `α` slider (alpha)
    - Labelled `α` with a value readout.
    - Range: `0.0` → `1.0`
    - Controls hybrid weighting:
        - Higher α → more weight on text/BM25 (exact words).
        - Lower α → more weight on vector/semantic similarity.
- Under the hood:
```txt
score_hybrid = α * normalized_text_score + (1 - α) * normalized_vector_score
```

### 4.2 Results display

Under the filters, results render as stacked cards.

For each result:

- Entity pill – `entity_type` (e.g. order, ticket, event, faq)
- Amount – numeric amount field
- Priority – `priority_score`
- Title – the main title from the record
- Body snippet – first ~260 chars of body
- Scores – diagnostic:

```text
text: <BM25-based score> | vec: <vector score> | hybrid: <combined>
```

**If there are no matches**: A subtle “No results.” message is shown.
**If there’s an error**: An orange error message appears (e.g. API unreachable).

### 4.3 Practical usage patterns

Some concrete ways to drive it:

- Global semantic search
    - Query: cloud services regulations
    - Leave all other filters blank.
    - Adjust α:
        - Start at 0.6
        - If too fuzzy, increase toward 0.8–1.0
        - If you want more concept matches, drop to 0.3–0.4
- Per-entity scoped search
    - Query: payment failure
    - Type: Tickets
    - → Only records with entity_type=ticket are returned.
- High-value records
    - Query: stock market
    - Min amount: 5000
    - → Only high-amount rows relevant to stock market.
- Narrow & strict
    - Query: invoice dispute
    - Type: Orders
    - Min amount: 1000
    - α: 0.9
    - → Orders with strong lexical match and amount ≥ 1000.
- Wide & semantic
    - Query: AI news
    - Type: FAQs
    - α: 0.3
    - → FAQs semantically related to AI, not just exact phrase.

All of this is powered by `/search/hybrid`, so anything visible in the UI can also be automated via curl or other clients.

## 5. Using the Chatbot Panel

The right panel is Chatbot, backed by:

- POST `/chat`
- Which internally calls:
    - `run_hybrid_search` to fetch top-K relevant rows
    - Local Ollama (Mistral) with those rows as context

### 5.1 Controls

- Question textarea
    - Type any natural language question.
    - Example:
        - “Show me important business records related to stock prices.”
        - “Summarize high-value orders concerning refunds.”
        - “What tickets mention delivery delay?”
- Ask button
    - Sends the question to /chat.
    - Shows “Thinking…” while waiting.
    - Disabled during request.

### 5.2 What happens under the hood

**Frontend sends**:

```json
{
  "question": "Show me important business records related to stock prices.",
  "k": 8,
  "alpha": 0.6
}
```

**Backend**:

- Runs hybrid search on doc.app_search with that question.
- Builds a context block of the top results (titles + snippets + metadata).
- Calls Ollama’s /api/chat with:
    - System message: “Only use this context, say you don't know otherwise.”
    - User message: context + question.
- Mistral responds with a grounded summary/answer.
- Backend returns:

```json
{
  "answer": "...",
  "sources": [ ...SearchResult objects... ]
}
```

**Frontend shows**:
- The answer as prose.
- A Sources list:
    - Title
    - entity_type
    - amount
    - hybrid score (for transparency)

![Front End Display](assets/frontend_1.png)

### 5.3 How to interpret / use

- The chatbot is not inventing from the internet; it’s summarizing what hybrid search found in `doc.app_search`.
- Use it when:
    - You want an overview vs raw result list.
    - You want to show your client a “chat-with-your-data” story.
- If the context is empty or irrelevant:
    - It should answer “I don’t know” (by design of the prompt).
    - Sources list will be small or empty; that’s your debugging signal.

## Note

- You may swap mistral + ollama with something else. For example, OpenAI, etc.
- You may swap sentence transformers with something else. For example, OpenAI Embeddings, etc.
- You may swap ag-news dataset used in this demo with something else. For example, custom datasets, etc. 