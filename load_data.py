# load_app_data.py

import os
import hashlib
import random
from datetime import datetime, timedelta
from typing import Optional

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from monkdb import client

MONKDB_URL = os.getenv("MONKDB_URL", "http://localhost:4200")
MONKDB_USERNAME = os.getenv("MONKDB_USERNAME", "testuser")
MONKDB_PASSWORD = os.getenv(
    "MONKDB_PASSWORD", "testpassword")  # if auth enabled


def get_conn():
    if MONKDB_PASSWORD:
        return client.connect(
            MONKDB_URL,
            username=MONKDB_USERNAME,
            password=MONKDB_PASSWORD,
        )
    return client.connect(MONKDB_URL, username=MONKDB_USERNAME)


def make_id(title: str, body: str) -> str:
    raw = f"{title}::{body}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def infer_entity_type(label: int) -> str:
    # Demo mapping of AG News labels → logical entity types
    if label == 1:
        return "ticket"
    if label == 2:
        return "event"
    if label == 3:
        return "order"
    if label == 4:
        return "faq"
    return "record"


def main(limit: Optional[int] = 5000):
    print("Loading dataset sh0416/ag_news ...")
    # we have used `sh0416/ag_news` dataset from HuggingFace to simulate this demo
    ds = load_dataset("sh0416/ag_news")

    records = list(ds["train"]) + list(ds["test"])
    if limit:
        records = records[:limit]

    print(f"Loaded {len(records)} records")

    print("Loading embedding model ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    conn = get_conn()
    cur = conn.cursor()

    batch_size = 256
    total = len(records)

    for i in range(0, total, batch_size):
        batch = records[i: i + batch_size]

        texts = [f"{r['title']} {r['description']}" for r in batch]
        embs = model.encode(texts, normalize_embeddings=True)

        rows = []
        for r, emb in zip(batch, embs):
            label = int(r["label"])
            title = r["title"]
            desc = r["description"]
            body = desc
            entity_type = infer_entity_type(label)
            doc_id = make_id(title, body)

            amount = round(random.uniform(10, 10000), 2)
            priority_score = round(random.uniform(0, 1), 3)
            created_at = datetime.utcnow() - timedelta(days=random.randint(0, 365))

            rows.append(
                (
                    doc_id,
                    entity_type,
                    title,
                    body,
                    amount,
                    priority_score,
                    created_at,
                    emb.tolist(),
                )
            )

        cur.executemany(
            """
            INSERT INTO doc.app_search (
                id, entity_type, title, body,
                amount, priority_score, created_at,
                embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                entity_type = EXCLUDED.entity_type,
                title = EXCLUDED.title,
                body = EXCLUDED.body,
                amount = EXCLUDED.amount,
                priority_score = EXCLUDED.priority_score,
                created_at = EXCLUDED.created_at,
                embedding = EXCLUDED.embedding
            """,
            rows,
        )

        print(f"Indexed {min(i + batch_size, total)} / {total}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
