import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

# === Device setup ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧  Running inference on → {device}")

# === Load FinBERT model/tokenizer once ======================================
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
if device.type == "cuda":
    model = model.half()           # fp16 speeds up GPU inference
model.to(device).eval()

# Build lowercase→index map from model config
id2label   = model.config.id2label
label2idx  = {lab.lower(): idx for idx, lab in id2label.items()}
pos_idx    = label2idx["positive"]
neutral_idx = label2idx["neutral"]
neg_idx    = label2idx["negative"]

# --------------------------------------------------------------------------- #
def batch_probs(texts):
    """Return (pos, neu, neg) probabilities for every headline in `texts`."""
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    # Move input tensors to GPU if available
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.nn.functional.softmax(logits, dim=1)

    return [
        (float(probs[i, pos_idx]),
         float(probs[i, neutral_idx]),
         float(probs[i, neg_idx]))
        for i in range(len(texts))
    ]

# --------------------------------------------------------------------------- #
def run_resumable_finbert_probs(
    db_path="/data/news.db",
    table_name="master0",
    batch_size=256
):
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Add FinBERT columns if they don't exist
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cursor.fetchall()}
    for col in ("finbert_sentiment_pos",
                "finbert_sentiment_neutral",
                "finbert_sentiment_neg"):
        if col not in existing:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} REAL")
    conn.commit()

    # Fetch rows still needing scores
    cursor.execute(f"""
        SELECT id, title_clean
          FROM {table_name}
         WHERE finbert_sentiment_pos IS NULL
    """)
    rows   = cursor.fetchall()
    total  = len(rows)
    print(f"📊  Headlines to score: {total}")

    processed = 0
    for start in range(0, total, batch_size):
        batch      = rows[start:start + batch_size]
        ids, texts = zip(*batch)
        texts      = [t or "" for t in texts]

        probs = batch_probs(texts)

        cursor.executemany(
            f"""
            UPDATE {table_name}
               SET finbert_sentiment_pos     = ?,
                   finbert_sentiment_neutral = ?,
                   finbert_sentiment_neg     = ?
             WHERE id = ?
            """,
            [(p_pos, p_neu, p_neg, _id)
             for (p_pos, p_neu, p_neg), _id in zip(probs, ids)]
        )
        conn.commit()

        processed += len(batch)
        last_id    = ids[-1]

        # Quick batch‑level summary
        avg_pos = sum(p[0] for p in probs) / len(probs)
        avg_neu = sum(p[1] for p in probs) / len(probs)
        avg_neg = sum(p[2] for p in probs) / len(probs)
        print(
            f"✅  {processed}/{total} (last id {last_id}) – "
            f"avg pos {avg_pos:.4f}, neu {avg_neu:.4f}, neg {avg_neg:.4f}"
        )

        # Free GPU memory between big batches
        if device.type == "cuda":
            torch.cuda.empty_cache()

    conn.close()
    print(f"🎯  Finished: scored {processed} rows.")

# --------------------------------------------------------------------------- #
def add_finbert_sentiment_final(db_path, table_name):
    """Create/update `finbert_sentiment_final` = pos − neg."""
    conn, cursor = sqlite3.connect(db_path), None
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = {c[1] for c in cursor.fetchall()}
    if "finbert_sentiment_final" not in cols:
        cursor.execute(f"""
            ALTER TABLE {table_name}
            ADD COLUMN finbert_sentiment_final REAL
        """)
        print("✅  Added column 'finbert_sentiment_final'")
    else:
        print("⚠️  Column already exists – overwriting values")

    cursor.execute(f"""
        UPDATE {table_name}
           SET finbert_sentiment_final =
               finbert_sentiment_pos - finbert_sentiment_neg
         WHERE finbert_sentiment_pos IS NOT NULL
           AND finbert_sentiment_neg IS NOT NULL
    """)
    conn.commit()
    conn.close()
    print("📈  Updated 'finbert_sentiment_final'")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FinBERT sentiment scoring with GPU acceleration"
    )
    parser.add_argument("--db-path",  default="data/news.db")
    parser.add_argument("--table-name", default="master0_revamped")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--final", action="store_true",
                        help="Compute finbert_sentiment_final")
    args = parser.parse_args()

    run_resumable_finbert_probs(
        db_path=args.db_path,
        table_name=args.table_name,
        batch_size=args.batch_size
    )

    if args.final:
        add_finbert_sentiment_final(args.db_path, args.table_name)