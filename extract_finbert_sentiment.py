import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

# === Load FinBERT model/tokenizer once ===
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
model.eval()

# Build lowercase→index map from model config
id2label = model.config.id2label
label_to_idx = {lab.lower(): idx for idx, lab in id2label.items()}
pos_idx = label_to_idx['positive']
neutral_idx = label_to_idx['neutral']
neg_idx = label_to_idx['negative']

def batch_probs(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
    return [
        (float(probs[i, pos_idx]),
         float(probs[i, neutral_idx]),
         float(probs[i, neg_idx]))
        for i in range(len(texts))
    ]

def run_resumable_finbert_probs(
    db_path="/data/news.db",
    table_name="master0",
    batch_size=64
):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Add finbert columns if they don't exist
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cursor.fetchall()}
    for col in ("finbert_sentiment_pos", "finbert_sentiment_neutral", "finbert_sentiment_neg"):
        if col not in existing:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} REAL")
    conn.commit()

    # Fetch rows needing scoring
    cursor.execute(f"""
        SELECT id, title_clean
          FROM {table_name}
         WHERE finbert_sentiment_pos IS NULL
    """)
    rows = cursor.fetchall()
    total = len(rows)
    print(f"📊 To score: {total} headlines")

    processed = 0

    for start in range(0, total, batch_size):
        batch = rows[start:start+batch_size]
        ids, texts = zip(*batch)
        texts = [t or "" for t in texts]

        probs = batch_probs(texts)

        # Bulk update the three columns
        updates = [
            (p_pos, p_neu, p_neg, _id)
            for (p_pos, p_neu, p_neg), _id in zip(probs, ids)
        ]
        cursor.executemany(
            f"""
            UPDATE {table_name}
               SET finbert_sentiment_pos     = ?,
                   finbert_sentiment_neutral = ?,
                   finbert_sentiment_neg     = ?
             WHERE id = ?
            """,
            updates
        )
        conn.commit()

        processed += len(batch)

        last_id_in_batch = ids[-1]

        # Compute and print batch averages
        avg_pos = sum(p[0] for p in probs) / len(probs)
        avg_neu = sum(p[1] for p in probs) / len(probs)
        avg_neg = sum(p[2] for p in probs) / len(probs)
        print(
            f"✅ Completed ({last_id_in_batch}/{total}). "
            f"Batch avg → pos: {avg_pos:.4f}, neu: {avg_neu:.4f}, neg: {avg_neg:.4f}"
        )

    conn.close()
    print(f"🎯 Completed: scored {processed} rows.")

def add_finbert_sentiment_final(db_path, table_name):
    """
    Connects to a SQLite database, computes 'finbert_sentiment_final' 
    (finbert_sentiment_pos - finbert_sentiment_neg), and updates the table.

    Args:
        db_path (str): Path to the SQLite database.
        table_name (str): Name of the table to update.

    Returns:
        None
    """
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the new column already exists
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = {col[1] for col in cursor.fetchall()}
    
    if "finbert_sentiment_final" not in existing_columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN finbert_sentiment_final REAL")
        print("✅ Added column 'finbert_sentiment_final'")
    else:
        print("⚠️ Column 'finbert_sentiment_final' already exists. Overwriting values.")
    
    # Update the table row-by-row
    update_query = f"""
    UPDATE {table_name}
    SET finbert_sentiment_final = finbert_sentiment_pos - finbert_sentiment_neg
    WHERE finbert_sentiment_pos IS NOT NULL AND finbert_sentiment_neg IS NOT NULL
    """
    cursor.execute(update_query)
    
    conn.commit()
    conn.close()
    print("✅ Finished updating 'finbert_sentiment_final' in table.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FinBERT sentiment scoring on a SQLite table.")
    parser.add_argument("--db-path", default="/data/news.db", help="Path to the SQLite database")
    parser.add_argument("--table-name", default="master0", help="Name of the table to process")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--final", action="store_true", help="Also compute finbert_sentiment_final column")

    args = parser.parse_args()

    run_resumable_finbert_probs(args.db_path, args.table_name, args.batch_size)

    if args.final:
        add_finbert_sentiment_final(args.db_path, args.table_name)

"""
run_resumable_finbert_probs(
    db_path=/data/news.db,
    table_name="master0",
    batch_size=64
)
"""

"""
add_finbert_sentiment_final(
    db_path=/data/news.db,
    table_name="master0"
)
"""