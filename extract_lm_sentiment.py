import re
import sqlite3
import argparse

def lm_sentiment_score_normalized(text, pos_lexicon, neg_lexicon):
    words = re.findall(r'\b\w+\b', text.lower())
    pos = sum(1 for w in words if w in pos_lexicon)
    neg = sum(1 for w in words if w in neg_lexicon)
    score = (pos - neg) / (pos + neg + 1e-5)
    return round(score, 4)

def run_lm_sentiment_scoring(db_path="data/news.db", table_name="master0",
                              pos_path="assets/lm_positive.txt", neg_path="assets/lm_negative.txt",
                              print_every=10000, batch_size=5000):

    # Load lexicons
    with open(pos_path) as f:
        positive_words = set(line.strip().lower() for line in f if line.strip())
    with open(neg_path) as f:
        negative_words = set(line.strip().lower() for line in f if line.strip())

    # Connect to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ensure lm_sentiment_score column exists
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if "lm_sentiment_score" not in existing_cols:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN lm_sentiment_score REAL")

    # Use resumable query: only process rows that haven't been scored
    cursor.execute(f"""
        SELECT id, title_clean FROM {table_name}
        WHERE title_clean IS NOT NULL AND lm_sentiment_score IS NULL
    """)
    rows = cursor

    # Begin transaction
    conn.execute("BEGIN TRANSACTION")

    updates = []
    total = 0
    for idx, (row_id, title) in enumerate(rows, start=1):
        score = lm_sentiment_score_normalized(title, positive_words, negative_words)
        updates.append((score, row_id))
        total += 1

        if len(updates) >= batch_size:
            cursor.executemany(
                f"UPDATE {table_name} SET lm_sentiment_score = ? WHERE id = ?",
                updates
            )
            updates.clear()
            if total % print_every < batch_size:
                print(f"✅ Updated {total} rows so far (last row_id: {row_id})...")

    if updates:
        cursor.executemany(
            f"UPDATE {table_name} SET lm_sentiment_score = ? WHERE id = ?",
            updates
        )

    conn.commit()
    conn.close()
    print(f"🎯 Done: {total} new rows scored and updated in '{table_name}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="data/news.db")
    parser.add_argument("--table-name", default="master0_revamped")
    parser.add_argument("--pos-path", default="assets/lm_positive.txt")
    parser.add_argument("--neg-path", default="assets/lm_negative.txt")
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    run_lm_sentiment_scoring(
        db_path=args.db_path,
        table_name=args.table_name,
        pos_path=args.pos_path,
        neg_path=args.neg_path,
        print_every=args.print_every,
        batch_size=args.batch_size
    )