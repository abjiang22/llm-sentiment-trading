import re
import sqlite3
import os
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv(override=True)
NEWS_DB_PATH = os.getenv('NEWS_DB_PATH')

def lm_sentiment_score_normalized(text, pos_lexicon, neg_lexicon):
    words = re.findall(r'\b\w+\b', text.lower())
    pos = sum(1 for w in words if w in pos_lexicon)
    neg = sum(1 for w in words if w in neg_lexicon)
    score = (pos - neg) / (pos + neg + 1e-5)
    return round(score, 4)

def run_lm_sentiment_scoring(db_path=NEWS_DB_PATH, table_name="master0",
                              pos_path="assets/lm_positive.txt", neg_path="assets/lm_negative.txt",
                              print_every=10000):
    # Load lexicons
    with open(pos_path) as f:
        positive_words = set(line.strip().lower() for line in f if line.strip())
    with open(neg_path) as f:
        negative_words = set(line.strip().lower() for line in f if line.strip())

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Add lm_sentiment_score column if missing
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    if "lm_sentiment_score" not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN lm_sentiment_score REAL")
        conn.commit()

    # Fetch titles to score
    cursor.execute(f"SELECT id, title_clean FROM {table_name}")
    rows = cursor.fetchall()
    total_rows = len(rows)
    print(f"📊 Total rows to score: {total_rows}")

    for idx, (row_id, title) in enumerate(rows, start=1):
        if title:
            score = lm_sentiment_score_normalized(title, positive_words, negative_words)
            cursor.execute(
                f"UPDATE {table_name} SET lm_sentiment_score = ? WHERE id = ?",
                (score, row_id)
            )

        # Print update every N rows
        if idx % print_every == 0 or idx == total_rows:
            print(f"✅ Up to row_id {row_id} ({idx}/{total_rows}) updated...")

    conn.commit()
    conn.close()
    print(f"🎯 Completed: {total_rows} rows scored and saved to '{table_name}'.")

# Example call
run_lm_sentiment_scoring(db_path="data/news_backup.db")