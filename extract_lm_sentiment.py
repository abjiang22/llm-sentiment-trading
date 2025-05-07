#!/usr/bin/env python3
# FOR LOUGHRAN‑MCDONALD SENTIMENT (resumable version)

import re
import sqlite3
import argparse
from pathlib import Path

TOKENS_RE = re.compile(r'\b\w+\b')          # pre‑compiled once

def lm_sentiment_score_normalized(text: str,
                                  pos_lex: set[str],
                                  neg_lex: set[str]) -> float:
    words = TOKENS_RE.findall(text.lower())
    pos = neg = 0
    for w in words:
        if w in pos_lex:
            pos += 1
        elif w in neg_lex:
            neg += 1
    den = pos + neg
    return 0.0 if den == 0 else round((pos - neg) / den, 4)


def run_lm_sentiment_scoring(db_path: str,
                             table_name: str,
                             pos_path: str,
                             neg_path: str,
                             print_every: int = 1_000,
                             batch_size: int = 5_000):

    # --- Load lexicons -------------------------------------------------------
    with open(pos_path) as f:
        pos_words = {ln.strip().lower() for ln in f if ln.strip()}
    with open(neg_path) as f:
        neg_words = {ln.strip().lower() for ln in f if ln.strip()}

    # --- Connect & prepare DB -----------------------------------------------
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # Fast journal settings (safe enough for single‑process work)
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")

        # Ensure target column and helper index exist
        cur.execute(f"PRAGMA table_info({table_name})")
        if "lm_sentiment_score" not in (col[1] for col in cur.fetchall()):
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN lm_sentiment_score REAL")

        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_lm_null
            ON {table_name} (lm_sentiment_score)
            WHERE lm_sentiment_score IS NULL
        """)

        # --- Stream rows in chunks ------------------------------------------
        total = 0
        cur.execute(f"""
            SELECT id, title_clean
            FROM {table_name}
            WHERE title_clean IS NOT NULL
              AND title_clean <> ''
              AND lm_sentiment_score IS NULL
        """)

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break

            updates = [
                (lm_sentiment_score_normalized(title, pos_words, neg_words), row_id)
                for row_id, title in rows
            ]

            cur.executemany(
                f"UPDATE {table_name} SET lm_sentiment_score = ? WHERE id = ?",
                updates
            )
            conn.commit()                      # <-- guarantees resumability
            total += len(updates)

            if total % print_every == 0:
                last_id = updates[-1][1]
                print(f"✅ Updated {total:,} rows (last id {last_id})")

    print(f"🎯 Done: {total:,} new rows scored and saved to '{table_name}'.")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path",     default="data/news.db")
    ap.add_argument("--table-name",  default="master0_revamped")
    ap.add_argument("--pos-path",    default="assets/lm_positive.txt")
    ap.add_argument("--neg-path",    default="assets/lm_negative.txt")
    ap.add_argument("--print-every", type=int, default=1000)
    ap.add_argument("--batch-size",  type=int, default=1000)
    args = ap.parse_args()

    run_lm_sentiment_scoring(
        db_path=args.db_path,
        table_name=args.table_name,
        pos_path=args.pos_path,
        neg_path=args.neg_path,
        print_every=args.print_every,
        batch_size=args.batch_size,
    )