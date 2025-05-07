import os
import re
import csv
import time
import sqlite3
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dotenv import load_dotenv
import openai
from openai import OpenAI

# === Configuration ===
FAILED_OUTPUTS_FILE   = "failed_sentiment_rows.csv"
INTER_REQUEST_DELAY   = 0.1    # seconds between calls
MAX_RETRIES           = 5
BASE_BACKOFF_DELAY    = 2.0    # initial backoff in seconds

PROMPT_TEMPLATES = {
    "zero_shot": """
You are given a news headline. Your task is to output a single float between -1 and 1 that represents the sentiment of the headline:
-1 = very negative, 0 = neutral, 1 = very positive.

Rules:
Output only the float, with no words, labels, or explanations.
If the sentiment is mixed but leans positive, output a number between 0 and 1.
If the sentiment is mixed but leans negative, output a number between -1 and 0.
If no sentiment is expressed, output a number close to 0.

Headline: "{headline}"
""".strip()
}

NUMBER_PATTERN = re.compile(r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$")

# Global client placeholder for workers
client: OpenAI

def _init_worker():
    """Initialize a fresh OpenAI client in each worker."""
    global client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

def format_prompt(headline: str, template_key="zero_shot") -> str:
    tmpl = PROMPT_TEMPLATES.get(template_key, PROMPT_TEMPLATES["zero_shot"])
    return tmpl.format(headline=headline)

def log_failed_row(row_id, headline, response_text):
    """Append a failure entry to the CSV."""
    with open(FAILED_OUTPUTS_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([row_id, headline, response_text])

def call_chatgpt(args):
    prompt, row_id, headline = args
    # small sleep to pace requests
    time.sleep(INTER_REQUEST_DELAY)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )
            content = resp.choices[0].message.content.strip()
            m = NUMBER_PATTERN.fullmatch(content)
            if not m:
                log_failed_row(row_id, headline, content)
                return None
            score = float(m.group(1))
            return max(-1.0, min(1.0, score))

        except Exception as e:
            err = str(e).lower()
            # detect rate‑limit by status code or message text
            if "429" in err or "rate limit" in err:
                backoff = BASE_BACKOFF_DELAY * (2 ** (attempt - 1))
                print(f"⚠️ Rate‑limit detected on row {row_id} (try {attempt}/{MAX_RETRIES}); sleeping {backoff:.1f}s")
                time.sleep(backoff)
                continue
            # other errors: log and bail
            print(f"⚠️ Error on row {row_id}: {e}")
            log_failed_row(row_id, headline, f"ERROR: {e}")
            return None

    # if we exit loop, retries exhausted
    log_failed_row(row_id, headline, "RATE_LIMIT_EXHAUSTED")
    return None

def generate_sentiment_scores(ids, headlines, template_key="zero_shot"):
    prompts = [(format_prompt(h, template_key), _id, h) for _id, h in zip(ids, headlines)]
    num_workers = max(1, cpu_count() // 2)
    with Pool(processes=num_workers, initializer=_init_worker) as pool:
        return pool.map(call_chatgpt, prompts)

def run_resumable_llm_sentiment_prompt(
    db_path="data/news.db",
    table_name="master0",
    batch_size=64,
    template_key="zero_shot"
):
    load_dotenv()

    # Initialize failure CSV header
    if not Path(FAILED_OUTPUTS_FILE).exists():
        with open(FAILED_OUTPUTS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id", "headline", "response"])

    # Connect to SQLite
    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ensure sentiment column exists
    sentiment_col = f"llm_sentiment_score_gpt4o_{template_key}"
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cursor.fetchall()}
    if sentiment_col not in existing:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {sentiment_col} REAL")
        conn.commit()

    # Fetch unsentenced rows
    cursor.execute(f"""
        SELECT id, title_clean
          FROM {table_name}
         WHERE {sentiment_col} IS NULL
    """)
    rows  = cursor.fetchall()
    total = len(rows)
    print(f"📊 To score: {total} headlines")

    processed  = 0
    start_time = time.time()

    for start in range(0, total, batch_size):
        batch   = rows[start : start + batch_size]
        ids, txt = zip(*batch)
        headlines = [t or "" for t in txt]

        scores  = generate_sentiment_scores(ids, headlines, template_key)
        updates = [(s, _id) for s, _id in zip(scores, ids) if s is not None]

        cursor.executemany(
            f"UPDATE {table_name} SET {sentiment_col} = ? WHERE id = ?",
            updates
        )
        conn.commit()

        processed += len(batch)
        elapsed   = time.time() - start_time
        rate      = processed / elapsed * 60
        valid     = [s for s in scores if s is not None]
        avg_score = (sum(valid) / len(valid)) if valid else 0.0

        print(f"✅ Processed {processed}/{total}. Batch avg: {avg_score:.4f}")
        print(f"⚡ Rate: {rate:.1f} headlines/min")

    conn.close()
    print(f"🎯 Completed: Scored {processed} headlines.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPT‑4o LLM sentiment scoring with rate‑limit handling")
    parser.add_argument("--db_path",    default="data/news.db")
    parser.add_argument("--table_name", default="master0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--template_key",
        choices=list(PROMPT_TEMPLATES.keys()),
        default="zero_shot"
    )
    args = parser.parse_args()

    run_resumable_llm_sentiment_prompt(
        db_path=args.db_path,
        table_name=args.table_name,
        batch_size=args.batch_size,
        template_key=args.template_key
    )