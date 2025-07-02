#!/usr/bin/env python
# async_sentiment.py ––– headline‑level sentiment scoring (async version)
#
# ▸ pip install "openai>=1.13" aiolimiter python-dotenv aiosqlite
# ▸ python async_sentiment.py --db_path data/news_5_7.db --table master0 --rpm 2000

import os, re, csv, json, asyncio, time, argparse, math
from pathlib import Path

import aiosqlite
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
FAILED_OUTPUTS_FILE = "failed_sentiment_rows.csv"
NUMBER_PATTERN      = re.compile(r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$")

PROMPT_TEMPLATES = {
    "zero_shot": """
    You are given a news headline published before the start of the trading day. Your task is to output a single float between -1.00 and 1.00 that represents the financial sentiment of the headline:
    -1.00 = very negative, 0.00 = neutral, 1.00 = very positive.

    Rules:
    Output only the float rounded to the nearest hundredth, with no words, labels, or explanations.
    If the headline is not related to financial markets, or if the financial sentiment is mixed or weak, use values close to 0.00. 
    The closer the number is to 1.00, the more positive the sentiment; the closer to -1.00, the more negative the sentiment.
    Headline: "{headline}"
    """.strip(),
    
    "few_shot": """
    You are given a news headline published before the start of the trading day. Your task is to output a single float between -1.00 and 1.00 that represents the financial sentiment of the headline:
    -1.00 = very negative, 0.00 = neutral, 1.00 = very positive.

    Rules:
    Output only the float rounded to the nearest hundredth, with no words, labels, or explanations.
    If the headline is not related to finance, or if the financial sentiment is mixed or weak, use values close to 0.00. 
    The closer the number is to 1.00, the more positive the sentiment; the closer to -1.00, the more negative the sentiment.

    Examples:
    "Japan stocks plunge, other Asia markets fall after US losses" → -1.00
    "One map shows how each US state generates the GDP of a country" → 0.00
    "Dow closes up more than 1,000 points in best day for Wall Street in 10 years as stocks rally back from Christmas Eve beating" → 1.00
    
    Headline: "{headline}"
    """.strip(),
    "market_zero_shot": """
    You are given a news headline published before the start of the trading day. Your task is to output a single float between -1.00 and 1.00 that represents whether the headline suggests that U.S. financial markets will increase or decrease today: 
    -1.00 = very negative, 0.00 = neutral, 1.00 = very positive.

    Rules:
    Output only the float rounded to the nearest hundredth, with no words, labels, or explanations.
    If the headline is not related to financial markets, or if the signal is weak, mixed, or irrelevant to today’s market movement, use values close to 0.00.
    Use values closer to 1.00 if the headline strongly suggests that U.S. markets will rise today, and values closer to -1.00 if it strongly suggests that U.S. markets will fall today.    
    Headline: "{headline}"
    """.strip()
}

# ── Helper functions ───────────────────────────────────────────────────────────
def format_prompt(headline: str, template_key="zero_shot") -> str:
    return PROMPT_TEMPLATES[template_key].format(headline=headline)

def log_failed_row(row_id, headline, response_text):
    with open(FAILED_OUTPUTS_FILE, "a", newline="", encoding="utf-8") as csvfile:
        csv.writer(csvfile).writerow([row_id, headline, response_text])

# ── Core async worker ──────────────────────────────────────────────────────────
async def score_headline(
    client: AsyncOpenAI,
    limiter: AsyncLimiter,
    row_id: int,
    headline: str,
    prompt_template="zero_shot",
    model="gpt-4.1-mini",
    max_retries=5,
    backoff_base=2.0,
):
    prompt = format_prompt(headline, prompt_template)

    for attempt in range(1, max_retries + 1):
        try:
            async with limiter:          # enforce RPM
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=5,
                    timeout=20,
                )

            content = resp.choices[0].message.content.strip()
            m = NUMBER_PATTERN.fullmatch(content)
            if not m:
                log_failed_row(row_id, headline, content)
                return row_id, None

            score = max(-1.0, min(1.0, float(m.group(1))))
            return row_id, score

        except Exception as e:           # includes 429 / network errors
            if attempt < max_retries:
                delay = backoff_base * (2 ** (attempt - 1))
                print(f"⚠️  row {row_id}: {e} → retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
            else:
                log_failed_row(row_id, headline, f"ERROR: {e}")
                return row_id, None

# ── Main async pipeline ────────────────────────────────────────────────────────
async def run_async_pipeline(
    db_path: str,
    table_name: str,
    batch_size: int,
    rpm_cap: int,
    prompt_template: str,
    model: str,
):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    # Async DB connection
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    cursor = await conn.execute(f"PRAGMA table_info({table_name})")
    cols = {row["name"] async for row in cursor}
    sentiment_col = f"llm_sentiment_score_{model}_{prompt_template}".replace("-", "").replace(".", "")
    if sentiment_col not in cols:
        await conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {sentiment_col} REAL")
        await conn.commit()

    # Headlines still needing scores
    cursor = await conn.execute(
        f"SELECT id, title_clean FROM {table_name} WHERE {sentiment_col} IS NULL"
    )
    rows = await cursor.fetchall()
    total = len(rows)
    print(f"📊  Headlines to score: {total}")

    # Prepare OpenAI client + limiter
    client  = AsyncOpenAI(api_key=api_key)
    limiter = AsyncLimiter(rpm_cap, time_period=60)

    # CSV header
    if not Path(FAILED_OUTPUTS_FILE).exists():
        with open(FAILED_OUTPUTS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id", "headline", "response"])

    start_time  = time.time()
    processed   = 0
    concurrency = math.ceil(rpm_cap * 0.03)      # 3% of target RPM ≈ good pipe‑fill
    sem         = asyncio.Semaphore(concurrency)

    async def bounded_score(row_id, headline):
        async with sem:
            return await score_headline(
                client, limiter, row_id, headline,
                prompt_template, model
            )

    # Chunk rows and fire coroutines
    for i in range(0, total, batch_size):
        chunk = rows[i : i + batch_size]
        ids, heads = zip(*[(r["id"], r["title_clean"] or "") for r in chunk])
        tasks = [asyncio.create_task(bounded_score(_id, h)) for _id, h in zip(ids, heads)]
        results = await asyncio.gather(*tasks)

        # Write back to DB
        updates = [(score, _id) for _id, score in results if score is not None]
        if updates:
            await conn.executemany(
                f"UPDATE {table_name} SET {sentiment_col} = ? WHERE id = ?",
                updates
            )
            await conn.commit()

        processed += len(chunk)
        elapsed   = time.time() - start_time
        rate      = processed / elapsed * 60
        avg_score = (
            sum(score for _, score in results if score is not None) /
            max(1, sum(score is not None for _, score in results))
        )
        print(f"✅ {processed}/{total} done – batch μ={avg_score:.3f} – {rate:,.0f} rows/min")

    await conn.close()
    print("🎯  Completed.")

# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Async GPT sentiment scorer")
    p.add_argument("--db_path",    default="data/news_5_7.db")
    p.add_argument("--table",      default="master0")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Number of DB rows fetched per outer loop")
    p.add_argument("--rpm",        type=int, default=4000,
                   help="Target requests‑per‑minute")
    p.add_argument("--template_key", choices=PROMPT_TEMPLATES.keys(),
                   default="market_zero_shot")
    p.add_argument("--model",      default="gpt-4.1-mini")
    args = p.parse_args()

    asyncio.run(
        run_async_pipeline(
            db_path=args.db_path,
            table_name=args.table,
            batch_size=args.batch_size,
            rpm_cap=args.rpm,
            prompt_template=args.template_key,
            model=args.model,
        )
    )