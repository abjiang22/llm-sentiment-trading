#FOR LARGE LANGUAGE MODEL SENTIMENT ANALYSIS - not to be confused with LM
import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_TOKEN"))
# === Load environment variables ===
load_dotenv(override=True)
NEWS_DB_PATH = os.getenv('NEWS_DB_PATH')

PROMPT_TEMPLATES = {
    "zero_shot": """
You are given a news headline, and your task is to output a float between -1 and 1 that represents the overall sentiment of the headline:
1 = extremely positive sentiment
-1 = extremely negative sentiment
0 = neutral or no clear sentiment

Rules:
Output only the float (no words, labels, or explanations).
If the sentiment is mixed but leans positive, output a value between 0 and 1.
If the sentiment is mixed but leans negative, output a value between -1 and 0.
If no sentiment is expressed, output a value close to 0.

Headline to score: "{headline}"
""".strip(),
    "few_shot": """
You are given a news headline, and your task is to output a float between -1 and 1 that represents the overall sentiment of the headline:
1 = extremely positive sentiment
-1 = extremely negative sentiment
0 = neutral or no clear sentiment

Rules:
Output only the float (no words, labels, or explanations).
If the sentiment is mixed but leans positive, output a value between 0 and 1.
If the sentiment is mixed but leans negative, output a value between -1 and 0.
If no sentiment is expressed, output a value close to 0.

Examples:
"Japan stocks plunge, other Asia markets fall after US losses" → -1
"Today in History" → 0
"Dow closes up more than 1,000 points in best day for Wall Street in 10 years as stocks rally back from Christmas Eve beating" → 1

Headline to score: "{headline}"
""".strip(),
    "us_economy": """
You are given a news headline. Your task is to output a single float between -1 and 1 representing the headline’s sentiment toward the U.S. economy:
1 = extremely positive sentiment toward the U.S. economy
-1 = extremely negative sentiment toward the U.S. economy
0 = neutral, no clear sentiment, or not relevant to the U.S. economy

Rules:
Output only the float (no words, labels, or explanations).
If the sentiment is mixed but leans positive, output a value between 0 and 1.
If the sentiment is mixed but leans negative, output a value between -1 and 0.
If no sentiment is expressed or the headline is unrelated to the U.S. economy, output a value close to 0.

Examples:
"Japan stocks plunge, other Asia markets fall after US losses" → -1
"Today in History" → 0
"Dow closes up more than 1,000 points in best day for Wall Street in 10 years as stocks rally back from Christmas Eve beating" → 1

Headline to score: "{headline}"
""".strip(),
    "sp500_sentiment": """
You are given a news headline published before the next U.S. trading day. Your task is to output a single float between -1 and 1 representing the headline’s predicted sentiment toward the movement of the S&P 500 index on the upcoming trading day:

1 = extremely positive sentiment toward the S&P 500 (expecting a price increase)
-1 = extremely negative sentiment toward the S&P 500 (expecting a price decrease)
0 = neutral, no clear sentiment, or unrelated to the S&P 500 or broader U.S. market

Rules:
Output only the float (no words, labels, or explanations).
If the sentiment is mixed but leans positive, output a value between 0 and 1.
If the sentiment is mixed but leans negative, output a value between -1 and 0.
If no sentiment is expressed or the headline is unrelated, output a value close to 0.

Examples:
"Stock markets tumble as investors pull back from American assets" → -1
"Today in History" → 0
"The Wall Street strategist who nailed the stock market's recent mega-rallies sees a 10-15% jump in the coming months" → 1

Headline to score: "{headline}"
""".strip()
}

# === Define Available Models ===
MODELS = {
    "llama3": "meta-llama/Llama-3.2-3B-Instruct",
    "deepseek_r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "phi4": "microsoft/Phi-4-mini-instruct"
}

# === Model Functions ===
def load_model(model_key="llama3"):
    model_name = MODELS.get(model_key)
    if not model_name:
        raise ValueError(f"Model key '{model_key}' not found. Available: {list(MODELS.keys())}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return tokenizer, model

def format_prompt(headline, template_key="zero_shot"):
    template = PROMPT_TEMPLATES.get(template_key, PROMPT_TEMPLATES["zero_shot"])
    return template.format(headline=headline)

def generate_sentiment_score(tokenizer, model, texts, template_key="zero_shot"):
    prompts = [format_prompt(text, template_key) for text in texts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    results = []
    for output in decoded:
        try:
            num = float(output.strip().splitlines()[-1])
            num = max(-1.0, min(1.0, num))
        except Exception:
            num = 0.0
        results.append(num)
    return results

def run_resumable_llm_sentiment_prompt(
    model_key="llama3",
    db_path=NEWS_DB_PATH,
    table_name="master0",
    batch_size=8,
    template_key="zero_shot"
):
    tokenizer, model = load_model(model_key)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cursor.fetchall()}
    if "llama_sentiment_score" not in existing:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN llama_sentiment_score REAL")
    conn.commit()

    cursor.execute(f"""
        SELECT id, title_clean
          FROM {table_name}
         WHERE llama_sentiment_score IS NULL
    """)
    rows = cursor.fetchall()
    total = len(rows)
    print(f"📊 To score: {total} headlines")

    processed = 0

    for start in range(0, total, batch_size):
        batch = rows[start:start+batch_size]
        ids, texts = zip(*batch)
        texts = [t or "" for t in texts]

        scores = generate_sentiment_score(tokenizer, model, texts, template_key)

        updates = [(score, _id) for score, _id in zip(scores, ids)]
        cursor.executemany(
            f"""
            UPDATE {table_name}
               SET llama_sentiment_score = ?
             WHERE id = ?
            """,
            updates
        )
        conn.commit()

        processed += len(batch)

        last_id_in_batch = ids[-1]
        avg_score = sum(scores) / len(scores)

        print(f"✅ Up to row_id ({last_id_in_batch}/{total}) updated. Batch avg score: {avg_score:.4f}")

    conn.close()
    print(f"🎯 Completed: scored {processed} rows.")

if __name__ == "__main__":
    run_resumable_llm_sentiment_prompt(
    model_key="deepseek_r1",
    db_path=NEWS_DB_PATH,
    table_name="master0",
    batch_size=8,
    template_key="zero_shot")