import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login
import os

# Load environment variables
load_dotenv(override=True)
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# === Prompt Templates ===
PROMPT_TEMPLATES = {
    "zero_shot": """You are given a news headline, and your task is to output a float between -1 and 1 that represents the overall sentiment of the headline:
1 = extremely positive sentiment
-1 = extremely negative sentiment
0 = neutral or no clear sentiment

Rules:
Output only the float (no words, labels, or explanations).
If the sentiment is mixed but leans positive, output a value between 0 and 1.
If the sentiment is mixed but leans negative, output a value between -1 and 0.
If no sentiment is expressed, output a value close to 0.

Headline to score: "{headline}"
""",
    # Add other templates as needed
}

# === Available Models ===
MODELS = {
    "llama3": "meta-llama/Llama-3.2-3B-Instruct",
    "deepseek_r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "phi4": "microsoft/Phi-4-mini-instruct",
    "mistral": "mistral/Mistral-7B-Instruct"
}

def load_model(model_key):
    model_name = MODELS.get(model_key)
    if not model_name:
        raise ValueError(f"Invalid model_key '{model_key}'. Options: {list(MODELS.keys())}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return tokenizer, model

def format_prompt(headline, template_key):
    template = PROMPT_TEMPLATES.get(template_key)
    if not template:
        raise ValueError(f"Invalid template_key '{template_key}'. Options: {list(PROMPT_TEMPLATES.keys())}")
    return template.format(headline=headline)

def extract_score(output):
    match = re.search(r"(-?\d+\.\d+)", output)
    return max(-1.0, min(1.0, float(match.group(1)))) if match else 0.0

def test_llm_sentiment(model_key, headline, template_key="zero_shot"):
    tokenizer, model = load_model(model_key)
    prompt = format_prompt(headline, template_key)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    score = extract_score(decoded)
    print(score)  # Output ONLY the score

# === Example usage ===
if __name__ == "__main__":
    test_llm_sentiment(
        model_key="phi4",
        headline="Markets rally after Fed signals rate cut",
        template_key="zero_shot"
    )
