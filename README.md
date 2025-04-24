# cis4980thesis
LLM Sentiment in Trading

Pre-Processing 

Step 1: Extract News data
store_guardian_headlines.py
store_newsAPI_headlines.py
store_nyt_headlines.py

Step 2: Consolidate News
consolidate_news.py

Step 3: Clean titles
clean_titles.py

Step 4: Extract sentiment
extract_lm_sentiment.py (run_resumable_finbert)
extract_finbert_sentiment.py
*extract_llm_sentiment.py

Step 5: Sort and Index by Date
table_ops.py (reindex_table_by_column)

Linear Regression:


Models:
https://huggingface.co/microsoft/Phi-4-mini-instruct (3.8B Parameters)

https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (1.5B Parameters)

https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct (3.2B parameters)