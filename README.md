# cis4980thesis
LLM Sentiment in Trading

Pre-Processing 

Step 1: Extract headlines and S&P500 data
store_headline_data.py
store_snp500_data.py

Step 2: Merge headlines into master table
merge_news.py

Step 3: Process headlines - prepare for sentiment analysis
process_headlines.py

Step 4: Extract sentiment
extract_lm_sentiment.py (run_resumable_finbert)
extract_finbert_sentiment.py
*extract_llm_sentiment.py

Step 5: Aggregate sentiment per trading day
aggregate_sentiment_snp500.py

Models:
https://huggingface.co/microsoft/Phi-4-mini-instruct (3.8B Parameters)
https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (1.5B Parameters)
https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct (3.2B parameters)