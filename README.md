# CIS4980 Thesis: LLM Sentiment Analysis in Trading

This repository contains the implementation of a thesis project exploring the use of Large Language Models (LLMs) for sentiment analysis in financial trading applications.

## Overview

The project analyzes news headlines and their sentiment correlation with S&P 500 market movements using multiple sentiment analysis approaches:
- Loughran-McDonald Dictionary (LMD)
- FinBERT
- GPT-4.1-mini (simple prompt)
- GPT-4.1-mini (market prompt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cis4980thesis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file with your API keys
   - Add `HUGGINGFACE_TOKEN` for model access
   - Add `OPENAI_API_KEY` for GPT models

## Data Processing Pipeline

### Step 1: Data Collection
- `store_headlines.py` - Extract news headlines from various sources
- `store_snp500.py` - Download S&P 500 historical data

### Step 2: Data Integration
- `merge_news.py` - Merge headlines into master table

### Step 3: Text Preprocessing
- `process_headlines.py` - Clean and prepare headlines for sentiment analysis

### Step 4: Sentiment Extraction
- `extract_lm_sentiment.py` - Loughran-McDonald dictionary-based sentiment
- `extract_finbert_sentiment.py` - FinBERT model sentiment analysis
- `extract_llm_sentiment_GPT.py` - OpenAI GPT models sentiment analysis

### Step 5: Aggregation
- `aggregate_snp500_sentiment.py` - Aggregate daily sentiment scores

### Step 6: Build a Model
- `logistic_regression_final.py` - Logistic regression model to predict S&P 500 increase from sentiment scores
- `random_forest_final.py` - Random forest model to predict S&P 500 increase from sentiment scores
- `xg_boost.py` - XG Boost model to predict S&P 500 increase from sentiment scores

### Machine Learning Models
- Logistic Regression
- Random Forest
- XGBoost
- Attention-based Neural Networks

## Project Structure

```
cis4980thesis/
├── assets/                 # Dictionary files and setup scripts
├── models/                 # ML model implementations
├── model_weights/          # Trained model weights
├── predictions/            # Model prediction outputs
├── *.py                   # Main processing scripts
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Usage

1. Run the data collection scripts in order
2. Execute sentiment extraction for desired models
3. Train and evaluate ML models using the scripts in `models/`
4. Analyze results using `correlation.py` and `plots.py`

## License

This project is part of a thesis submission for CIS4980.