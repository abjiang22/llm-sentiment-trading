import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect("data/news.db")

# Query the required sentiment columns
query = """
SELECT
    -- LMD (lm_sentiment)
    1.0 * SUM(CASE WHEN CAST(lm_sentiment AS REAL) = -1 THEN 1 ELSE 0 END) / COUNT(lm_sentiment) AS pct_neg1_lm_sentiment,
    1.0 * SUM(CASE WHEN CAST(lm_sentiment AS REAL) =  0 THEN 1 ELSE 0 END) / COUNT(lm_sentiment) AS pct_0_lm_sentiment,
    1.0 * SUM(CASE WHEN CAST(lm_sentiment AS REAL) =  1 THEN 1 ELSE 0 END) / COUNT(lm_sentiment) AS pct_1_lm_sentiment,

    -- FinBERT
    1.0 * SUM(CASE WHEN CAST(finbert_sentiment AS REAL) = -1 THEN 1 ELSE 0 END) / COUNT(finbert_sentiment) AS pct_neg1_finbert_sentiment,
    1.0 * SUM(CASE WHEN CAST(finbert_sentiment AS REAL) =  0 THEN 1 ELSE 0 END) / COUNT(finbert_sentiment) AS pct_0_finbert_sentiment,
    1.0 * SUM(CASE WHEN CAST(finbert_sentiment AS REAL) =  1 THEN 1 ELSE 0 END) / COUNT(finbert_sentiment) AS pct_1_finbert_sentiment,

    -- GPT-4o
    1.0 * SUM(CASE WHEN CAST(llm_gpt4o_sentiment AS REAL) = -1 THEN 1 ELSE 0 END) / COUNT(llm_gpt4o_sentiment) AS pct_neg1_llm_gpt4o_sentiment,
    1.0 * SUM(CASE WHEN CAST(llm_gpt4o_sentiment AS REAL) =  0 THEN 1 ELSE 0 END) / COUNT(llm_gpt4o_sentiment) AS pct_0_llm_gpt4o_sentiment,
    1.0 * SUM(CASE WHEN CAST(llm_gpt4o_sentiment AS REAL) =  1 THEN 1 ELSE 0 END) / COUNT(llm_gpt4o_sentiment) AS pct_1_llm_gpt4o_sentiment,

    -- GPT-4.1-mini
    1.0 * SUM(CASE WHEN CAST(llm_gpt41mini_sentiment AS REAL) = -1 THEN 1 ELSE 0 END) / COUNT(llm_gpt41mini_sentiment) AS pct_neg1_llm_gpt41mini_sentiment,
    1.0 * SUM(CASE WHEN CAST(llm_gpt41mini_sentiment AS REAL) =  0 THEN 1 ELSE 0 END) / COUNT(llm_gpt41mini_sentiment) AS pct_0_llm_gpt41mini_sentiment,
    1.0 * SUM(CASE WHEN CAST(llm_gpt41mini_sentiment AS REAL) =  1 THEN 1 ELSE 0 END) / COUNT(llm_gpt41mini_sentiment) AS pct_1_llm_gpt41mini_sentiment,

    -- GPT-4.1-mini-market
    1.0 * SUM(CASE WHEN CAST(llm_gpt41mini_market_sentiment AS REAL) = -1 THEN 1 ELSE 0 END) / COUNT(llm_gpt41mini_market_sentiment) AS pct_neg1_llm_gpt41mini_market_sentiment,
    1.0 * SUM(CASE WHEN CAST(llm_gpt41mini_market_sentiment AS REAL) =  0 THEN 1 ELSE 0 END) / COUNT(llm_gpt41mini_market_sentiment) AS pct_0_llm_gpt41mini_market_sentiment,
    1.0 * SUM(CASE WHEN CAST(llm_gpt41mini_market_sentiment AS REAL) =  1 THEN 1 ELSE 0 END) / COUNT(llm_gpt41mini_market_sentiment) AS pct_1_llm_gpt41mini_market_sentiment

FROM master0;
"""

"""
SELECT
    100.0 * SUM(CASE WHEN ABS(lm_sentiment) <= 0.00001 THEN 1 ELSE 0 END) / COUNT(lm_sentiment) AS pct_near_zero_lm_sentiment,
    100.0 * SUM(CASE WHEN ABS(finbert_sentiment) <= 0.0001 THEN 1 ELSE 0 END) / COUNT(finbert_sentiment) AS pct_near_zero_finbert_sentiment,
    100.0 * SUM(CASE WHEN ABS(llm_gpt4o_sentiment) <= 0.0001 THEN 1 ELSE 0 END) / COUNT(llm_gpt4o_sentiment) AS pct_near_zero_llm_gpt4o_sentiment,
    100.0 * SUM(CASE WHEN ABS(llm_gpt41mini_sentiment) <= 0.0001 THEN 1 ELSE 0 END) / COUNT(llm_gpt41mini_sentiment) AS pct_near_zero_llm_gpt41mini_sentiment,
    100.0 * SUM(CASE WHEN ABS(llm_gpt41mini_market_sentiment) <= 0.0001 THEN 1 ELSE 0 END) / COUNT(llm_gpt41mini_market_sentiment) AS pct_near_zero_llm_gpt41mini_market_sentiment
FROM master0;
"""

# Run the query and load into a DataFrame
avg_df = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# Display the result
print("Average sentiment values:")
print(avg_df.T.rename(columns={0: "Average"}))

#-----------------------------------
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to SQLite
conn = sqlite3.connect("data/news.db")  # Update this if needed

# Read relevant sentiment columns
query = """
SELECT
    lm_sentiment,
    finbert_sentiment,
    llm_gpt4o_sentiment,
    llm_gpt41mini_sentiment,
    llm_gpt41mini_market_sentiment
FROM master0
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Rename with mapping (order doesn't matter)
rename_map = {
    "lm_sentiment": "LMD",
    "finbert_sentiment": "FinBERT",
    "llm_gpt4o_sentiment": "LLM-G",
    "llm_gpt41mini_sentiment": "LLM-F",
    "llm_gpt41mini_market_sentiment": "LLM-M"
}
df = df.rename(columns=rename_map)
# Convert to numeric
df = df.astype(float)

# Set up plot
plt.figure(figsize=(12, 8))

# Plot each distribution
for col in df.columns:
    sns.kdeplot(df[col], label=col, fill=True, alpha=0.4)

# Label plot
plt.xlabel("Sentiment Score")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()