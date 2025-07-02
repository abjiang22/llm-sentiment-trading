import sqlite3
import pandas as pd

# Connect to your database
db_path = "data/news.db"
conn = sqlite3.connect(db_path)

# Query data from the database (adjust table_name to your actual table name)
query = """
SELECT 
    lm_sentiment_score,
    finbert_sentiment_final,
    llm_sentiment_score_gpt4o_zero_shot,
    llm_sentiment_score_gpt41mini_zero_shot,
    llm_sentiment_score_gpt41mini_market_zero_shot
FROM master0
"""

# Load data into a pandas DataFrame
df = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# Compute the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print(correlation_matrix)
