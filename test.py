import sqlite3
import pandas as pd


# Connect to database
conn = sqlite3.connect("data/news.db")
cursor = conn.cursor()

# Load 100 rows of the table into a DataFrame
df = pd.read_sql_query(f"SELECT * FROM master0 LIMIT 100", conn)
for row in df.itertuples():
    print(float(row.finbert_sentiment_neg) + float(row.finbert_sentiment_pos) + float(row.finbert_sentiment_neutral))

