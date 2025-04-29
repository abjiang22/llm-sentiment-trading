import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import pytz

# === CONFIG ===
SNP500_DB_PATH = "data/snp500.db"
NEWS_DB_PATH = "data/news.db"

def connect_databases():
    conn_snp = sqlite3.connect(SNP500_DB_PATH)
    conn_news = sqlite3.connect(NEWS_DB_PATH)
    return conn_snp, conn_news

def ensure_column_exists(cursor, sentiment_name):
    column_name = f"avg_sentiment_{sentiment_name}"
    cursor.execute("PRAGMA table_info(snp500)")
    columns = [row[1] for row in cursor.fetchall()]
    if column_name not in columns:
        cursor.execute(f"ALTER TABLE snp500 ADD COLUMN {column_name} REAL")

def load_data(conn_snp, conn_news, sentiment_column):
    snp_df = pd.read_sql_query("SELECT * FROM snp500 ORDER BY trade_date", conn_snp)
    news_df = pd.read_sql_query(
        f"SELECT published_at, {sentiment_column} FROM master0 WHERE {sentiment_column} IS NOT NULL",
        conn_news
    )
    eastern = pytz.timezone('US/Eastern')
    news_df['published_at'] = pd.to_datetime(news_df['published_at'], utc=True).dt.tz_convert(eastern)
    news_df[sentiment_column] = pd.to_numeric(news_df[sentiment_column], errors="coerce")
    return snp_df, news_df

def calculate_and_update_sentiment(snp_df, news_df, sentiment_column, sentiment_name, cursor):
    news_df['published_at_utc'] = pd.to_datetime(news_df['published_at'], utc=True)
    eastern = pytz.timezone('US/Eastern')
    utc = pytz.UTC
    col = f"avg_sentiment_{sentiment_name}"

    for i in range(1, len(snp_df)):
        curr = snp_df.iloc[i]
        prev = snp_df.iloc[i - 1]

        # build window bounds and convert to UTC
        prev_local = eastern.localize(
            datetime.strptime(prev['trade_date'], "%Y-%m-%d") + timedelta(hours=16)
        )
        curr_local = eastern.localize(
            datetime.strptime(curr['trade_date'], "%Y-%m-%d") + timedelta(hours=9, minutes=30)
        )
        prev_utc = prev_local.astimezone(utc)
        curr_utc = curr_local.astimezone(utc)

        # filter & coerce
        mask = (
            (news_df['published_at_utc'] >= prev_utc) &
            (news_df['published_at_utc'] <= curr_utc)
        )
        window = pd.to_numeric(
            news_df.loc[mask, sentiment_column],
            errors='coerce'
        ).dropna()

        avg = float(window.mean()) if not window.empty else None

        # DEBUG: print what we're about to write
        print(f"[{curr['trade_date']}] matched {len(window)} rows → avg = {avg}")

        # UPDATE by trade_date instead of trade_id
        cursor.execute(
            f"UPDATE snp500 SET {col} = ? WHERE trade_date = ?",
            (avg, curr['trade_date'])
        )
        print("  ▶ sqlite3 rowcount:", cursor.rowcount)

    # commit once after the loop
    cursor.connection.commit()

def build_snp500_sentiment_column(sentiment_column: str, sentiment_name: str):
    conn_snp, conn_news = connect_databases()
    cursor_snp = conn_snp.cursor()

    ensure_column_exists(cursor_snp, sentiment_name)
    snp_df, news_df = load_data(conn_snp, conn_news, sentiment_column)
    calculate_and_update_sentiment(snp_df, news_df, sentiment_column, sentiment_name, cursor_snp)

    conn_snp.commit()
    conn_snp.close()
    conn_news.close()

    print(f"✅ Column avg_sentiment_{sentiment_name} added and populated using '{sentiment_column}'.")

build_snp500_sentiment_column("finbert_sentiment_final", "finbert")