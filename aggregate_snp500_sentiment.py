import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import pytz

def connect_databases(snp_db_path, news_db_path):
    conn_snp = sqlite3.connect(snp_db_path)
    conn_news = sqlite3.connect(news_db_path)
    return conn_snp, conn_news

def ensure_column_exists(cursor, snp_table, sentiment_label_name):
    column_name = f"avg_sentiment_{sentiment_label_name}"
    cursor.execute(f"PRAGMA table_info({snp_table})")
    columns = [row[1] for row in cursor.fetchall()]
    if column_name not in columns:
        cursor.execute(f"ALTER TABLE {snp_table} ADD COLUMN {column_name} REAL")

def load_data(conn_snp, conn_news, snp_table, news_table, sentiment_column):
    snp_df = pd.read_sql_query(f"SELECT * FROM {snp_table} ORDER BY trade_date", conn_snp)
    news_df = pd.read_sql_query(
        f"SELECT published_at, {sentiment_column} FROM {news_table} WHERE {sentiment_column} IS NOT NULL",
        conn_news
    )
    eastern = pytz.timezone('US/Eastern')
    news_df['published_at'] = pd.to_datetime(news_df['published_at'], utc=True).dt.tz_convert(eastern)
    news_df[sentiment_column] = pd.to_numeric(news_df[sentiment_column], errors="coerce")
    return snp_df, news_df

def calculate_and_update_sentiment(snp_df, news_df, snp_table, sentiment_column, sentiment_label_name, cursor):
    news_df['published_at_utc'] = pd.to_datetime(news_df['published_at'], utc=True)
    eastern = pytz.timezone('US/Eastern')
    utc = pytz.UTC
    col = f"avg_sentiment_{sentiment_label_name}"

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

        print(f"[{curr['trade_date']}] matched {len(window)} rows → avg = {avg}")

        cursor.execute(
            f"UPDATE {snp_table} SET {col} = ? WHERE trade_date = ?",
            (avg, curr['trade_date'])
        )
        print("  ▶ sqlite3 rowcount:", cursor.rowcount)

    cursor.connection.commit()

def aggregate_snp500_sentiment(
    sentiment_column: str,
    sentiment_label_name: str,
    snp_db_path: str,
    news_db_path: str,
    snp_table: str = "snp500",
    news_table: str = "master0"
):
    conn_snp, conn_news = connect_databases(snp_db_path, news_db_path)
    cursor_snp = conn_snp.cursor()

    ensure_column_exists(cursor_snp, snp_table, sentiment_label_name)
    snp_df, news_df = load_data(conn_snp, conn_news, snp_table, news_table, sentiment_column)
    calculate_and_update_sentiment(snp_df, news_df, snp_table, sentiment_column, sentiment_label_name, cursor_snp)

    conn_snp.commit()
    conn_snp.close()
    conn_news.close()

    print(f"✅ Column avg_sentiment_{sentiment_label_name} added and populated in '{snp_table}' using '{sentiment_column}' from '{news_table}'.")

# Example usage
aggregate_snp500_sentiment(
    sentiment_column="finbert_sentiment_final",
    sentiment_label_name="finbert",
    snp_db_path="data/snp500.db",
    snp_table="snp500",
    news_db_path="data/news.db",
    news_table="master0"
)