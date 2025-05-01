import sqlite3
import re
import pandas as pd
from datetime import time as dtime
from table_ops import reindex_table_by_column

def exclude_market_hours(news_db_path="data/news.db", news_table_name="master0", snp500_db_path="data/snp500.db", snp500_table_name="snp500"):
    # Connect to databases
    with sqlite3.connect(news_db_path) as news_conn, sqlite3.connect(snp500_db_path) as snp_conn:
        # Load trading dates
        trading_days = pd.read_sql(f"SELECT trade_date FROM {snp500_table_name}", snp_conn)
        trading_days['trade_date'] = pd.to_datetime(trading_days['trade_date']).dt.date
        trading_days_set = set(trading_days['trade_date'])

        # Load unfiltered news
        df = pd.read_sql(f"SELECT * FROM {news_table_name}", news_conn)
        df['published_at'] = pd.to_datetime(df['published_at'])
        df['pub_date'] = df['published_at'].dt.date
        df['pub_time'] = df['published_at'].dt.time

        market_open = dtime(9, 30)
        market_close = dtime(16, 0)

        # Exclude rows
        filtered_df = df[~(
            (df['pub_date'].isin(trading_days_set)) &
            (df['pub_time'] >= market_open) & (df['pub_time'] <= market_close)
        )]

        # Save filtered table
        filtered_df.drop(columns=['pub_date', 'pub_time']).to_sql(f"{news_table_name}", news_conn, if_exists="replace", index=False)

    print(f"Filtered dataset saved to f'{news_table_name}'. Rows kept: {len(filtered_df)}")

def remove_duplicates(db_path: str, table_name: str, group_by_cols: list):
    """Remove rows which have the same values in group_by_cols, keeping the first occurrence."""
    if not group_by_cols:
        raise ValueError("group_by_cols must contain at least one column name.")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    group_cols_str = ", ".join(group_by_cols)
    temp_table = f"{table_name}_dedup_temp"

    # === Step 1: Get original row count ===
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    original_count = cursor.fetchone()[0]

    # === Step 2: Create deduplicated temp table ===
    cursor.execute(f"DROP TABLE IF EXISTS {temp_table}")
    cursor.execute(f"""
        CREATE TABLE {temp_table} AS
        SELECT *
        FROM {table_name}
        WHERE ROWID IN (
            SELECT MIN(ROWID)
            FROM {table_name}
            GROUP BY {group_cols_str}
        )
    """)

    # === Step 3: Get deduplicated row count and report deletions ===
    cursor.execute(f"SELECT COUNT(*) FROM {temp_table}")
    dedup_count = cursor.fetchone()[0]
    deleted_count = original_count - dedup_count
    print(f"Deleted {deleted_count} duplicate row(s) from '{table_name}'.")

    # === Step 4: Replace original table with deduplicated one ===
    cursor.execute(f"DROP TABLE {table_name}")
    cursor.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")

    conn.commit()
    conn.close()

def clean_titles(db_path="data/news.db", table_name="master0"):
    """Ensure title_clean column exists and populate it with cleaned text from title column."""
    def clean_text(text):
        text = re.sub(r"[^\w\s]", "", text)     # Remove punctuation
        text = re.sub(r"\s+", " ", text)        # Normalize spaces
        return text.strip().lower()             # Trim and lowercase

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Add title_clean column if not exists
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = [row[1] for row in cursor.fetchall()]
    if "title_clean" not in existing_columns:
        print(f"Adding 'title_clean' column to {table_name}...")
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN title_clean TEXT")

    # Fetch rows with non-null titles
    cursor.execute(f"SELECT id, title FROM {table_name} WHERE title IS NOT NULL")
    rows = cursor.fetchall()

    for row_id, title in rows:
        if title:
            cleaned = clean_text(title)
            cursor.execute(
                f"UPDATE {table_name} SET title_clean = ? WHERE id = ?",
                (cleaned, row_id)
            )

    conn.commit()
    conn.close()
    print(f"✅ title_clean column populated for table '{table_name}'.")

def process_headlines(news_db_path="data/news.db", news_table_name="master0", snp500_db_path="data/snp500.db", snp500_table_name="snp500", group_by_cols=["title"], index_col="published_at"):
    exclude_market_hours(news_db_path, news_table_name, snp500_db_path, snp500_table_name)
    remove_duplicates(news_db_path, news_table_name, group_by_cols)
    clean_titles(news_db_path, news_table_name)
    reindex_table_by_column(news_db_path, news_table_name, index_col)

#remove_duplicates("data/news.db", "master0", ["title"])

#exclude_market_hours("data/news.db", "data/snp500.db")
#clean_titles(db_path="data/news.db", table_name="master0")