import sqlite3
import pandas as pd

def filter_and_reindex_to_new_table(
    db_path: str,
    original_table: str,
    new_table: str = "filtered_articles"
):
    conn = sqlite3.connect(db_path)

    # Load data from the original table
    df = pd.read_sql_query(f"SELECT * FROM {original_table}", conn)

    # Filter by allowed sources
    allowed_sources = {"bloomberg", "financialpost", "businessinsider", "wsj"}
    df_filtered = df[df['source'].isin(allowed_sources)].copy()

    # Sort by published_at and reassign id
    df_filtered = df_filtered.sort_values('published_at').reset_index(drop=True)
    df_filtered['id'] = df_filtered.index + 1  # If you want 1-based indexing

    # Save to a new table
    df_filtered.to_sql(new_table, conn, if_exists='replace', index=False)

    conn.close()
    print(f"Saved filtered data to new table '{new_table}'.")

    return df_filtered


filter_and_reindex_to_new_table(
    db_path="data/news.db",
    original_table="master0_date_filtered_all_news_4_30",
    new_table="master0"
)