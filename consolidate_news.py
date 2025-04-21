import sqlite3
from table_ops import reindex_table_by_column

def merge_tables_into_master_and_reindex(db_path, tables, master_table="master0", sort_column="published_at"):
    """
    Merges rows from specified tables into master0, avoiding duplicate URLs.
    If master0 does not exist, creates it with id, source, title, published_at, url columns.
    After merging, reindexes master0 by specified column so id is 1...N ordered by that column.

    Parameters:
        db_path (str): Path to the SQLite database.
        tables (list): List of table names to merge into master0.
        master_table (str): Name of the master table (default is 'master0').
        sort_column (str): Column to sort master_table by after merging (default is 'published_at').
    """

    if not tables:
        print("⚠️ No tables provided to merge.")
        return

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    total_rows_added = 0

    try:
        cursor.execute("BEGIN TRANSACTION;")

        # ✅ Check if master_table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (master_table,))
        exists = cursor.fetchone()

        # ✅ If not exists, create master_table
        if not exists:
            print(f"🛠 Creating master table '{master_table}'...")
            cursor.execute(f"""
                CREATE TABLE {master_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    title TEXT,
                    published_at TEXT,
                    url TEXT
                )
            """)

        for table in tables:
            print(f"🔄 Processing table: {table}")

            # Insert new unique rows based on URL
            cursor.execute(f"""
                INSERT INTO {master_table} (source, title, published_at, url)
                SELECT 
                    '{table}' AS source, 
                    title, 
                    published_at, 
                    url
                FROM {table}
                WHERE url NOT IN (SELECT url FROM {master_table})
            """)

            rows_added = cursor.rowcount
            total_rows_added += rows_added
            print(f"   ➡️ {rows_added} rows added from '{table}'.")

        conn.commit()
        print(f"✅ Total {total_rows_added} rows added into '{master_table}' successfully.")

    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to merge tables: {e}")
        conn.close()
        return

    finally:
        conn.close()

    # ✅ Reindex master table after merge
    print(f"⚙️ Reindexing '{master_table}' by '{sort_column}'...")
    reindex_table_by_column(db_path, master_table, sort_column)


tables_to_merge = ['apnews', 'bloomberg', 'businessinsider', 'financialpost', 'fortune', 'guardian', 'nyt', 'time', 'washingtonpost', 'wsj']

merge_tables_into_master_and_reindex(
    db_path="data/news_backup.db",
    tables=tables_to_merge
)