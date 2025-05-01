import sqlite3

def merge_tables_into_target(db_path, tables, target_table):
    """
    Merges rows from specified tables into a target table, avoiding duplicate URLs.
    If the target table does not exist, creates it with id, source, title, published_at, url columns.

    Parameters:
        db_path (str): Path to the SQLite database.
        tables (list): List of table names to merge.
        target_table (str): Name of the destination (target) table to merge into.
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

        # Check if target table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (target_table,))
        exists = cursor.fetchone()

        # Create target table if it doesn't exist
        if not exists:
            print(f"🛠 Creating target table '{target_table}'...")
            cursor.execute(f"""
                CREATE TABLE {target_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    title TEXT,
                    published_at TEXT,
                    url TEXT
                )
            """)

        # Merge in rows from each source table
        for table in tables:
            print(f"🔄 Processing table: {table}")
            cursor.execute(f"""
                INSERT INTO {target_table} (source, title, published_at, url)
                SELECT 
                    '{table}' AS source, 
                    title, 
                    published_at, 
                    url
                FROM {table}
                WHERE url NOT IN (SELECT url FROM {target_table})
            """)
            rows_added = cursor.rowcount
            total_rows_added += rows_added
            print(f"   ➡️ {rows_added} rows added from '{table}'.")

        conn.commit()
        print(f"✅ Total {total_rows_added} rows added into '{target_table}' successfully.")

    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to merge tables: {e}")

    finally:
        conn.close()

tables_to_merge = ['bloomberg', 'businessinsider', 'financialpost', 'wsj']
merge_tables_into_target(db_path="data/news.db", tables=tables_to_merge, target_table="master0")