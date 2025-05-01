import sqlite3
import os
import shutil

# === Step 2: Connect to source and target DBs ===
src_conn = sqlite3.connect("data/news_4_30.db")
dst_conn = sqlite3.connect("data/news.db")

src_cursor = src_conn.cursor()
dst_cursor = dst_conn.cursor()

# === Step 3: Get the schema for master0 ===
src_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='master0'")
create_table_sql = src_cursor.fetchone()

if create_table_sql is None:
    raise ValueError("Table 'master0' does not exist in the source database.")

# === Step 4: Create the table in the destination ===
dst_cursor.execute(create_table_sql[0])

# === Step 5: Copy data from master0 ===
src_cursor.execute("SELECT * FROM master0")
rows = src_cursor.fetchall()

# Get column count to construct insert statement
col_count = len(src_cursor.description)
placeholders = ", ".join("?" * col_count)
dst_cursor.executemany(f"INSERT INTO master0 VALUES ({placeholders})", rows)

# === Step 6: Commit and close ===
dst_conn.commit()
src_conn.close()
dst_conn.close()

print("Successfully copied table 'master0' to news.db.")
