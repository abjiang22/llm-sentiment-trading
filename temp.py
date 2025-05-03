import sqlite3

import sqlite3

def create_revamped_table():
    # Connect to the target DB (where we create the new table)
    conn_main = sqlite3.connect("data/news_5_3_NEW_copy.db")
    cursor_main = conn_main.cursor()

    # Attach the source DB (which has the extra sentiment columns)
    cursor_main.execute("ATTACH DATABASE 'data/news_5_3_CORRECT_DATASET.db' AS news_extra")

    # Drop existing table if it exists
    cursor_main.execute("DROP TABLE IF EXISTS master0_revamped")

    # Create the new table by joining on title_clean and pulling sentiment fields
    cursor_main.execute("""
        CREATE TABLE master0_revamped AS
        SELECT 
            a.*, 
            b.lm_sentiment_score,
            b.finbert_sentiment_pos,
            b.finbert_sentiment_neutral,
            b.finbert_sentiment_neg,
            b.finbert_sentiment_final,
            b.llm_sentiment_score_gpt4o_zero_shot
        FROM master0 a
        LEFT JOIN news_extra.master0_revamped b
        ON a.title_clean = b.title_clean
    """)

    # Detach the extra DB
    cursor_main.execute("DETACH DATABASE news_extra")

    conn_main.commit()
    conn_main.close()

    print("✅ Created master0_revamped with sentiment columns merged from news.db.")

create_revamped_table()


"""

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
"""

