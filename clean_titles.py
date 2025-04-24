import sqlite3
import re

def connect_db(db_path="data/news.db"):
    """Connect to SQLite database."""
    conn = sqlite3.connect(db_path)
    return conn, conn.cursor()

def ensure_title_clean_column(cursor, table_name="master0"):
    """Add title_clean column if it doesn't already exist."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    if "title_clean" not in columns:
        print(f"Adding 'title_clean' column to {table_name}...")
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN title_clean TEXT")

def clean_text(text):
    """Clean text by removing punctuation, reducing spaces, and lowering case."""
    cleaned = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    cleaned = re.sub(r"\s+", " ", cleaned)  # Normalize spaces
    cleaned = cleaned.strip().lower()
    return cleaned

def populate_title_clean(db_path="data/news.db", table_name="master0"):
    """Populate the title_clean column with cleaned titles."""
    conn, cursor = connect_db(db_path=db_path)
    ensure_title_clean_column(cursor, table_name)

    cursor.execute(f"SELECT id, title FROM {table_name} WHERE title IS NOT NULL")
    rows = cursor.fetchall()

    for row_id, title in rows:
        if title:
            cleaned_title = clean_text(title)
            cursor.execute(
                f"UPDATE {table_name} SET title_clean = ? WHERE id = ?",
                (cleaned_title, row_id)
            )

    conn.commit()
    conn.close()
    print(f"✅ title_clean column populated for table '{table_name}'.")

# === Entrypoint ===
populate_title_clean(db_path="data/news.db", table_name="master0")