import requests
import sqlite3
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv(override=True)
NYT_API_KEY = os.getenv('NYT_API_KEY')
NEWS_DB_PATH = os.getenv('NEWS_DB_PATH')

# === Database Setup ===
def connect_db(db_path=NEWS_DB_PATH):
    """Connect to SQLite database."""
    conn = sqlite3.connect(db_path)
    return conn, conn.cursor(), conn

def create_table_nyt(cursor):
    """Create 'nyt' table if it doesn't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nyt (
            title TEXT,
            published_at TEXT,
            url TEXT UNIQUE,
            section_name TEXT
        )
    """)

def insert_article_nyt(cursor, title, published_at, url, section_name):
    """Insert a single NYT article into the nyt table."""
    cursor.execute("""
        INSERT OR IGNORE INTO nyt (title, published_at, url, section_name)
        VALUES (?, ?, ?, ?)
    """, (title, published_at, url, section_name))

# === API Fetching Functions ===

def fetch_nyt_month(conn, cursor, year, month, start_date, end_date):
    """Fetch all articles from the NYT archive for a specific month."""
    print(f"\n🔍 Fetching NYT archive for {year}-{month:02d}...")
    archive_url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={NYT_API_KEY}"
    response = requests.get(archive_url)

    if response.status_code != 200:
        print(f"Failed to fetch NYT data for {year}-{month:02d}. Status code: {response.status_code}")
        return

    data = response.json()
    docs = data.get('response', {}).get('docs', [])
    print(f"Found {len(docs)} articles in the archive for {year}-{month:02d}")

    for doc in docs:
        headline = doc.get('headline', {}).get('main')
        web_url = doc.get('web_url')
        raw_pub_date = doc.get('pub_date')
        section_name = doc.get('section_name')

        if not (headline and web_url and raw_pub_date):
            continue

        try:
            dt = datetime.fromisoformat(raw_pub_date.replace('+0000', '+00:00'))
            dt_naive = dt.replace(tzinfo=None)
        except Exception as e:
            print(f"Error converting date {raw_pub_date}: {e}")
            continue

        if dt_naive < start_date or dt_naive > end_date:
            continue

        pub_date_formatted = dt_naive.strftime("%Y-%m-%dT%H:%M:%SZ")

        insert_article_nyt(cursor, headline, pub_date_formatted, web_url, section_name)
        print(f"Inserted: {headline}")

    # Commit after processing one month
    conn.commit()

# === Master Function ===

def get_nyt_headlines(start_date_str, end_date_str, db_path=NEWS_DB_PATH):
    """Fetch NYT headlines between start_date and end_date."""
    conn, cursor, conn_obj = connect_db(db_path=db_path)
    create_table_nyt(cursor)

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d') + timedelta(hours=23, minutes=59, seconds=59)

    current = start_date.replace(day=1)

    while current <= end_date:
        year = current.year
        month = current.month

        fetch_nyt_month(conn_obj, cursor, year, month, start_date, end_date)

        print("Sleeping for 12 seconds to respect NYT API rate limits...")
        time.sleep(12)

        # Move to the next month
        if month == 12:
            current = current.replace(year=year + 1, month=1)
        else:
            current = current.replace(month=month + 1)

    conn_obj.commit()
    conn_obj.close()
    print("\n✅ All NYT headlines have been saved to database:", db_path)

# === Entrypoint ===
if __name__ == "__main__":
    get_nyt_headlines('2018-12-25', '2018-12-31', db_path=NEWS_DB_PATH)
# Completed: 2018-12-25 to 2025-04-14