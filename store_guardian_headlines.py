import requests
import sqlite3
import time
import calendar
from datetime import datetime
import os
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv(override=True)
GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')
NEWS_DB_PATH = os.getenv('NEWS_DB_PATH', 'News.db')

# === Database Setup ===

def connect_db(db_path=NEWS_DB_PATH):
    """Connect to SQLite database."""
    conn = sqlite3.connect(db_path)
    return conn, conn.cursor(), conn

def create_table_guardian(cursor):
    """Create 'guardian' table if it doesn't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS guardian (
            title TEXT,
            published_at TEXT,
            url TEXT UNIQUE,
            section TEXT
        )
    """)

def insert_article_guardian(cursor, title, published_at, url, section):
    """Insert a single article into the guardian table."""
    cursor.execute("""
        INSERT OR IGNORE INTO guardian (title, published_at, url, section)
        VALUES (?, ?, ?, ?)
    """, (title, published_at, url, section))

# === API Fetching Functions ===

def fetch_guardian_articles(conn, cursor, month_start, month_end, year, month, start_date, end_date):
    """Fetch Guardian articles for a given month and insert only within desired date range."""
    page = 1
    while True:
        params = {
            'api-key': GUARDIAN_API_KEY,
            'from-date': month_start,
            'to-date': month_end,
            'page-size': 200,
            'page': page,
            'show-fields': 'headline'
        }
        url = "https://content.guardianapis.com/search"
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json().get("response", {})
            results = data.get("results", [])
            print(f"Found {len(results)} articles on page {page}.")

            for item in results:
                title = item.get('webTitle')
                url_ = item.get('webUrl')
                pub_date = item.get('webPublicationDate')
                section = item.get('sectionName', '')

                if pub_date:
                    try:
                        dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        pub_date_formatted = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    except:
                        pub_date_formatted = ''
                else:
                    pub_date_formatted = ''

                if title and pub_date_formatted and url_:
                    try:
                        dt_article = datetime.strptime(pub_date_formatted, "%Y-%m-%dT%H:%M:%SZ")
                        if start_date <= dt_article <= end_date:
                            insert_article_guardian(cursor, title, pub_date_formatted, url_, section)
                            print(f"Inserted: {title}")
                        else:
                            print(f"Skipping article from {dt_article.date()}: Outside requested date range")
                    except Exception as e:
                        print(f"Error parsing article date: {e}")

            conn.commit()

            current_page = data.get("currentPage", 1)
            total_pages = data.get("pages", 1)
            if current_page < total_pages:
                page += 1
                time.sleep(1)  # Delay to respect API limits
            else:
                break
        else:
            print(f"Failed to fetch Guardian data for {year}-{month:02d} page {page}. Status code: {response.status_code}")
            break

def get_guardian_headlines(start_date_str, end_date_str, db_path=NEWS_DB_PATH):
    """Fetch Guardian headlines between start_date and end_date."""
    conn, cursor, conn_obj = connect_db(db_path=db_path)
    create_table_guardian(cursor)

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    current = start_date.replace(day=1)

    while current <= end_date:
        year = current.year
        month = current.month

        month_start = current.strftime('%Y-%m-%d')
        last_day = calendar.monthrange(year, month)[1]
        month_end_dt = datetime(year, month, last_day)
        if month_end_dt > end_date:
            month_end_dt = end_date
        month_end = month_end_dt.strftime('%Y-%m-%d')

        print(f"\n🔍 Fetching Guardian headlines for {year}-{month:02d} from {month_start} to {month_end}...")
        fetch_guardian_articles(conn_obj, cursor, month_start, month_end, year, month, start_date, end_date)

        print("Sleeping for 5 seconds before moving to the next month...")
        time.sleep(5)

        # Move to next month
        if month == 12:
            current = current.replace(year=year + 1, month=1)
        else:
            current = current.replace(month=month + 1)

    conn_obj.commit()
    conn_obj.close()
    print("\n✅ All Guardian headlines have been saved to database:", db_path)

# === Entrypoint ===
if __name__ == "__main__":
    get_guardian_headlines("2018-12-25", "2018-12-31", db_path=NEWS_DB_PATH)
# Completed: 2018-12-25 to 2025-04-14