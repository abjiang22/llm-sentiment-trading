import requests
import sqlite3
from datetime import datetime, timedelta
import os
import time
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv(override=True)
API_KEY = os.getenv('NEWSAPI_KEY')
NEWS_DB_PATH = os.getenv('NEWS_DB_PATH')

# === Configuration ===
PAGE_SIZE = 100
REQUEST_DELAY = 0.1

DOMAINS = [
    "bloomberg.com", "businessinsider.com", "financialpost.com", "fortune.com",
    "apnews.com", "washingtonpost.com", "time.com", "wsj.com"
]

# === Database Setup ===

def connect_db(db_path=NEWS_DB_PATH):
    """Connect to SQLite database."""
    conn = sqlite3.connect(db_path)
    return conn, conn.cursor()

def create_table(cursor, source_name):
    """Create table for a specific domain if it does not exist."""
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {source_name} (
            title TEXT,
            published_at TEXT,
            url TEXT UNIQUE
        )
    """)

def insert_article(cursor, source_name, title, published_at, url):
    """Insert a single article into the domain's table."""
    cursor.execute(f"""
        INSERT OR IGNORE INTO {source_name} (title, published_at, url)
        VALUES (?, ?, ?)
    """, (title, published_at, url))

# === API Request Functions ===

def fetch_total_results(domain, from_date, to_date):
    """Fetch total number of articles available for a domain in a date range."""
    params = {
        'apiKey': API_KEY,
        'from': from_date,
        'to': to_date,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': PAGE_SIZE,
        'domains': domain,
        'page': 1
    }
    response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")

    data = response.json()
    if data.get('status') != 'ok':
        raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
    
    return data.get('totalResults', 0)

def fetch_and_store_articles(cursor, domain, from_date, to_date):
    """Fetch articles from a domain between dates and store them in the database."""
    print(f"\n🔍 Fetching articles from {domain} ({from_date} to {to_date})...")
    source_name = domain.split('.')[0].lower()
    create_table(cursor, source_name)

    try:
        total_results = fetch_total_results(domain, from_date, to_date)
        if total_results == 0:
            print("No articles found for this period.")
            return

        total_pages = (total_results + PAGE_SIZE - 1) // PAGE_SIZE
        print(f"Total results: {total_results}, Total pages: {total_pages}")

    except Exception as e:
        print(f"❌ Request failed during initial call: {e}")
        return

    for page in range(1, total_pages + 1):
        params = {
            'apiKey': API_KEY,
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': PAGE_SIZE,
            'domains': domain,
            'page': page
        }
        try:
            response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
            if response.status_code != 200:
                print(f"❌ Error {response.status_code} on page {page}: {response.text}")
                break

            data = response.json()
            if data.get('status') != 'ok':
                print(f"❌ API Error on page {page}: {data.get('message', 'Unknown error')}")
                break

            articles = data.get('articles', [])
            if not articles:
                print("No more articles found on this page.")
                break

            print(f"✅ Retrieved {len(articles)} articles from page {page}")

            for article in articles:
                raw_title = article.get('title')
                if raw_title is None:
                    continue

                title = raw_title.strip()
                published_at = article.get('publishedAt') or ''
                url = article.get('url') or ''

                if title and published_at and url:
                    insert_article(cursor, source_name, title, published_at, url)

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"❌ Request failed on page {page}: {e}")
            break

def date_to_iso(dt):
    """Convert datetime object to ISO 8601 string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# === Main Store Logic ===

def store_newsapi_data(start_date_str, end_date_str, db_path=NEWS_DB_PATH):
    """Fetch and store NewsAPI data across all configured domains for a given date range."""
    conn, cursor = connect_db(db_path=db_path)
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

    current_from = start_dt
    while current_from < end_dt:
        current_to = current_from + timedelta(weeks=1)
        if current_to > end_dt:
            current_to = end_dt

        from_iso = date_to_iso(current_from)
        to_iso = date_to_iso(current_to)

        for domain in DOMAINS:
            fetch_and_store_articles(cursor, domain, from_iso, to_iso)

        current_from = current_to

    conn.commit()
    conn.close()
    print("\n✅ All articles saved to database:", db_path)

# === Entrypoint ===

if __name__ == "__main__":
    store_newsapi_data("2018-12-25", "2018-12-31", db_path=NEWS_DB_PATH)
# Completed: 2018-12-25 to 2025-04-14