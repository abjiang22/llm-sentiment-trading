import sqlite3
import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv(override=True)
NYT_API_KEY = os.getenv('NYT_API_KEY')

def connect_db(db_path):
    conn = sqlite3.connect(db_path)
    return conn, conn.cursor()

def create_snp500_table(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS snp500 (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date DATE UNIQUE,
            open_price REAL,
            close_price REAL,
            high_price REAL,
            low_price REAL,
            volume INTEGER
        );
    ''')

def insert_snp500_data(cursor, trade_date, open_price, close_price, high_price, low_price, volume):
    cursor.execute('''
        INSERT OR IGNORE INTO snp500 (trade_date, open_price, close_price, high_price, low_price, volume)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (trade_date, open_price, close_price, high_price, low_price, volume))

def fetch_snp500_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)

def insert_bulk_data(cursor, data):
    for index, row in data.iterrows():
        trade_date_str = index.strftime('%Y-%m-%d')
        insert_snp500_data(
            cursor,
            trade_date_str,
            float(row['Open']),
            float(row['Close']),
            float(row['High']),
            float(row['Low']),
            int(row['Volume'])
        )

def display_sample_data(conn):
    df = pd.read_sql_query("SELECT * FROM snp500 ORDER BY trade_date LIMIT 10", conn)
    print(df)

def fetch_and_insert_snp500_data(start_date="2017-12-25", end_date="2025-04-14", db_path="data/snp500.db"):
    conn, cursor = connect_db(db_path)
    # Setup table
    create_snp500_table(cursor)
    conn.commit()

    # Fetch and insert data
    symbol = '^GSPC'
    data = fetch_snp500_data(symbol, start_date=start_date, end_date=end_date)
    insert_bulk_data(cursor, data)
    conn.commit()

    print("\u2705 S&P 500 data successfully inserted into the database.")

    # Display sample data
    display_sample_data(conn)

    # Close connection
    conn.close()

def reorder_snp500(db_path="data/snp500.db", table_name="snp500"):
    conn = sqlite3.connect(db_path)
    
    # Read entire table including all columns
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    # Sort by trade_date
    df_sorted = df.sort_values(by="trade_date").reset_index(drop=True)

    # Optional: Overwrite the table in-place (non-destructive to columns)
    df_sorted.to_sql(table_name, conn, if_exists="replace", index=False)

    conn.close()
    print("✅ S&P 500 table reordered by ascending trade_date")


def add_percent_change_column(db_path="data/snp500.db", table_name="snp500"):
    # Connect to the database
    conn = sqlite3.connect(db_path)

    # Load the table into a DataFrame, sorted by trade_date
    df = pd.read_sql_query(f"SELECT trade_id, trade_date, close_price FROM {table_name} ORDER BY trade_date", conn)

    # Calculate percent change between current and previous close_price
    df["percent_change_close"] = df["close_price"].pct_change() * 100
    df["percent_change_close"] = df["percent_change_close"].round(4)

    # Check if column exists, add it if not
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    if "percent_change_close" not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN percent_change_close REAL")

    # Write the percent_change values back into the database using trade_id
    for row in df.itertuples():
        if pd.notnull(row.percent_change_close):
            cursor.execute(f"""
                UPDATE {table_name}
                SET percent_change_close = ?
                WHERE trade_id = ?
            """, (row.percent_change_close, row.trade_id))

    conn.commit()
    conn.close()
    print("✅ percent_change_close column updated efficiently using pandas.")

#fetch_and_insert_snp500_data(start_date="2017-12-25", end_date="2025-04-14", db_path="data/snp500.db")
#add_percent_change_column("data/snp500.db", "snp500")

reorder_snp500()
# Note: Data stored from 12/25/2018 to 4/14/2025