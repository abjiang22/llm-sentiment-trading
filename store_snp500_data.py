import sqlite3
import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv(override=True)
NYT_API_KEY = os.getenv('NYT_API_KEY')
SNP_DB_PATH = os.getenv('SNP_DB_PATH')

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

def fetch_and_insert_snp500_data(db_path=SNP_DB_PATH):
    # Load environment and connect to database
    conn, cursor = connect_db(db_path)

    # Setup table
    create_snp500_table(cursor)
    conn.commit()

    # Fetch and insert data
    symbol = '^GSPC'
    data = fetch_snp500_data(symbol, start_date="2018-12-25", end_date="2025-04-14")
    insert_bulk_data(cursor, data)
    conn.commit()

    print("\u2705 S&P 500 data successfully inserted into the database.")

    # Display sample data
    display_sample_data(conn)

    # Close connection
    conn.close()

def reorder_snp500(db_path=SNP_DB_PATH):
    conn, cursor = connect_db(db_path)

    df = pd.read_sql_query("SELECT * FROM snp500 ORDER BY trade_date ASC", conn)

    cursor.execute("BEGIN TRANSACTION;")
    cursor.execute("DROP TABLE IF EXISTS snp500")
    create_snp500_table(cursor)

    records = [
        (
            row['trade_date'],
            row['open_price'],
            row['close_price'],
            row['high_price'],
            row['low_price'],
            row['volume']
        )
        for _, row in df.iterrows()
    ]

    cursor.executemany('''
        INSERT OR IGNORE INTO snp500 (trade_date, open_price, close_price, high_price, low_price, volume)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', records)
    conn.commit()

    print("\u2705 S&P 500 table reordered by ascending trade_date.")
    conn.close()

reorder_snp500()

# Note: Data stored from 12/25/2018 to 4/14/2025