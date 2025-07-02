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

def add_increase_and_daily_return_columns(db_path="data/snp500.db", table_name="snp500"):
    # Connect to the database
    conn = sqlite3.connect(db_path)

    # Load the table into a DataFrame
    df = pd.read_sql_query(
        f"SELECT trade_id, trade_date, open_price, close_price FROM {table_name} ORDER BY trade_date", conn
    )

    df["open_price"] = pd.to_numeric(df["open_price"], errors="coerce")
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")

    # Calculate 'increase' (1 if close > open, else 0)
    df["increase"] = (df["close_price"] > df["open_price"]).astype(int)

    # Calculate 'return' = (close - open) / open
    df["daily_return"] = (df["close_price"] - df["open_price"]) / df["open_price"]

    # Check existing columns
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]

    # Add columns if they don't exist
    if "increase" not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN increase INTEGER")
    if "daily_return" not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN daily_return REAL")

    # Batch update values
    updates = [(row.increase, row.daily_return, row.trade_id) for row in df.itertuples() if pd.notnull(row.daily_return)]

    cursor.executemany(f"""
        UPDATE {table_name}
        SET increase = ?, daily_return = ?
        WHERE trade_id = ?
    """, updates)

    conn.commit()
    conn.close()
    print("✅ 'increase' and 'daily_return' columns updated successfully.")


#fetch_and_insert_snp500_data(start_date="2017-12-25", end_date="2023-12-31", db_path="data/snp500.db")
add_increase_and_daily_return_columns()
#add_percent_change_column("data/snp500.db", "snp500")
# Note: Data stored from 12/25/2018 to 4/14/2025