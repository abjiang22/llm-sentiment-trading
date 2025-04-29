import sqlite3
import pandas as pd
from datetime import time as dtime

def exclude_market_hours(news_db_path, snp500_db_path):
    # Connect to databases
    with sqlite3.connect(news_db_path) as news_conn, sqlite3.connect(snp500_db_path) as snp_conn:
        # Load trading dates
        trading_days = pd.read_sql("SELECT trade_date FROM snp500", snp_conn)
        trading_days['trade_date'] = pd.to_datetime(trading_days['trade_date']).dt.date
        trading_days_set = set(trading_days['trade_date'])

        # Load unfiltered news
        df = pd.read_sql("SELECT * FROM master0_unfiltered_4_29", news_conn)
        df['published_at'] = pd.to_datetime(df['published_at'])
        df['pub_date'] = df['published_at'].dt.date
        df['pub_time'] = df['published_at'].dt.time

        market_open = dtime(9, 30)
        market_close = dtime(16, 0)

        # Exclude rows
        filtered_df = df[~(
            (df['pub_date'].isin(trading_days_set)) &
            (df['pub_time'] >= market_open) & (df['pub_time'] <= market_close)
        )]

        # Save filtered table
        filtered_df.drop(columns=['pub_date', 'pub_time']).to_sql("master0", news_conn, if_exists="replace", index=False)

    print(f"Filtered dataset saved to 'master0'. Rows kept: {len(filtered_df)}")

# Call the function
exclude_market_hours("data/news.db", "data/snp500.db")