import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_sentiment_vs_price_change(db_path="data/snp500.db", table_name="snp500", sentiment_source="lm"):
    sentiment_column = f"avg_sentiment_{sentiment_source}"

    # Connect and load relevant data
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"""
        SELECT percent_change_close, {sentiment_column}
        FROM {table_name}
        WHERE percent_change_close IS NOT NULL AND {sentiment_column} IS NOT NULL
    """, conn)
    conn.close()

    df = df.dropna()

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(df[sentiment_column], df['percent_change_close'])

    # Plot scatter + regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(df[sentiment_column], df['percent_change_close'], alpha=0.6, label="Data points")
    x_vals = df[sentiment_column]
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='red', label=f"y = {slope:.4f}x + {intercept:.4f}\nR = {r_value:.4f}")
    plt.xlabel(sentiment_column)
    plt.ylabel("percent_change_close")
    plt.title("Price Change vs. Sentiment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"📈 Regression Results:")
    print(f"  Slope: {slope:.4f}")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  Pearson R: {r_value:.4f}")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.4e}")

# plot_sentiment_vs_price_change("data/snp500.db", "snp500", sentiment_source="lm")
# plot_sentiment_vs_price_change("data/snp500.db", "snp500", sentiment_source="finbert")