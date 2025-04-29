import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def run_simple_regression(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    train_start: str = '2019-01-02',
    train_end: str = '2024-06-30',
    test_start: str = '2024-07-01',
    test_end: str = '2025-04-14',
    save_path_with_sentiment: str = 'simple_predictions_with_sentiment.csv',
    save_path_without_sentiment: str = 'simple_predictions_without_sentiment.csv'
):
    # === STEP 1: Load dataset ===
    conn = sqlite3.connect(db_path)
    query = f"SELECT trade_date, open_price, close_price, {sentiment_column} FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['date'] = pd.to_datetime(df['trade_date'])
    df = df.rename(columns={
        'open_price': 'open',
        'close_price': 'close',
        sentiment_column: 'sentiment'
    })
    df = df.sort_values('date').reset_index(drop=True)

    # === STEP 2: Train-test split ===
    train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
    test_df  = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

    y_train = train_df['close']
    y_test = test_df['close']

    # === WITH sentiment ===
    X_train_with = train_df[['open', 'sentiment']]
    X_test_with = test_df[['open', 'sentiment']]

    model_with = LinearRegression()
    model_with.fit(X_train_with, y_train)
    y_pred_with = model_with.predict(X_test_with)
    mse_with = mean_squared_error(y_test, y_pred_with)
    print(f"Test MSE with sentiment: {mse_with:.4f}")

    preview_with = test_df[['date']].copy()
    preview_with['actual_close'] = y_test.values
    preview_with['predicted_close'] = y_pred_with
    preview_with.to_csv(save_path_with_sentiment, index=False)
    print(f"Predictions with sentiment saved to {save_path_with_sentiment}")

    # === WITHOUT sentiment ===
    X_train_without = train_df[['open']]
    X_test_without = test_df[['open']]

    model_without = LinearRegression()
    model_without.fit(X_train_without, y_train)
    y_pred_without = model_without.predict(X_test_without)
    mse_without = mean_squared_error(y_test, y_pred_without)
    print(f"Test MSE without sentiment: {mse_without:.4f}")

    preview_without = test_df[['date']].copy()
    preview_without['actual_close'] = y_test.values
    preview_without['predicted_close'] = y_pred_without
    preview_without.to_csv(save_path_without_sentiment, index=False)
    print(f"Predictions without sentiment saved to {save_path_without_sentiment}")

    # === Comparison ===
    improvement = (mse_without - mse_with) / mse_without * 100
    print(f"MSE improvement using sentiment: {improvement:.2f}%")

run_simple_regression(
    'data/snp500.db',
    'snp500',
    'avg_sentiment_lm',
    save_path_with_sentiment='predictions/lm_simple_sentiment.csv',
    save_path_without_sentiment='predictions/lm_simple_no_sentiment.csv'
)
