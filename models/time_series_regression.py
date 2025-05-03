import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def run_lagged_price_model(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    x_days: int = 5,
    train_start: str = '2019-01-02',
    train_end: str = '2024-06-30',
    test_start: str = '2024-07-01',
    test_end: str = '2025-04-14',
    save_path_with_sentiment: str = 'predictions_with_sentiment.csv',
    save_path_without_sentiment: str = 'predictions_without_sentiment.csv'
):
    # === STEP 1: Load full dataset ===
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

    # === STEP 2: Create lagged features ===
    for lag in range(1, x_days + 1):
        df[f'open_lag{lag}'] = df['open'].shift(lag)
        df[f'close_lag{lag}'] = df['close'].shift(lag)
        df[f'sentiment_lag{lag}'] = df['sentiment'].shift(lag)

    df = df.dropna().reset_index(drop=True)

    # === STEP 3: Train-test split ===
    train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
    test_df  = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

    base_features = ['open'] + [col for col in df.columns if 'lag' in col]
    features_with_sentiment = base_features
    features_without_sentiment = [col for col in base_features if 'sentiment_lag' not in col]

    # === STEP 4A: Train model WITH sentiment ===
    X_train_with = train_df[features_with_sentiment]
    y_train = train_df['close']
    X_test_with = test_df[features_with_sentiment]
    y_test = test_df['close']

    model_with = LinearRegression()
    model_with.fit(X_train_with, y_train)
    y_pred_with = model_with.predict(X_test_with)
    mse_with = mean_squared_error(y_test, y_pred_with)
    print(f"Test MSE with sentiment: {mse_with:.4f}")

    # Save predictions with sentiment
    preview_with = test_df[['date']].copy()
    preview_with['actual_close'] = y_test.values
    preview_with['predicted_close'] = y_pred_with
    preview_with.to_csv(save_path_with_sentiment, index=False)
    print(f"Predictions with sentiment saved to {save_path_with_sentiment}")

    # === STEP 4B: Train model WITHOUT sentiment ===
    X_train_without = train_df[features_without_sentiment]
    X_test_without = test_df[features_without_sentiment]

    model_without = LinearRegression()
    model_without.fit(X_train_without, y_train)
    y_pred_without = model_without.predict(X_test_without)
    mse_without = mean_squared_error(y_test, y_pred_without)
    print(f"Test MSE without sentiment: {mse_without:.4f}")

    # Save predictions without sentiment
    preview_without = test_df[['date']].copy()
    preview_without['actual_close'] = y_test.values
    preview_without['predicted_close'] = y_pred_without
    preview_without.to_csv(save_path_without_sentiment, index=False)
    print(f"Predictions without sentiment saved to {save_path_without_sentiment}")

    # === Comparison ===
    improvement = (mse_without - mse_with) / mse_without * 100
    print(f"MSE improvement using sentiment: {improvement:.2f}%")

# Example usage:
run_lagged_price_model(
    'data/snp500.db',
    'snp500',
    'avg_sentiment_finbert',
    save_path_with_sentiment='finbert_ts_sentiment.csv',
    save_path_without_sentiment='finbert_ts_no_sentiment.csv'
)