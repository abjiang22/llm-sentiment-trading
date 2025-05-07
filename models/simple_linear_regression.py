import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import List, Optional
from sklearn.preprocessing import StandardScaler

def run_simple_regression(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    model_name: str,
    train_start: str = '2018-01-02',
    train_end: str = '2023-06-30',
    test_start: str = '2023-07-01',
    test_end: str = '2025-03-01'
):
    # === STEP 1: Load dataset ===
    conn = sqlite3.connect(db_path)
    query = f"SELECT trade_date, open_price, close_price, {sentiment_column} FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()

    save_path_with_sentiment = f'predictions/{model_name}_simple_with_sentiment.csv'
    save_path_without_sentiment = f'predictions/{model_name}_simple_without_sentiment.csv'


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

    
    # --- PRINT coefficients for "with sentiment" ---
    print("Coefficients (with sentiment):")
    print(f"  Intercept: {model_with.intercept_:.4f}")
    print(f"  open coef: {model_with.coef_[0]:.4f}")
    print(f"  sentiment coef: {model_with.coef_[1]:.4f}")

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

    # --- PRINT coefficients for "without sentiment" ---
    print("Coefficients (without sentiment):")
    print(f"  Intercept: {model_without.intercept_:.4f}")
    print(f"  open coef: {model_without.coef_[0]:.4f}")

    preview_without = test_df[['date']].copy()
    preview_without['actual_close'] = y_test.values
    preview_without['predicted_close'] = y_pred_without
    preview_without.to_csv(save_path_without_sentiment, index=False)
    print(f"Predictions without sentiment saved to {save_path_without_sentiment}")

    # === Comparison ===
    improvement = (mse_without - mse_with) / mse_without * 100
    print(f"MSE improvement using sentiment: {improvement:.2f}%")

def run_regression_by_source(
    snp_db_path: str,
    snp_table: str,
    sentiment_column: str,
    model_name: str,
    train_start: str = '2018-01-02',
    train_end: str = '2023-06-30',
    test_start: str = '2023-07-01',
    test_end: str = '2025-03-01'
):
    import sqlite3
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    # === 1. Load S&P data with sentiment columns ===
    conn = sqlite3.connect(snp_db_path)
    df = pd.read_sql_query(
        f"SELECT * FROM {snp_table}",
        conn,
        parse_dates=['trade_date']
    )
    conn.close()

    # === 2. Identify sentiment-by-source columns ===
    sentiment_cols = [col for col in df.columns if col.startswith(f"{sentiment_column}_")]
    if not sentiment_cols:
        raise ValueError(f"No columns found matching prefix '{sentiment_column}_'")

    std_cols = [col.replace(f"{sentiment_column}_", "sentiment_") for col in sentiment_cols]

    # === 3. Standardize per-source sentiment columns ===
    df_std = df.copy()
    scaler = StandardScaler()
    train_mask = (df_std.trade_date >= train_start) & (df_std.trade_date <= train_end)
    scaler.fit(df_std.loc[train_mask, sentiment_cols])
    df_std[std_cols] = scaler.transform(df_std[sentiment_cols])
    df_std[std_cols] = df_std[std_cols].fillna(0)

    # === 4. Train/test split ===
    train = df_std[(df_std.trade_date >= train_start) & (df_std.trade_date <= train_end)]
    test  = df_std[(df_std.trade_date >= test_start)  & (df_std.trade_date <= test_end)]

    y_train = train['close_price']
    y_test  = test['close_price']

    X_train_open = train[['open_price']]
    X_test_open  = test[['open_price']]

    X_train_sent = train[['open_price'] + std_cols]
    X_test_sent  = test[['open_price'] + std_cols]

    # === 5. Fit models ===
    m0 = LinearRegression().fit(X_train_open, y_train)
    pred0 = m0.predict(X_test_open)
    mse0 = mean_squared_error(y_test, pred0)
    print(f"MSE without sentiment: {mse0:.4f}")

    m1 = LinearRegression().fit(X_train_sent, y_train)
    pred1 = m1.predict(X_test_sent)
    mse1 = mean_squared_error(y_test, pred1)
    print(f"MSE with {sentiment_column} by source: {mse1:.4f}")

    # === 6. Output coefficients ===
    features = ['open_price'] + std_cols
    coefs = pd.Series([m1.intercept_] + m1.coef_.tolist(),
                      index=['intercept'] + features)
    print("\nLearned coefficients:")
    print(coefs)

    imp = (mse0 - mse1) / mse0 * 100
    print(f"\nMSE improvement: {imp:.2f}%")

    # === 7. Save results ===
    base = f"predictions/{model_name}"
    
    out0 = test[['trade_date', 'open_price']].copy()
    out0['actual_close'] = y_test.values
    out0['predicted_close'] = pred0
    out0.to_csv(f"{base}_by_source_no_sentiment.csv", index=False)

    out1 = test[['trade_date', 'open_price', 'close_price'] + sentiment_cols + std_cols].copy()
    out1['predicted_close'] = pred1
    out1.to_csv(f"{base}_by_source_with_sentiment.csv", index=False)

    print(f"Saved predictions to {base}_*.csv")

    return {
        'mse_without': mse0,
        'mse_with': mse1,
        'improvement_pct': imp,
        'coefficients': coefs
    }

run_simple_regression(
    'data/snp500.db',
    'snp500',
    'avg_sentiment_lm',
    'finbert'
)

run_regression_by_source(
    snp_db_path           = 'data/snp500.db',
    snp_table             = 'snp500',
    sentiment_column      = 'avg_sentiment_lm',
    model_name            = 'finbert',
)


"""
run_simple_regression(
    'data/snp500.db',
    'snp500',
    'avg_sentiment_gpt4o_zero_shot',
    'llm'
)

run_regression_by_source(
    snp_db_path           = 'data/snp500.db',
    snp_table             = 'snp500',
    sentiment_column      = 'avg_sentiment_gpt4o_zero_shot',
    model_name            = 'llm',
)

"""
