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
    train_start: str = '2019-01-02',
    train_end: str = '2024-06-30',
    test_start: str = '2024-07-01',
    test_end: str = '2025-04-14',
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

    save_path_with_sentiment = f'{model_name}_simple_with_sentiment.csv',
    save_path_without_sentiment = f'{model_name}_simple_without_sentiment.csv'

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
    news_db_path: str,
    news_table: str,
    sentiment_column: str,
    model_name: str,
    source_column: str = 'source',
    train_start: str = '2019-01-02',
    train_end: str = '2024-06-30',
    test_start: str = '2024-07-01',
    test_end: str = '2025-04-14',
   
):
    # === 1. Load S&P data ===
    conn_snp = sqlite3.connect(snp_db_path)
    snp_df = pd.read_sql_query(
        f"SELECT trade_date, open_price AS open, close_price AS close FROM {snp_table}",
        conn_snp,
        parse_dates=['trade_date']
    )
    conn_snp.close()

    # === 2. Load & aggregate news sentiment ===
    conn_news = sqlite3.connect(news_db_path)
    news_df = pd.read_sql_query(
        f"""
        SELECT 
          DATE(published_at) AS trade_date,
          {source_column},
          {sentiment_column}
        FROM {news_table}
        WHERE {sentiment_column} IS NOT NULL
        """,
        conn_news,
        parse_dates=['trade_date']
    )
    conn_news.close()

    news_df[sentiment_column] = pd.to_numeric(news_df[sentiment_column], errors='coerce')
    daily_src = (
        news_df
        .groupby(['trade_date', source_column])[sentiment_column]
        .mean()
        .reset_index()
    )

    # Pivot raw averages
    raw_pivot = (
        daily_src
        .pivot(index='trade_date', columns=source_column, values=sentiment_column)
        .add_prefix('avg_sentiment_')
    ).fillna(0)

    # === 3. Merge raw averages with S&P data ===
    df = (
        snp_df
        .set_index('trade_date')
        .join(raw_pivot, how='left')
        .fillna(0)
        .reset_index()
        .sort_values('trade_date')
    )

    # Identify raw average columns
    avg_cols = [c for c in df.columns if c.startswith('avg_sentiment_')]

    # Copy raw into standardized columns
    df_std = df.copy()
    std_cols = [c.replace('avg_sentiment_', 'sentiment_') for c in avg_cols]

    # === 4. Standardize the sentiment columns ===
    scaler = StandardScaler()
    train_mask = (df_std.trade_date >= train_start) & (df_std.trade_date <= train_end)
    scaler.fit(df_std.loc[train_mask, avg_cols])
    df_std[std_cols] = scaler.transform(df_std[avg_cols])

    # === 5. Train/test split ===
    train = df_std[(df_std.trade_date >= train_start) & (df_std.trade_date <= train_end)]
    test  = df_std[(df_std.trade_date >= test_start)  & (df_std.trade_date <= test_end)]

    y_train = train['close']
    y_test  = test['close']

    X_train_open = train[['open']]
    X_test_open  = test[['open']]

    X_train_src = train[['open'] + std_cols]
    X_test_src  = test[['open'] + std_cols]

    # === 6. Fit & evaluate ===
    m0   = LinearRegression().fit(X_train_open, y_train)
    pred0= m0.predict(X_test_open)
    mse0 = mean_squared_error(y_test, pred0)
    print(f"MSE without sentiment: {mse0:.4f}")

    m1   = LinearRegression().fit(X_train_src, y_train)
    pred1= m1.predict(X_test_src)
    mse1 = mean_squared_error(y_test, pred1)
    print(f"MSE with standardized per‑source sentiment: {mse1:.4f}")

    # Print coefficients
    features = ['open'] + std_cols
    coefs = pd.Series([m1.intercept_] + m1.coef_.tolist(),
                      index=['intercept'] + features)
    print("\nLearned coefficients:")
    print(coefs)

    imp = (mse0 - mse1) / mse0 * 100
    print(f"\nMSE improvement: {imp:.2f}%")

    # === 7. Save results ===
    save_with             = f'predictions/{model_name}_simple_with_source_sentiment.csv',
    save_without          = f'predictions/{model_name}_simple_without_source_sentiment.csv'
    if save_without:
        out0 = test[['trade_date', 'open']].copy()
        out0['actual_close']    = y_test.values
        out0['predicted_close'] = pred0
        out0.to_csv(save_without, index=False)
        print(f"Saved no‑sentiment predictions to {save_without}")

    if save_with:
        # include open, raw avg sentiment, standardized sentiment
        cols = ['trade_date', 'open'] + avg_cols + std_cols
        out1 = test[cols].copy()
        out1['actual_close']    = y_test.values
        out1['predicted_close'] = pred1
        out1.to_csv(save_with, index=False)
        print(f"Saved with‑sentiment predictions to {save_with}")

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
    'lm'
)

run_regression_by_source(
    snp_db_path           = 'data/snp500.db',
    snp_table             = 'snp500',
    news_db_path          = 'data/news.db',
    news_table            = 'master0',
    sentiment_column      = 'lm_sentiment_score',
    source_column         = 'source',
    model_name            = 'lm',
)


"""
run_regression_by_source(
    snp_db_path           = 'data/snp500.db',
    snp_table             = 'snp500',
    news_db_path          = 'data/news.db',
    news_table            = 'master0',
    sentiment_column      = 'llm_sentiment_score_gpt4o_zero_shot',
    source_column         = 'source',
    model_name            = 'gpt4o_zero_shot',
)
"""


"""
run_simple_regression(
    'data/snp500.db',
    'snp500',
    'lm_sentiment_score',
    save_path_with_sentiment='predictions/lm_simple_sentiment.csv',
    save_path_without_sentiment='predictions/lm_simple_no_sentiment.csv'
)
"""