import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

def predict_increase_by_source_upgraded_with_price_history(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    model_name: str,
    sources: list = ["financialpost", "wsj", "bloomberg", "businessinsider"], 
    train_start: str = '2018-01-02',
    train_end: str = '2023-12-31',
    test_start: str = '2024-01-01',
    test_end: str = '2024-12-31',
    z_clip_threshold: float = 3.0,
    n_days_lag: int = 3
):
    # === Load data ===
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"""
        SELECT * FROM {table_name}
        WHERE increase IS NOT NULL
        """, conn, parse_dates=["trade_date"]
    )
    conn.close()

    # === Remove duplicate columns if present ===
    df = df.loc[:, ~df.columns.duplicated()]

    # === Verify required price columns exist ===
    if 'open_price' not in df.columns or 'close_price' not in df.columns:
        raise ValueError("Missing required columns: 'open_price' and/or 'close_price'.")

    # === Sentiment columns by source ===
    sentiment_cols = [f"{sentiment_column}_{src}" for src in sources if f"{sentiment_column}_{src}" in df.columns]
    if not sentiment_cols:
        raise ValueError(f"No matching sentiment columns found for selected sources: {sources}")

    # === Sort chronologically and engineer lag features ===
    df = df.sort_values('trade_date').reset_index(drop=True)

    for lag in range(1, n_days_lag + 1):
        df[f'prev_return_t-{lag}'] = (
            df['close_price'].shift(lag) - df['open_price'].shift(lag)
        ) / df['open_price'].shift(lag)
        df[f'close_price_t-{lag}'] = df['close_price'].shift(lag)

    lag_features = [col for col in df.columns if col.startswith('prev_return_t-') or col.startswith('close_price_t-')]

    # === Drop rows with any missing feature ===
    feature_cols = sentiment_cols + lag_features
    df = df.dropna(subset=feature_cols)

    # === Standardize ===
    train_mask = (df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feature_cols])

    df_std = df.copy()
    df_std[feature_cols] = pd.DataFrame(
        scaler.transform(df[feature_cols]),
        columns=feature_cols,
        index=df.index
    )

    df_std[sentiment_cols] = df_std[sentiment_cols].clip(-z_clip_threshold, z_clip_threshold)

    # === Train-test split ===
    train = df_std[train_mask]
    test  = df_std[(df_std['trade_date'] >= test_start) & (df_std['trade_date'] <= test_end)]

    X_train = train[feature_cols]
    y_train = train['increase']
    X_test  = test[feature_cols]
    y_test  = test['increase']

    # === Train logistic regression ===
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        solver='liblinear',
        max_iter=1000
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # === Evaluation ===
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"✅ Accuracy (with price history): {acc:.4f}")
    print(f"✅ AUC (with price history): {auc:.4f}")
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("=== Coefficients ===")
    print(pd.Series(model.coef_[0], index=feature_cols))
    print(f"Intercept: {model.intercept_[0]:.4f}")

    # === Save predictions ===
    output = test[['trade_date'] + feature_cols].copy()
    output['actual_increase'] = y_test.values
    output['predicted_proba'] = y_proba
    output['predicted_label'] = y_pred
    save_csv = f"predictions/{model_name}_with_price_history.csv"
    output.to_csv(save_csv, index=False)
    print(f"📁 Saved predictions to {save_csv}")

    # === Save model and scaler ===
    joblib.dump(model, f"models/{model_name}_logreg.pkl")
    joblib.dump(scaler, f"models/{model_name}_scaler.pkl")
    print(f"💾 Model and scaler saved to /models")


predict_increase_by_source_upgraded_with_price_history(
    db_path='data/snp500_5_6.db',
    table_name='snp500',
    sentiment_column='avg_sentiment_gpt4o_zero_shot',
    model_name='gpt4o_price_lag',
    n_days_lag=3
)