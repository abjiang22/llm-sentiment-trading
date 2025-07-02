from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
import sqlite3, pandas as pd, numpy as np, joblib, os

def xgb_baseline(
    db_path: str,
    table_name: str,
    model_name: str = "xgb_baseline_returns_only",
    train_start: str = '2018-01-01',
    train_end: str = '2022-12-31',
    test_start: str = '2023-01-01',
    test_end: str = '2023-12-31',
    n_days_lag: int = 3
):
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("model_weights", exist_ok=True)

    # === Load and clean ===
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"SELECT * FROM {table_name} WHERE increase IS NOT NULL AND daily_return IS NOT NULL",
        conn, parse_dates=["trade_date"]
    )
    conn.close()

    df["open_price"] = pd.to_numeric(df["open_price"], errors="coerce")
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")

    df = df.sort_values("trade_date").reset_index(drop=True)

    # === Generate lagged return features
    lag_features = []
    for lag in range(1, n_days_lag + 1):
        col = f"prev_return_t-{lag}"
        df[col] = (
            (df["close_price"].shift(lag) - df["open_price"].shift(lag)) / df["open_price"].shift(lag)
        )
        lag_features.append(col)

    # === Drop missing rows
    df = df.dropna(subset=lag_features)

    # === Split masks
    train_mask = (df["trade_date"] >= train_start) & (df["trade_date"] <= train_end)
    test_mask = (df["trade_date"] >= test_start) & (df["trade_date"] <= test_end)

    # === Impute (mostly redundant here but included for robustness)
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(df.loc[train_mask, lag_features])
    df[lag_features] = imputer.transform(df[lag_features])

    # === Prepare data
    X_train = df.loc[train_mask, lag_features]
    y_train = df.loc[train_mask, "increase"]
    X_test = df.loc[test_mask, lag_features]
    y_test = df.loc[test_mask, "increase"]

    # === Handle class imbalance
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    # === Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # === Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"✅ AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"✅ Log Loss: {log_loss(y_test, y_proba):.4f}")
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print("=== Feature Importances ===")
    print(pd.Series(model.feature_importances_, index=lag_features).sort_values(ascending=False))

    # === Save predictions
    output = df.loc[test_mask, ["trade_date"]].copy()
    output["actual_increase"] = y_test.values
    output["predicted_proba"] = y_proba
    output["predicted_label"] = y_pred
    save_path = f"predictions/{model_name}_xgb.csv"
    output.to_csv(save_path, index=False)
    print(f"📁 Saved predictions to {save_path}")

    # === Save model
    joblib.dump(model, f"model_weights/{model_name}_xgb.pkl")
    print(f"💾 Model artifacts saved to /model_weights")

def xgb_consolidated_sentiment(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    model_name: str,
    train_start: str = '2018-01-01',
    train_end: str = '2022-12-31',
    test_start: str = '2023-01-01',
    test_end: str = '2023-12-31',
    n_days_lag: int = 3
):
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("model_weights", exist_ok=True)

    # === Load and clean ===
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"SELECT * FROM {table_name} WHERE increase IS NOT NULL", conn, parse_dates=["trade_date"]
    )
    conn.close()

    df["open_price"] = pd.to_numeric(df["open_price"], errors="coerce")
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")

    if "open_price" not in df.columns or "close_price" not in df.columns:
        raise ValueError("Missing 'open_price' or 'close_price' columns.")
    if sentiment_column not in df.columns:
        raise ValueError(f"Sentiment column '{sentiment_column}' not found.")

    # === Generate lagged return features
    lag_features = []
    for lag in range(1, n_days_lag + 1):
        col = f"prev_return_t-{lag}"
        df[col] = (df["close_price"].shift(lag) - df["open_price"].shift(lag)) / df["open_price"].shift(lag)
        lag_features.append(col)

    feature_cols = lag_features + [sentiment_column]

    # === Split masks
    train_mask = (df["trade_date"] >= train_start) & (df["trade_date"] <= train_end)
    test_mask = (df["trade_date"] >= test_start) & (df["trade_date"] <= test_end)

    # === Impute missing values from training set only
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(df.loc[train_mask, feature_cols])
    df[feature_cols] = imputer.transform(df[feature_cols])

    # === Prepare data
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "increase"]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, "increase"]

    # === Class imbalance handling
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    # === Train XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # === Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"✅ AUC: {roc_auc_score(y_test, y_proba):.4f}")
    ll = log_loss(y_test, y_proba)
    print(f"✅ Log Loss: {ll:.4f}")
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print("=== Top Feature Importances ===")
    print(pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10))

    # === Save predictions
    output = df.loc[test_mask, ["trade_date"] + feature_cols].copy()
    output["actual_increase"] = y_test.values
    output["predicted_proba"] = y_proba
    output["predicted_label"] = y_pred
    save_path = f"predictions/{model_name}_xgb.csv"
    output.to_csv(save_path, index=False)
    print(f"📁 Saved predictions to {save_path}")

    # === Save model
    joblib.dump(model, f"model_weights/{model_name}_xgb.pkl")
    print(f"💾 Model artifacts saved to /model_weights")

def xgb_sentiment_by_source(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    model_name: str,
    sources: list = ["financialpost", "wsj", "bloomberg", "businessinsider"], 
    train_start: str = '2018-01-02',
    train_end: str = '2023-12-31',
    test_start: str = '2024-01-01',
    test_end: str = '2024-12-31',
    n_days_lag: int = 3
):
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("model_weights", exist_ok=True)

    # === Load and clean ===
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"SELECT * FROM {table_name} WHERE increase IS NOT NULL", conn, parse_dates=["trade_date"]
    )
    conn.close()

    df["open_price"] = pd.to_numeric(df["open_price"], errors="coerce")
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")

    if "open_price" not in df.columns or "close_price" not in df.columns:
        raise ValueError("Missing 'open_price' or 'close_price' columns.")

    # === Generate lagged return features
    lag_features = []
    for lag in range(1, n_days_lag + 1):
        col = f"prev_return_t-{lag}"
        df[col] = (df["close_price"].shift(lag) - df["open_price"].shift(lag)) / df["open_price"].shift(lag)
        lag_features.append(col)

    # === Select sentiment columns
    sentiment_cols = [f"{sentiment_column}_{s}" for s in sources if f"{sentiment_column}_{s}" in df.columns]
    if not sentiment_cols:
        raise ValueError("No sentiment columns found.")

    feature_cols = lag_features + sentiment_cols

    # === Split masks
    train_mask = (df["trade_date"] >= train_start) & (df["trade_date"] <= train_end)
    test_mask = (df["trade_date"] >= test_start) & (df["trade_date"] <= test_end)

    # === Impute missing values from training set only
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(df.loc[train_mask, feature_cols])
    df[feature_cols] = imputer.transform(df[feature_cols])

    # === Prepare data
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "increase"]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, "increase"]

    # === Class imbalance handling
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    # === Train XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # === Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"✅ AUC: {roc_auc_score(y_test, y_proba):.4f}")
    ll = log_loss(y_test, y_proba)
    print(f"✅ Log Loss: {ll:.4f}")
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print("=== Top Feature Importances ===")
    print(pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10))

    # === Save predictions
    output = df.loc[test_mask, ["trade_date"] + feature_cols].copy()
    output["actual_increase"] = y_test.values
    output["predicted_proba"] = y_proba
    output["predicted_label"] = y_pred
    save_path = f"predictions/{model_name}_xgb.csv"
    output.to_csv(save_path, index=False)
    print(f"📁 Saved predictions to {save_path}")

    # === Save model
    joblib.dump(model, f"model_weights/{model_name}_xgb.pkl")
    print(f"💾 Model artifacts saved to /model_weights")


def runLM(model_name='lm', sentiment_column='avg_sentiment_lm', db_path='data/snp500.db', table_name='snp500',
          train_start='2018-01-01', train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31',
          n_days_lag=3, sources=["bloomberg", "businessinsider", "wsj"]):
    xgb_baseline(db_path, table_name, 'baseline_returns_only_xgb', train_start, train_end, test_start, test_end, n_days_lag)
    xgb_consolidated_sentiment(db_path, table_name, sentiment_column, model_name, train_start, train_end, test_start, test_end, n_days_lag)
    xgb_sentiment_by_source(db_path, table_name, sentiment_column, model_name, sources, train_start, train_end, test_start, test_end, n_days_lag)

def runFinbert(model_name='finbert', sentiment_column='avg_sentiment_finbert', db_path='data/snp500.db', table_name='snp500',
               train_start='2018-01-01', train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31',
               n_days_lag=3, sources=["bloomberg", "businessinsider", "wsj"]):
    xgb_baseline(db_path, table_name, 'baseline_returns_only_xgb', train_start, train_end, test_start, test_end, n_days_lag)
    xgb_consolidated_sentiment(db_path, table_name, sentiment_column, model_name, train_start, train_end, test_start, test_end, n_days_lag)
    xgb_sentiment_by_source(db_path, table_name, sentiment_column, model_name, sources, train_start, train_end, test_start, test_end, n_days_lag)

def runGPT4o(model_name='gpt4o', sentiment_column='avg_sentiment_llm_gpt4o', db_path='data/snp500.db', table_name='snp500',
             train_start='2018-01-01', train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31',
             n_days_lag=3, sources=["bloomberg", "businessinsider", "wsj"]):
    xgb_baseline(db_path, table_name, 'baseline_returns_only_xgb', train_start, train_end, test_start, test_end, n_days_lag)
    xgb_consolidated_sentiment(db_path, table_name, sentiment_column, model_name, train_start, train_end, test_start, test_end, n_days_lag)
    xgb_sentiment_by_source(db_path, table_name, sentiment_column, model_name, sources, train_start, train_end, test_start, test_end, n_days_lag)

def runGPT41mini(model_name='gpt41mini', sentiment_column='avg_sentiment_llm_gpt41mini', db_path='data/snp500.db', table_name='snp500',
                 train_start='2018-01-01', train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31',
                 n_days_lag=3, sources=["bloomberg", "businessinsider", "wsj"]):
    xgb_baseline(db_path, table_name, 'baseline_returns_only_xgb', train_start, train_end, test_start, test_end, n_days_lag)
    xgb_consolidated_sentiment(db_path, table_name, sentiment_column, model_name, train_start, train_end, test_start, test_end, n_days_lag)
    xgb_sentiment_by_source(db_path, table_name, sentiment_column, model_name, sources, train_start, train_end, test_start, test_end, n_days_lag)

def runGPT41miniMarket(model_name='gpt41mini_market', sentiment_column='avg_sentiment_llm_gpt41mini_market', db_path='data/snp500.db', table_name='snp500',
                       train_start='2018-01-01', train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31',
                       n_days_lag=3, sources=["bloomberg", "businessinsider", "wsj"]):
    xgb_baseline(db_path, table_name, 'baseline_returns_only_xgb', train_start, train_end, test_start, test_end, n_days_lag)
    xgb_consolidated_sentiment(db_path, table_name, sentiment_column, model_name, train_start, train_end, test_start, test_end, n_days_lag)
    xgb_sentiment_by_source(db_path, table_name, sentiment_column, model_name, sources, train_start, train_end, test_start, test_end, n_days_lag)

db_path = 'data/snp500.db'
table_name = 'snp500'
train_start = '2018-01-01'
train_end = '2022-12-31'
test_start = '2023-01-01'
test_end = '2023-12-31'
n_days_lag = 3

runLM(db_path=db_path, table_name=table_name, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, n_days_lag=n_days_lag)
runFinbert(db_path=db_path, table_name=table_name, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, n_days_lag=n_days_lag)
runGPT4o(db_path=db_path, table_name=table_name, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, n_days_lag=n_days_lag)
runGPT41mini(db_path=db_path, table_name=table_name, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, n_days_lag=n_days_lag)
runGPT41miniMarket(db_path=db_path, table_name=table_name, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, n_days_lag=n_days_lag)