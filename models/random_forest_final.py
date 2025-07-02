import sqlite3
import pandas as pd
import numpy as np
import joblib
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, log_loss
import os
import sqlite3
import pandas as pd
import numpy as np
import joblib

def rf_baseline(
    db_path: str,
    table_name: str,
    model_name: str = 'rf_baseline',
    train_start: str = '2018-01-01',
    train_end: str = '2022-12-31',
    test_start: str = '2023-01-01',
    test_end: str = '2023-12-31',
    n_days_lag: int = 3
):
    
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("model_weights", exist_ok=True)

    # === Load data ===
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"SELECT * FROM {table_name} WHERE increase IS NOT NULL AND daily_return IS NOT NULL",
        conn, parse_dates=["trade_date"]
    )
    conn.close()
    df = df.sort_values("trade_date").reset_index(drop=True)

    # === Generate lag features
    lag_features = []
    for lag in range(1, n_days_lag + 1):
        col = f'prev_return_t-{lag}'
        df[col] = df['daily_return'].shift(lag)
        lag_features.append(col)

    # === Drop rows with missing lags
    df = df.dropna(subset=lag_features)

    # === Train-test split
    mask_train = (df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)
    mask_test = (df['trade_date'] >= test_start) & (df['trade_date'] <= test_end)

    X_train = df.loc[mask_train, lag_features]
    y_train = df.loc[mask_train, 'increase']
    X_test = df.loc[mask_test, lag_features]
    y_test = df.loc[mask_test, 'increase']

    # === Grid search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 10],
        'min_samples_split': [5, 10, 20],
        'max_features': ['sqrt', 'log2']
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=TimeSeriesSplit(n_splits=5),
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    print(f"✅ Best Params: {grid.best_params_}")

    # === Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ AUC: {auc:.4f}")
    ll = log_loss(y_test, y_proba)
    print(f"✅ Log Loss: {ll:.4f}")
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print("=== Feature Importances ===")
    print(pd.Series(model.feature_importances_, index=lag_features).sort_values(ascending=False))

    # === Save predictions
    output = df.loc[mask_test, ['trade_date']].copy()
    output['actual_increase'] = y_test.values
    output['predicted_proba'] = y_proba
    output['predicted_label'] = y_pred
    save_csv = f"predictions/{model_name}_nlag{n_days_lag}.csv"
    output.to_csv(save_csv, index=False)
    print(f"📁 Saved predictions to {save_csv}")

    # === Save model
    joblib.dump(model, f"model_weights/{model_name}_rf.pkl")
    print(f"💾 Model artifacts saved to /model_weights")

def rf_consolidated_sentiment(
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
        f"SELECT * FROM {table_name} WHERE increase IS NOT NULL",
        conn, parse_dates=["trade_date"]
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
        df[col] = (
            (df["close_price"].shift(lag) - df["open_price"].shift(lag)) / df["open_price"].shift(lag)
        )
        lag_features.append(col)

    # === Define features
    feature_cols = lag_features + [sentiment_column]

    # === Impute missing values from training set
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(df.loc[:, feature_cols])
    df[feature_cols] = imputer.transform(df[feature_cols])

    # === Train/test split
    mask_train = (df["trade_date"] >= train_start) & (df["trade_date"] <= train_end)
    mask_test = (df["trade_date"] >= test_start) & (df["trade_date"] <= test_end)

    X_train = df.loc[mask_train, feature_cols]
    y_train = df.loc[mask_train, "increase"]
    X_test = df.loc[mask_test, feature_cols]
    y_test = df.loc[mask_test, "increase"]

    # === Grid search with time series CV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=TimeSeriesSplit(n_splits=5),
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    print(f"✅ Best Params: {grid.best_params_}")

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
    output = df.loc[mask_test, ["trade_date"] + feature_cols].copy()
    output["actual_increase"] = y_test.values
    output["predicted_proba"] = y_proba
    output["predicted_label"] = y_pred
    save_path = f"predictions/{model_name}_rf.csv"
    output.to_csv(save_path, index=False)
    print(f"📁 Saved predictions to {save_path}")

    # === Save model artifacts
    joblib.dump(model, f"model_weights/{model_name}_rf.pkl")
    print(f"💾 Model artifacts saved to /model_weights")

def rf_sentiment_by_source(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    model_name: str,
    sources: list = ["bloomberg", "wsj", "businessinsider"],
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
        f"SELECT * FROM {table_name} WHERE increase IS NOT NULL",
        conn, parse_dates=["trade_date"]
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

    # === Grid search for best RF params
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=TimeSeriesSplit(n_splits=5),
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    print(f"✅ Best Params: {grid.best_params_}")

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
    save_path = f"predictions/{model_name}_rf.csv"
    output.to_csv(save_path, index=False)
    print(f"📁 Saved predictions to {save_path}")

    # === Save model
    joblib.dump(model, f"model_weights/{model_name}_rf.pkl")
    print(f"💾 Model artifacts saved to /model_weights")

def runLM(model_name='lm', sentiment_column = 'avg_sentiment_lm', db_path='data/snp500.db', table_name='snp500', train_start='2018-01-01', train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31', n_days_lag=5, sources = ["bloomberg", "businessinsider", "wsj"]):

    rf_consolidated_sentiment(
        db_path=db_path,
        table_name=table_name,
        sentiment_column=sentiment_column,
        model_name=model_name,
        train_start = train_start,
        train_end = train_end,
        test_start = test_start,
        test_end = test_end,
        n_days_lag=n_days_lag,
    )
    rf_sentiment_by_source(
        db_path=db_path,
        table_name=table_name,
        sentiment_column=sentiment_column,
        model_name=model_name,
        sources = sources, 
        train_start = train_start,
        train_end = train_end,
        test_start = test_start,
        test_end = test_end,
        n_days_lag=n_days_lag,
    )

def runFinbert(model_name='finbert', sentiment_column = 'avg_sentiment_finbert', db_path='data/snp500.db', table_name='snp500', train_start='2018-01-01', train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31', n_days_lag=5, sources = ["bloomberg", "businessinsider", "wsj"]):

    rf_consolidated_sentiment(
        db_path=db_path,
        table_name=table_name,
        sentiment_column=sentiment_column,
        model_name=model_name,
        train_start = train_start,
        train_end = train_end,
        test_start = test_start,
        test_end = test_end,
        n_days_lag=n_days_lag,
    )
    rf_sentiment_by_source(
        db_path=db_path,
        table_name=table_name,
        sentiment_column=sentiment_column,
        model_name=model_name,
        sources = sources, 
        train_start = train_start,
        train_end = train_end,
        test_start = test_start,
        test_end = test_end,
        n_days_lag=n_days_lag,
    )

def runGPT4o(model_name='gpt4o', sentiment_column = 'avg_sentiment_llm_gpt4o', db_path='data/snp500.db', table_name='snp500', train_start='2018-01-01', train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31', n_days_lag=5, sources = ["bloomberg", "businessinsider", "wsj"]):


    rf_consolidated_sentiment(
        db_path=db_path,
        table_name=table_name,
        sentiment_column=sentiment_column,
        model_name=model_name,
        train_start = train_start,
        train_end = train_end,
        test_start = test_start,
        test_end = test_end,
        n_days_lag=n_days_lag,
    )
    rf_sentiment_by_source(
        db_path=db_path,
        table_name=table_name,
        sentiment_column=sentiment_column,
        model_name=model_name,
        sources = sources, 
        train_start = train_start,
        train_end = train_end,
        test_start = test_start,
        test_end = test_end,
        n_days_lag=n_days_lag,
    )

def runGPT41mini(model_name='gpt41mini', sentiment_column = 'avg_sentiment_llm_gpt41mini', db_path='data/snp500.db', table_name='snp500', train_start='2018-01-01', train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31', n_days_lag=5, sources = ["bloomberg", "businessinsider", "wsj"]):

    rf_consolidated_sentiment(
        db_path=db_path,
        table_name=table_name,
        sentiment_column=sentiment_column,
        model_name=model_name,
        train_start = train_start,
        train_end = train_end,
        test_start = test_start,
        test_end = test_end,
        n_days_lag=n_days_lag,
    )
    rf_sentiment_by_source(
        db_path=db_path,
        table_name=table_name,
        sentiment_column=sentiment_column,
        model_name=model_name,
        sources = sources, 
        train_start = train_start,
        train_end = train_end,
        test_start = test_start,
        test_end = test_end,
        n_days_lag=n_days_lag,
    )

def runGPT41miniMarket(model_name='gpt41mini_market', sentiment_column = 'avg_sentiment_llm_gpt41mini_market', db_path='data/snp500.db', table_name='snp500', train_start='2018-01-01', train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31', n_days_lag=5, sources = ["bloomberg", "businessinsider", "wsj"]):

    rf_consolidated_sentiment(
        db_path=db_path,
        table_name=table_name,
        sentiment_column=sentiment_column,
        model_name=model_name,
        train_start = train_start,
        train_end = train_end,
        test_start = test_start,
        test_end = test_end,
        n_days_lag=n_days_lag,
    )
    rf_sentiment_by_source(
        db_path=db_path,
        table_name=table_name,
        sentiment_column=sentiment_column,
        model_name=model_name,
        sources = sources, 
        train_start = train_start,
        train_end = train_end,
        test_start = test_start,
        test_end = test_end,
        n_days_lag=n_days_lag,
    )

db_path = 'data/snp500.db'
table_name = 'snp500'
train_start = '2018-01-01'
train_end = '2022-12-31'
test_start = '2023-01-01'
test_end = '2023-12-31'
n_days_lag = 3

rf_baseline(
db_path=db_path,
table_name=table_name,
model_name='baseline_returns_only',
train_start = train_start,
train_end = train_end,
test_start = test_start,
test_end = test_end,
n_days_lag=n_days_lag
)

runLM(db_path=db_path, table_name=table_name, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, n_days_lag=n_days_lag)
runFinbert(db_path=db_path, table_name=table_name, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, n_days_lag=n_days_lag)
runGPT4o(db_path=db_path, table_name=table_name, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, n_days_lag=n_days_lag)
runGPT41mini(db_path=db_path, table_name=table_name, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, n_days_lag=n_days_lag)
runGPT41miniMarket(db_path=db_path, table_name=table_name, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, n_days_lag=n_days_lag)