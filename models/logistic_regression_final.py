import sqlite3
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, log_loss

def baseline_logistic_regression(
    db_path: str,
    table_name: str,
    model_name: str = 'baseline_logistic_regression',
    train_start: str = '2018-01-02',
    train_end: str = '2022-12-31',
    test_start: str = '2023-01-01',
    test_end: str = '2023-12-31',
    n_days_lag: int = 5
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

    # === Compute lag features from daily_return ===
    lag_features = []
    for lag in range(1, n_days_lag + 1):
        col = f'prev_return_t-{lag}'
        df[col] = df['daily_return'].shift(lag)
        lag_features.append(col)

    # === Create interaction terms ===
    interaction_terms = []
    for i in range(len(lag_features)):
        for j in range(i + 1, len(lag_features)):
            col1 = lag_features[i]
            col2 = lag_features[j]
            inter_col = f"{col1}*{col2}"
            df[inter_col] = df[col1] * df[col2]
            interaction_terms.append(inter_col)

    # === Final feature list ===
    feature_cols = lag_features + interaction_terms
    df = df.dropna(subset=lag_features)

    # === Train-test split ===
    mask_train = (df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)
    mask_test = (df['trade_date'] >= test_start) & (df['trade_date'] <= test_end)

    X_train = df.loc[mask_train, feature_cols]
    y_train = df.loc[mask_train, 'increase']
    X_test = df.loc[mask_test, feature_cols]
    y_test = df.loc[mask_test, 'increase']

    # === Scale ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Grid search over C and l1_ratio ===
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1],
        'l1_ratio': [0, 0.25, 0.5, 0.75, 1]
    }
    grid = GridSearchCV(
        LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        param_grid,
        scoring='roc_auc',
        cv=TimeSeriesSplit(n_splits=5)
    )
    grid.fit(X_train_scaled, y_train)
    model = grid.best_estimator_

    print(f"✅ Best Params: {grid.best_params_}")
    # === Evaluate ===
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"✅ Best C: {grid.best_params_['C']}")
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ AUC: {auc:.4f}")
    ll = log_loss(y_test, y_proba)
    print(f"✅ Log Loss: {ll:.4f}")
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("=== Coefficients ===")
    print(pd.Series(model.coef_[0], index=feature_cols))
    print(f"Intercept: {model.intercept_[0]:.4f}")



    # === Save predictions ===
    output = df.loc[mask_test, ['trade_date']].copy()
    output['actual_increase'] = y_test.values
    output['predicted_proba'] = y_proba
    output['predicted_label'] = y_pred
    save_csv = f"predictions/{model_name}_nlag{n_days_lag}.csv"
    output.to_csv(save_csv, index=False)
    print(f"📁 Saved predictions to {save_csv}")

    # === Save model ===
    joblib.dump(model, f"model_weights/{model_name}_logreg.pkl")
    joblib.dump(scaler, f"model_weights/{model_name}_scaler.pkl")
    print(f"💾 Model artifacts saved to /model_weights")

def logistic_regression_consolidated_sentiment(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    model_name: str,
    train_start: str = '2018-01-02',
    train_end: str = '2022-12-31',
    test_start: str = '2023-01-01',
    test_end: str = '2023-12-31',
    n_days_lag: int = 5
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

    df = df.loc[:, ~df.columns.duplicated()]
    if sentiment_column not in df.columns:
        raise ValueError(f"Sentiment column '{sentiment_column}' not found.")
    if 'daily_return' not in df.columns:
        raise ValueError("Missing 'daily_return' column.")

    # === Impute sentiment ===
    df[sentiment_column] = df[sentiment_column].fillna(df[sentiment_column].mean())

    # === Lag return features and interaction terms ===
    df = df.sort_values('trade_date').reset_index(drop=True)

    lag_features = []
    for lag in range(1, n_days_lag + 1):
        lag_col = f'prev_return_t-{lag}'
        df[lag_col] = df['daily_return'].shift(lag)
        lag_features.append(lag_col)

    interaction_terms = []

    # 1. Sentiment × Lagged Return
    for lag_col in lag_features:
        inter_col = f"{sentiment_column}*{lag_col}"
        df[inter_col] = df[sentiment_column] * df[lag_col]
        interaction_terms.append(inter_col)

    # 2. Lagged Return × Lagged Return
    for i in range(len(lag_features)):
        for j in range(i + 1, len(lag_features)):
            col1 = lag_features[i]
            col2 = lag_features[j]
            inter_col = f"{col1}*{col2}"
            df[inter_col] = df[col1] * df[col2]
            interaction_terms.append(inter_col)

    # Drop rows with missing lag returns (which also ensures interaction terms are valid)
    df = df.dropna(subset=lag_features)

    # === Feature matrix ===
    feature_cols = [sentiment_column] + lag_features + interaction_terms
    X = df[feature_cols]
    y = df['increase']

    # === Train-test split ===
    date_mask_train = (df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)
    date_mask_test = (df['trade_date'] >= test_start) & (df['trade_date'] <= test_end)

    X_train = X[date_mask_train]
    y_train = y[date_mask_train]
    X_test = X[date_mask_test]
    y_test = y[date_mask_test]

    # === Standardize ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Grid search over C and l1_ratio ===
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1],
        'l1_ratio': [0, 0.25, 0.5, 0.75, 1]
    }
    grid = GridSearchCV(
        LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        param_grid,
        scoring='roc_auc',
        cv=TimeSeriesSplit(n_splits=5)
    )
    grid.fit(X_train_scaled, y_train)
    model = grid.best_estimator_

    print(f"✅ Best Params: {grid.best_params_}")

    # === Evaluate ===
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"✅ Best C: {grid.best_params_['C']}")
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ AUC: {auc:.4f}")
    ll = log_loss(y_test, y_proba)
    print(f"✅ Log Loss: {ll:.4f}")
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("=== Coefficients ===")
    print(pd.Series(model.coef_[0], index=feature_cols))

    # === Save predictions ===
    output = df.loc[date_mask_test, ['trade_date']].copy()
    output['actual_increase'] = y_test.values
    output['predicted_proba'] = y_proba
    output['predicted_label'] = y_pred
    save_csv = f"predictions/{model_name}_nlag{n_days_lag}.csv"
    output.to_csv(save_csv, index=False)
    print(f"📁 Saved predictions to {save_csv}")

    # === Save model ===
    joblib.dump(model, f"model_weights/{model_name}_logreg.pkl")
    joblib.dump(scaler, f"model_weights/{model_name}_scaler.pkl")
    print(f"💾 Model artifacts saved to /model_weights")

def logistic_regression_sentiment_sources(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    model_name: str,
    sources: list = ["bloomberg", "businessinsider", "wsj"], 
    train_start: str = '2018-01-02',
    train_end: str = '2022-12-31',
    test_start: str = '2023-01-01',
    test_end: str = '2023-12-31',
    n_days_lag: int = 5
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

    df = df.loc[:, ~df.columns.duplicated()]
    if 'daily_return' not in df.columns:
        raise ValueError("Missing required column: 'daily_return'.")

    # === Sentiment columns by source ===
    sentiment_cols = [f"{sentiment_column}_{src}" for src in sources if f"{sentiment_column}_{src}" in df.columns]
    if not sentiment_cols:
        raise ValueError(f"No matching sentiment columns found for selected sources: {sources}")

    # === Mean impute missing sentiment ===
    for col in sentiment_cols:
        df[col] = df[col].fillna(df[col].mean())

    # === Lagged returns from daily_return ===
    df = df.sort_values('trade_date').reset_index(drop=True)
    lag_features = []
    for lag in range(1, n_days_lag + 1):
        col = f'prev_return_t-{lag}'
        df[col] = df['daily_return'].shift(lag)
        lag_features.append(col)

    # === Interaction terms ===
    interaction_terms = []

    # 1. Sentiment × Lagged Return
    for s_col in sentiment_cols:
        for r_col in lag_features:
            inter_col = f"{s_col}*{r_col}"
            df[inter_col] = df[s_col] * df[r_col]
            interaction_terms.append(inter_col)

    # 2. Lagged Return × Lagged Return
    for i in range(len(lag_features)):
        for j in range(i + 1, len(lag_features)):
            col1 = lag_features[i]
            col2 = lag_features[j]
            inter_col = f"{col1}*{col2}"
            df[inter_col] = df[col1] * df[col2]
            interaction_terms.append(inter_col)

    # === Final features ===
    feature_cols = sentiment_cols + lag_features + interaction_terms
    df = df.dropna(subset=lag_features)

    # === Train-test split ===
    train_mask = (df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)
    test_mask = (df['trade_date'] >= test_start) & (df['trade_date'] <= test_end)

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'increase']
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, 'increase']

    # === Standardize ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === Grid search over C and l1_ratio ===
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1],
        'l1_ratio': [0, 0.25, 0.5, 0.75, 1]
    }
    grid = GridSearchCV(
        LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        param_grid,
        scoring='roc_auc',
        cv=TimeSeriesSplit(n_splits=5)
    )
    grid.fit(X_train_scaled, y_train)
    model = grid.best_estimator_

    print(f"✅ Best Params: {grid.best_params_}")
    # === Evaluate ===
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"✅ Best C: {grid.best_params_['C']}")
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ AUC: {auc:.4f}")
    ll = log_loss(y_test, y_proba)
    print(f"✅ Log Loss: {ll:.4f}")    
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("=== Coefficients ===")
    print(pd.Series(model.coef_[0], index=feature_cols))

    # === Save outputs ===
    output = df.loc[test_mask, ['trade_date']].copy()
    output['actual_increase'] = y_test.values
    output['predicted_proba'] = y_proba
    output['predicted_label'] = y_pred
    save_csv = f"predictions/{model_name}_gridsearch.csv"
    output.to_csv(save_csv, index=False)
    print(f"📁 Saved predictions to {save_csv}")

    # === Save artifacts ===
    joblib.dump(model, f"model_weights/{model_name}_logreg.pkl")
    joblib.dump(scaler, f"model_weights/{model_name}_scaler.pkl")
    print(f"💾 Model and preprocessing artifacts saved to /model_weights")

def runLM(model_name='lm', sentiment_column = 'avg_sentiment_lm', db_path='data/snp500.db', table_name='snp500', train_start='2018-01-01', train_end='2022-12-31', test_start='2023-01-01', test_end='2023-12-31', n_days_lag=5, sources = ["bloomberg", "businessinsider", "wsj"]):

    logistic_regression_consolidated_sentiment(
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
    logistic_regression_sentiment_sources(
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
    logistic_regression_consolidated_sentiment(
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
    logistic_regression_sentiment_sources(
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
    logistic_regression_consolidated_sentiment(
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
    logistic_regression_sentiment_sources(
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

    logistic_regression_consolidated_sentiment(
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
    logistic_regression_sentiment_sources(
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
    logistic_regression_consolidated_sentiment(
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
    logistic_regression_sentiment_sources(
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

baseline_logistic_regression(
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