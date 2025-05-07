import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

def predict_increase_with_random_forest(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    model_name: str,
    sources: list = ["financialpost", "bloomberg", "wsj", "businessinsider"], 
    train_start: str = '2018-01-02',
    train_end: str = '2022-12-31',
    test_start: str = '2023-01-01',
    test_end: str = '2024-12-31',
    z_clip_threshold: float = 3.0,
    n_days_lag: int = 3
):
    import sqlite3, joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    # === Load data ===
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"SELECT * FROM {table_name} WHERE increase IS NOT NULL",
        conn, parse_dates=["trade_date"]
    )
    conn.close()

    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_values('trade_date').reset_index(drop=True)

    if 'open_price' not in df.columns or 'close_price' not in df.columns:
        raise ValueError("Missing required columns: 'open_price' and/or 'close_price'.")

    # === Build lagged return features only ===
    for lag in range(1, n_days_lag + 1):
        df[f'prev_return_t-{lag}'] = (
            df['close_price'].shift(lag) - df['open_price'].shift(lag)
        ) / df['open_price'].shift(lag)
    lag_features = [f'prev_return_t-{i}' for i in range(1, n_days_lag + 1)]

    # === Sentiment columns ===
    sentiment_cols = [f"{sentiment_column}_{src}" for src in sources if f"{sentiment_column}_{src}" in df.columns]
    if not sentiment_cols:
        raise ValueError(f"No matching sentiment columns found for selected sources: {sources}")

    feature_cols = sentiment_cols + lag_features
    df = df.dropna(subset=feature_cols)

    # === Train/test split ===
    train_mask = (df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)
    test_mask = (df['trade_date'] >= test_start) & (df['trade_date'] <= test_end)

    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feature_cols])

    df_std = df.copy()
    df_std[feature_cols] = pd.DataFrame(
        scaler.transform(df[feature_cols]),
        columns=feature_cols,
        index=df.index
    )
    df_std[sentiment_cols] = df_std[sentiment_cols].clip(-z_clip_threshold, z_clip_threshold)

    train = df_std[train_mask]
    test = df_std[test_mask]

    X_train = train[feature_cols]
    y_train = train['increase']
    X_test = test[feature_cols]
    y_test = test['increase']

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"🌲 Random Forest Accuracy: {acc:.4f}")
    print(f"🌲 Random Forest AUC: {auc:.4f}")
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    feature_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("=== Top Feature Importances ===")
    print(feature_importance.head(10))

    output = test[['trade_date'] + feature_cols].copy()
    output['actual_increase'] = y_test.values
    output['predicted_proba'] = y_proba
    output['predicted_label'] = y_pred
    save_csv = f"predictions/{model_name}_rf_returns_only.csv"
    output.to_csv(save_csv, index=False)
    print(f"📁 Saved predictions to {save_csv}")

    joblib.dump(model, f"models/{model_name}_rf.pkl")
    joblib.dump(scaler, f"models/{model_name}_scaler.pkl")
    print(f"💾 Model and scaler saved to /models")

def compare_rf_sentiment_vs_price_only_returns_only_strict(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    model_name: str,
    sources: list = ["financialpost", "bloomberg", "wsj", "businessinsider"],
    train_start: str = '2018-01-02',
    train_end: str = '2022-12-31',
    test_start: str = '2023-01-01',
    test_end: str = '2024-12-31',
    z_clip_threshold: float = 3.0,
    n_days_lag: int = 3
):
    import sqlite3, joblib
    import pandas as pd, numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    def bootstrap_auc_diff(y_true, y_proba_1, y_proba_2, n_bootstrap=10000):
        np.random.seed(42)
        auc_diffs = []
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, len(y_true), len(y_true))
            auc1 = roc_auc_score(y_true[idx], y_proba_1[idx])
            auc2 = roc_auc_score(y_true[idx], y_proba_2[idx])
            auc_diffs.append(auc1 - auc2)
        observed = roc_auc_score(y_true, y_proba_1) - roc_auc_score(y_true, y_proba_2)
        pval = np.mean(np.abs(auc_diffs) >= np.abs(observed))
        print(f"\n🎯 AUC diff (with - without): {observed:.4f}")
        print(f"📊 Bootstrap p-value: {pval:.4f}")
        return observed, pval

    # === Load and clean ===
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE increase IS NOT NULL", conn, parse_dates=["trade_date"])
    conn.close()
    df = df.loc[:, ~df.columns.duplicated()].sort_values("trade_date").reset_index(drop=True)

    if "open_price" not in df.columns or "close_price" not in df.columns:
        raise ValueError("Missing open/close_price")

    # === Generate prev_return_t-* features only
    for lag in range(1, n_days_lag + 1):
        df[f"prev_return_t-{lag}"] = (df["close_price"].shift(lag) - df["open_price"].shift(lag)) / df["open_price"].shift(lag)
    lag_features = [f"prev_return_t-{i}" for i in range(1, n_days_lag + 1)]

    # === Sentiment features
    sentiment_cols = [f"{sentiment_column}_{s}" for s in sources if f"{sentiment_column}_{s}" in df.columns]
    if not sentiment_cols:
        raise ValueError("No sentiment columns found.")

    # === All features for full filtering
    feature_cols = lag_features + sentiment_cols
    df = df.dropna(subset=feature_cols).copy()

    # === Train/test split masks (shared)
    train_mask = (df["trade_date"] >= train_start) & (df["trade_date"] <= train_end)
    test_mask = (df["trade_date"] >= test_start) & (df["trade_date"] <= test_end)

    # === Fit single scaler on full feature set
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feature_cols])
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    df_scaled[sentiment_cols] = df_scaled[sentiment_cols].clip(-z_clip_threshold, z_clip_threshold)

    results = {}
    for label, used_features in [
        ("with_sentiment", lag_features + sentiment_cols),
        ("without_sentiment", lag_features)
    ]:
        train = df_scaled[train_mask]
        test = df_scaled[test_mask]

        X_train = train[used_features]
        y_train = train["increase"]
        X_test = test[used_features]
        y_test = test["increase"]

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"\n🌲 Random Forest ({label})")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
        print("=== Classification Report ===")
        print(classification_report(y_test, y_pred, digits=4))
        print("=== Top Feature Importances ===")
        print(pd.Series(model.feature_importances_, index=used_features).sort_values(ascending=False).head(10))

        results[label] = {
            "y_true": y_test.reset_index(drop=True),
            "y_proba": pd.Series(y_proba, index=y_test.index).reset_index(drop=True)
        }

        save_path = f"predictions/{model_name}_{label}.csv"
        output = test[["trade_date"] + used_features].copy()
        output["actual_increase"] = y_test.values
        output["predicted_proba"] = y_proba
        output["predicted_label"] = y_pred
        output.to_csv(save_path, index=False)
        joblib.dump(model, f"models/{model_name}_{label}.pkl")
        joblib.dump(scaler, f"models/{model_name}_{label}_scaler.pkl")
        print(f"📁 Saved predictions and model for: {label}")

    if "with_sentiment" in results and "without_sentiment" in results:
        bootstrap_auc_diff(
            results["with_sentiment"]["y_true"].values,
            results["with_sentiment"]["y_proba"].values,
            results["without_sentiment"]["y_proba"].values
        )
"""
predict_increase_with_random_forest(
    db_path="data/snp500.db",
    table_name="snp500",
    sentiment_column="avg_sentiment_gpt4o_zero_shot",
    model_name="rf_gpt4o_price_lag3",
    sources = ["financialpost", "bloomberg", "businessinsider"], 
    train_start = '2018-01-02',
    train_end = '2022-12-31',
    test_start = '2023-01-01',
    test_end = '2024-12-31',
    n_days_lag=3
)
"""

compare_rf_sentiment_vs_price_only_returns_only_strict(
    db_path="data/snp500_5_6.db",
    table_name="snp500",
    sentiment_column="avg_sentiment_gpt4o_zero_shot",
    model_name="rf_fixed_final",
    sources=["bloomberg",  "businessinsider", "financialpost"],
    train_start='2018-01-01',
    train_end='2022-12-31',
    test_start='2023-01-01',
    test_end='2024-12-31',
    n_days_lag=3
)

