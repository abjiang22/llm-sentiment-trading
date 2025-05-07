import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

def classical_pipeline_with_headline_embeddings_and_sentiment(
    news_db="data/news.db",
    news_table="master0",
    snp_db="data/snp500.db",
    snp_table="snp500",
    embed_model_name="all-MiniLM-L6-v2",
    sentiment_cols=["avg_sentiment_gpt4o_zero_shot"],
    n_top=3,
    n_days_lag=3,
    train_start_date="2018-01-01",
    train_end_date="2022-12-31",
    test_start_date="2023-01-01",
    test_end_date="2024-12-31"
):
    # === Load headline + sentiment data ===
    news_conn = sqlite3.connect(news_db)
    query = f"SELECT title_clean, published_at, {', '.join(sentiment_cols)} FROM {news_table}"
    headlines = pd.read_sql_query(query, news_conn)
    news_conn.close()

    headlines["published_at"] = pd.to_datetime(headlines["published_at"])
    headlines["trade_date"] = headlines["published_at"].dt.date

    # === Load S&P 500 data ===
    snp_conn = sqlite3.connect(snp_db)
    snp = pd.read_sql_query(
        f"SELECT trade_date, open_price, close_price, increase FROM {snp_table}",
        snp_conn,
        parse_dates=["trade_date"]
    )
    snp_conn.close()
    snp["trade_date"] = snp["trade_date"].dt.date

    # === Filter and merge ===
    df = headlines.merge(snp, on="trade_date", how="inner")
    df = df.dropna(subset=["title_clean"])

    # === Convert sentiment columns to numeric ===
    for col in sentiment_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # === Compute embeddings ===
    model = SentenceTransformer(embed_model_name)
    embeddings = model.encode(df["title_clean"].tolist(), convert_to_numpy=True)
    df["embedding"] = list(embeddings)

    # === Group by trade_date to summarize embeddings ===
    grouped = df.groupby("trade_date")["embedding"].agg(list).reset_index()

    def get_summary_features(emb_list):
        emb_arr = np.vstack(emb_list)
        return np.concatenate([
            emb_arr.mean(axis=0),
            emb_arr.max(axis=0),
            emb_arr.min(axis=0),
            np.sort(np.linalg.norm(emb_arr, axis=1))[-n_top:],
        ])

    grouped["features"] = grouped["embedding"].apply(get_summary_features)
    grouped = grouped[["trade_date", "features"]]

    # === Aggregate sentiment scores per day ===
    daily_sentiment = df.groupby("trade_date")[sentiment_cols].mean().reset_index()

    # === Merge into S&P base ===
    snp = snp.merge(grouped, on="trade_date", how="left")
    snp = snp.merge(daily_sentiment, on="trade_date", how="left")
    snp = snp.dropna(subset=["features"] + sentiment_cols)

    # === Add lagged return features ===
    for lag in range(1, n_days_lag + 1):
        snp[f"prev_return_t-{lag}"] = (
            snp["close_price"].shift(lag) - snp["open_price"].shift(lag)
        ) / snp["open_price"].shift(lag)

    lag_cols = [f"prev_return_t-{i}" for i in range(1, n_days_lag + 1)]
    snp = snp.dropna(subset=lag_cols)

    # === Final feature matrix ===
    X_embed = np.vstack(snp["features"])
    X_returns = snp[lag_cols].values
    X_sentiment = snp[sentiment_cols].values
    X = np.hstack([X_embed, X_returns, X_sentiment])
    y = snp["increase"].astype(int)

    # === Train/test split with full date ranges ===
    train_start_date = pd.to_datetime(train_start_date).date()
    train_end_date = pd.to_datetime(train_end_date).date()
    test_start_date = pd.to_datetime(test_start_date).date()
    test_end_date = pd.to_datetime(test_end_date).date()

    train_mask = (snp["trade_date"] >= train_start_date) & (snp["trade_date"] <= train_end_date)
    test_mask = (snp["trade_date"] >= test_start_date) & (snp["trade_date"] <= test_end_date)

    scaler = StandardScaler()
    X[train_mask] = scaler.fit_transform(X[train_mask])
    X[test_mask] = scaler.transform(X[test_mask])

    # === Train + Evaluate ===
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
    model.fit(X[train_mask], y[train_mask])

    preds = model.predict(X[test_mask])
    probs = model.predict_proba(X[test_mask])[:, 1]

    print(f"✅ Accuracy: {accuracy_score(y[test_mask], preds):.4f}")
    print(f"📈 AUC: {roc_auc_score(y[test_mask], probs):.4f}")


classical_pipeline_with_headline_embeddings_and_sentiment(
    news_db="data/news_5_6.db",
    news_table="master0",
    snp_db="data/snp500_5_6.db",
    snp_table="snp500",
    embed_model_name="all-MiniLM-L6-v2",
    sentiment_cols=["llm_sentiment_score_gpt4o_zero_shot"],
    n_top=3,
    n_days_lag=3,
    train_start_date="2018-01-01",
    train_end_date="2022-12-31",
    test_start_date="2023-01-01",
    test_end_date="2024-12-31"
)