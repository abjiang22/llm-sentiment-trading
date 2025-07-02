import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


class HeadlineDualAttentionModel(nn.Module):
    def __init__(self, embed_dim: int, sentiment_dim: int, n_lags: int, proj_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.sentiment_dim = sentiment_dim
        self.n_lags = n_lags

        self.sentiment_proj = nn.Sequential(
            nn.Linear(sentiment_dim, proj_dim),
            nn.ReLU()
        )

        self.text_attn = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.sent_attn = nn.Sequential(
            nn.Linear(proj_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + proj_dim + n_lags, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, headline_features, lagged_returns):
        embeddings = headline_features[..., :-self.sentiment_dim]
        sentiments = headline_features[..., -self.sentiment_dim:]
        sentiments_proj = self.sentiment_proj(sentiments)

        text_weights = F.softmax(self.text_attn(embeddings).squeeze(-1), dim=1).unsqueeze(-1)
        sent_weights = F.softmax(self.sent_attn(sentiments_proj).squeeze(-1), dim=1).unsqueeze(-1)

        text_out = (embeddings * text_weights).sum(dim=1)
        sent_out = (sentiments_proj * sent_weights).sum(dim=1)

        x = torch.cat([text_out, sent_out, lagged_returns], dim=1)
        return torch.sigmoid(self.classifier(x).squeeze(-1))


class NewsDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    headlines, lags, labels = zip(*batch)
    padded_headlines = pad_sequence(headlines, batch_first=True)
    return padded_headlines, torch.stack(lags), torch.tensor(labels).float()


def prepare_attention_data(
    news_db, news_table, snp_db, snp_table,
    sentiment_cols, embed_model_name="all-MiniLM-L6-v2", n_days_lag=3
):
    model = SentenceTransformer(embed_model_name)

    conn_news = sqlite3.connect(news_db)
    query = f"SELECT title_clean, published_at, {', '.join(sentiment_cols)} FROM {news_table}"
    news_df = pd.read_sql_query(query, conn_news)
    conn_news.close()

    conn_snp = sqlite3.connect(snp_db)
    snp_df = pd.read_sql_query(f"SELECT * FROM {snp_table}", conn_snp, parse_dates=["trade_date"])
    conn_snp.close()

    news_df["published_at"] = pd.to_datetime(news_df["published_at"])
    news_df["trade_date"] = news_df["published_at"].dt.date
    snp_df["trade_date"] = snp_df["trade_date"].dt.date

    for col in sentiment_cols:
        news_df[col] = pd.to_numeric(news_df[col], errors="coerce")

    merged = news_df.merge(snp_df, on="trade_date", how="inner")
    merged = merged.dropna(subset=["title_clean", "increase"] + sentiment_cols)

    embeddings = model.encode(merged["title_clean"].tolist(), convert_to_numpy=True)
    merged["embedding"] = list(embeddings)
    merged["combined"] = merged.apply(lambda row: np.concatenate([row["embedding"]] + [np.array([row[col]]) for col in sentiment_cols]), axis=1)

    grouped = merged.groupby("trade_date")["combined"].agg(list).reset_index()
    daily = pd.merge(grouped, snp_df, on="trade_date", how="inner")
    daily = daily.dropna(subset=["combined", "increase", "open_price", "close_price"])

    for lag in range(1, n_days_lag + 1):
        daily[f"prev_return_t-{lag}"] = (
            daily["close_price"].shift(lag) - daily["open_price"].shift(lag)
        ) / daily["open_price"].shift(lag)

    for col in lag_cols:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")

    daily = daily.dropna(subset=lag_cols)

    return daily, lag_cols, len(sentiment_cols)


def train_attention_model(model, train_loader, device="cuda", epochs=2, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        for headlines, lags, labels in train_loader:
            headlines, lags, labels = headlines.to(device), lags.to(device), labels.to(device)
            preds = model(headlines, lags)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def rolling_evaluation(daily, lag_cols, sentiment_dim, embed_dim=384, n_lags=3,
                        device="cuda", train_start=None, test_start=None, test_end=None):
    results = []
    model = HeadlineDualAttentionModel(embed_dim=embed_dim, sentiment_dim=sentiment_dim, n_lags=n_lags).to(device)

    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    train_start = pd.to_datetime(train_start)
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)

    test_range = daily[(daily["trade_date"] >= test_start) & (daily["trade_date"] <= test_end)]
    test_indices = test_range.index.tolist()

    for i in tqdm(test_indices):
        train_rows = daily[(daily["trade_date"] >= train_start) & (daily["trade_date"] < daily.iloc[i]["trade_date"])]
        test_row = daily.iloc[i]

        if len(train_rows) < 10:
            continue

        train_samples = []
        for _, row in train_rows.iterrows():
            emb = torch.tensor(np.vstack(row["combined"]), dtype=torch.float)
            lags = torch.tensor(row[lag_cols].astype(np.float32).values, dtype=torch.float)
            label = int(row["increase"])
            train_samples.append((emb, lags, label))

        test_emb = torch.tensor(np.vstack(test_row["combined"]), dtype=torch.float).unsqueeze(0)
        test_lags = torch.tensor(test_row[lag_cols].astype(np.float32).values, dtype=torch.float).unsqueeze(0)
        true_label = int(test_row["increase"])

        train_loader = DataLoader(NewsDataset(train_samples), batch_size=16, shuffle=True, collate_fn=collate_fn)
        train_attention_model(model, train_loader, device=device, epochs=2)

        model.eval()
        with torch.no_grad():
            pred = model(test_emb.to(device), test_lags.to(device)).item()

        results.append((test_row["trade_date"], true_label, pred))

    df_result = pd.DataFrame(results, columns=["trade_date", "true", "pred"])
    acc = accuracy_score(df_result["true"], np.round(df_result["pred"]))
    auc = roc_auc_score(df_result["true"], df_result["pred"])
    print(f"\nRolling Evaluation — Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    return df_result


if __name__ == "__main__":
    sentiment_cols = ["llm_sentiment_score_gpt4o_zero_shot"]
    daily, lag_cols, sentiment_dim = prepare_attention_data(
        news_db="data/news_5_6.db",
        news_table="master0",
        snp_db="data/snp500_5_6.db",
        snp_table="snp500",
        sentiment_cols=sentiment_cols,
        n_days_lag=3
    )
    result = rolling_evaluation(
        daily, lag_cols,
        sentiment_dim=sentiment_dim,
        embed_dim=384,
        n_lags=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_start="2018-01-01",
        test_start="2023-01-01",
        test_end="2024-12-31"
    )