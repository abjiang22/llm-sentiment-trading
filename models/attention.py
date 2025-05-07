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


class HeadlineAttentionModel(nn.Module):
    def __init__(self, embed_dim: int, n_lags: int):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + n_lags, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, headline_embeds, lagged_returns):
        attn_scores = self.attn_net(headline_embeds).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        weighted_embed = (headline_embeds * attn_weights).sum(dim=1)
        x = torch.cat([weighted_embed, lagged_returns], dim=1)
        logits = self.classifier(x).squeeze(-1)
        return torch.sigmoid(logits)


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
    news_db, news_table, snp_db, snp_table, embed_model_name="all-MiniLM-L6-v2", n_days_lag=3
):
    model = SentenceTransformer(embed_model_name)

    conn_news = sqlite3.connect(news_db)
    news_df = pd.read_sql_query(f"SELECT title_clean, published_at FROM {news_table}", conn_news)
    conn_news.close()

    conn_snp = sqlite3.connect(snp_db)
    snp_df = pd.read_sql_query(f"SELECT * FROM {snp_table}", conn_snp, parse_dates=["trade_date"])
    conn_snp.close()

    news_df["published_at"] = pd.to_datetime(news_df["published_at"])
    news_df["trade_date"] = news_df["published_at"].dt.date
    snp_df["trade_date"] = snp_df["trade_date"].dt.date

    merged = news_df.merge(snp_df, on="trade_date", how="inner")
    merged = merged.dropna(subset=["title_clean", "increase"])

    merged["embedding"] = list(model.encode(merged["title_clean"].tolist(), convert_to_numpy=True))
    grouped = merged.groupby("trade_date")["embedding"].agg(list).reset_index()
    daily = pd.merge(grouped, snp_df, on="trade_date", how="inner")
    daily = daily.dropna(subset=["embedding", "increase", "open_price", "close_price"])

    for lag in range(1, n_days_lag + 1):
        daily[f"prev_return_t-{lag}"] = (
            daily["close_price"].shift(lag) - daily["open_price"].shift(lag)
        ) / daily["open_price"].shift(lag)

    lag_cols = [f"prev_return_t-{i}" for i in range(1, n_days_lag + 1)]
    daily = daily.dropna(subset=lag_cols)

    return daily, lag_cols


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


def rolling_evaluation(daily, lag_cols, window_size=250, embed_dim=384, n_lags=3, device="cuda"):
    results = []
    model = HeadlineAttentionModel(embed_dim, n_lags).to(device)

    for i in tqdm(range(window_size, len(daily))):
        train_rows = daily.iloc[i - window_size:i]
        test_row = daily.iloc[i]

        train_samples = []
        for _, row in train_rows.iterrows():
            emb = torch.tensor(np.vstack(row["embedding"]), dtype=torch.float)
            lags = torch.tensor(row[lag_cols].values, dtype=torch.float)
            label = int(row["increase"])
            train_samples.append((emb, lags, label))

        test_emb = torch.tensor(np.vstack(test_row["embedding"]), dtype=torch.float).unsqueeze(0)
        test_lags = torch.tensor(test_row[lag_cols].values, dtype=torch.float).unsqueeze(0)
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
    daily, lag_cols = prepare_attention_data(
        news_db="data/news_5_6.db",
        news_table="master0",
        snp_db="data/snp500_5_6.db",
        snp_table="snp500",
        n_days_lag=3
    )
    result = rolling_evaluation(daily, lag_cols, window_size=250)