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


# === 1. Attention-Based Model ===
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


# === 2. Dataset + Collate ===
class NewsDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples  # list of (headline_embeds, lag_features, label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    headlines, lags, labels = zip(*batch)
    padded_headlines = pad_sequence(headlines, batch_first=True)
    return padded_headlines, torch.stack(lags), torch.tensor(labels).float()


# === 3. Data Preparation ===
def prepare_attention_data(
    news_db, news_table, snp_db, snp_table, sentiment_col,
    embed_model_name="all-MiniLM-L6-v2", n_days_lag=3,
    train_end="2022-12-31", test_start="2023-01-01"
):
    model = SentenceTransformer(embed_model_name)

    # Load news and price data
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

    # Compute embeddings
    merged["embedding"] = list(model.encode(merged["title_clean"].tolist(), convert_to_numpy=True))

    # Group by date
    grouped = merged.groupby("trade_date")["embedding"].agg(list).reset_index()

    # Merge back to get increase + prices
    daily = pd.merge(grouped, snp_df, on="trade_date", how="inner")
    daily = daily.dropna(subset=["embedding", "increase", "open_price", "close_price"])

    # Add lagged return features
    for lag in range(1, n_days_lag + 1):
        daily[f"prev_return_t-{lag}"] = (
            daily["close_price"].shift(lag) - daily["open_price"].shift(lag)
        ) / daily["open_price"].shift(lag)

    lag_cols = [f"prev_return_t-{i}" for i in range(1, n_days_lag + 1)]
    daily = daily.dropna(subset=lag_cols)

    # Create samples
    samples = []
    for _, row in daily.iterrows():
        headline_tensor = torch.tensor(np.vstack(row["embedding"]), dtype=torch.float)
        lag_tensor = torch.tensor(row[lag_cols].values, dtype=torch.float)
        label = int(row["increase"])
        samples.append((headline_tensor, lag_tensor, label))

    # Split
    train_end = pd.to_datetime(train_end).date()
    test_start = pd.to_datetime(test_start).date()

    train_samples = [s for i, s in enumerate(samples) if daily.iloc[i]["trade_date"] <= train_end]
    test_samples = [s for i, s in enumerate(samples) if daily.iloc[i]["trade_date"] >= test_start]

    return NewsDataset(train_samples), NewsDataset(test_samples)


# === 4. Train Loop ===
def train_attention_model(model, train_loader, val_loader, device="cuda", epochs=10, lr=1e-3):
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

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for headlines, lags, labels in val_loader:
                headlines, lags = headlines.to(device), lags.to(device)
                preds = model(headlines, lags).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, np.round(all_preds))
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: Accuracy = {acc:.4f}, AUC = {auc:.4f}")


# === 5. Run All ===
if __name__ == "__main__":
    train_ds, test_ds = prepare_attention_data(
        news_db="data/news_5_6.db",
        news_table="master0",
        snp_db="data/snp500_5_6.db",
        snp_table="snp500",
        sentiment_col="llm_sentiment_score_gpt4o_zero_shot",
        n_days_lag=3
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = HeadlineAttentionModel(embed_dim=384, n_lags=3)
    train_attention_model(model, train_loader, test_loader, device="cuda" if torch.cuda.is_available() else "cpu")