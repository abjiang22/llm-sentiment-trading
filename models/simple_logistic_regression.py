from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

def predict_increase_from_sentiment(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    model_name: str,
    train_start: str = '2018-01-01',
    train_end: str = '2022-12-31',
    test_start: str = '2023-01-01',
    test_end: str = '2024-12-31',
):
    # === Load dataset ===
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT trade_date, increase, {sentiment_column}
        FROM {table_name}
        WHERE increase IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, parse_dates=["trade_date"])
    conn.close()

    df = df.rename(columns={sentiment_column: 'sentiment'})
    df = df.dropna(subset=['sentiment'])

    # === Train-test split ===
    train_df = df[(df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)]
    test_df  = df[(df['trade_date'] >= test_start) & (df['trade_date'] <= test_end)]

    X_train = train_df[['sentiment']]
    y_train = train_df['increase']
    X_test  = test_df[['sentiment']]
    y_test  = test_df['increase']

    # === Train logistic regression ===
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Coefficients:")
    print(f"  Intercept: {model.intercept_[0]:.4f}")
    print(f"  Sentiment coef: {model.coef_[0][0]:.4f}")

    output = test_df[['trade_date']].copy()
    output['actual_increase'] = y_test.values
    output['predicted_proba'] = y_proba
    output['predicted_label'] = y_pred
    save_path = f"predictions/{model_name}_logistic_prediction.csv"
    output.to_csv(save_path, index=False)
    print(f"Saved predictions to {save_path}")

def predict_increase_by_source(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    model_name: str,
    sources: list = ["financialpost", "wsj", "bloomberg", "businessinsider"],
    train_start: str = '2018-01-01',
    train_end: str = '2022-12-31',
    test_start: str = '2023-01-01',
    test_end: str = '2024-12-31',
):

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE increase IS NOT NULL", conn, parse_dates=['trade_date'])
    conn.close()

    sentiment_cols = [f"{sentiment_column}_{src}" for src in sources if f"{sentiment_column}_{src}" in df.columns]
    if not sentiment_cols:
        raise ValueError(f"No matching sentiment columns found for selected sources: {sources}")

    df = df.dropna(subset=sentiment_cols)

    scaler = StandardScaler()
    train_mask = (df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)
    scaler.fit(df.loc[train_mask, sentiment_cols])
    df_std = df.copy()
    df_std[sentiment_cols] = pd.DataFrame(
        scaler.transform(df[sentiment_cols]),
        columns=sentiment_cols,
        index=df.index).fillna(0)

    train = df_std[train_mask]
    test  = df_std[(df_std['trade_date'] >= test_start) & (df_std['trade_date'] <= test_end)]

    X_train = train[sentiment_cols]
    y_train = train['increase']
    X_test  = test[sentiment_cols]
    y_test  = test['increase']

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy (by source): {acc:.4f}")
    print(f"AUC (by source): {auc:.4f}")
    print("Coefficients:")
    print(pd.Series(model.coef_[0], index=sentiment_cols))
    print(f"Intercept: {model.intercept_[0]:.4f}")

    output = test[['trade_date'] + sentiment_cols].copy()
    output['actual_increase'] = y_test.values
    output['predicted_proba'] = y_proba
    output['predicted_label'] = y_pred
    save_path = f"predictions/{model_name}_logistic_by_source.csv"
    output.to_csv(save_path, index=False)
    print(f"Saved predictions to {save_path}")

def predict_increase_by_source_upgraded(
    db_path: str,
    table_name: str,
    sentiment_column: str,
    model_name: str,
    sources: list = ["financialpost", "wsj", "bloomberg", "businessinsider"], 
    train_start: str = '2018-01-02',
    train_end: str = '2023-12-31',
    test_start: str = '2024-01-01',
    test_end: str = '2024-12-31',
    z_clip_threshold: float = 3.0
):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE increase IS NOT NULL", conn, parse_dates=["trade_date"])
    conn.close()

    sentiment_cols = [f"{sentiment_column}_{src}" for src in sources if f"{sentiment_column}_{src}" in df.columns]
    if not sentiment_cols:
        raise ValueError(f"No matching sentiment columns found for selected sources: {sources}")

    df = df.dropna(subset=sentiment_cols)

    train_mask = (df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, sentiment_cols])

    df_std = df.copy()
    df_std[sentiment_cols] = pd.DataFrame(
        scaler.transform(df[sentiment_cols]),
        columns=sentiment_cols,
        index=df.index
    )

    df_std[sentiment_cols] = df_std[sentiment_cols].clip(-z_clip_threshold, z_clip_threshold)

    train = df_std[train_mask]
    test  = df_std[(df['trade_date'] >= test_start) & (df['trade_date'] <= test_end)]

    X_train = train[sentiment_cols]
    y_train = train['increase']
    X_test  = test[sentiment_cols]
    y_test  = test['increase']

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

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"✅ Accuracy (by source): {acc:.4f}")
    print(f"✅ AUC (by source): {auc:.4f}")
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("=== Coefficients ===")
    print(pd.Series(model.coef_[0], index=sentiment_cols))
    print(f"Intercept: {model.intercept_[0]:.4f}")

    output = test[['trade_date'] + sentiment_cols].copy()
    output['actual_increase'] = y_test.values
    output['predicted_proba'] = y_proba
    output['predicted_label'] = y_pred
    save_csv = f"predictions/{model_name}_logistic_by_source_upgraded.csv"
    output.to_csv(save_csv, index=False)
    print(f"📁 Saved predictions to {save_csv}")

    joblib.dump(model, f"models/{model_name}_logreg.pkl")
    joblib.dump(scaler, f"models/{model_name}_scaler.pkl")
    print(f"💾 Model and scaler saved to /models")


def run_gpt4o_zero_shot():
    predict_increase_from_sentiment(
        db_path='data/snp500.db',
        table_name='snp500',
        sentiment_column='avg_sentiment_gpt4o_zero_shot',
        model_name='gpt4o_zero_shot'
    )

    predict_increase_by_source(
        db_path='data/snp500.db',
        table_name='snp500',
        sentiment_column='avg_sentiment_gpt4o_zero_shot',
        model_name='gpt4o_zero_shot',
        sources=["bloomberg", "businessinsider"]
    )

    predict_increase_by_source_upgraded(
        db_path='data/snp500.db',
        table_name='snp500',
        sentiment_column='avg_sentiment_gpt4o_zero_shot',
        model_name='gpt4o_zero_shot',
        sources=["bloomberg", "businessinsider"]
    )

def run_finbert():
    predict_increase_from_sentiment(
        db_path='data/snp500.db',
        table_name='snp500',
        sentiment_column='avg_sentiment_finbert',
        model_name='finbert'
    )

    predict_increase_by_source(
        db_path='data/snp500.db',
        table_name='snp500',
        sentiment_column='avg_sentiment_finbert',
        model_name='finbert',
        sources=["bloomberg", "businessinsider"]
    )

    predict_increase_by_source_upgraded(
        db_path='data/snp500.db',
        table_name='snp500',
        sentiment_column='avg_sentiment_finbert',
        model_name='finbert',
        sources=["bloomberg", "businessinsider"]
    )


def run_lm():
    predict_increase_from_sentiment(
        db_path='data/snp500.db',
        table_name='snp500',
        sentiment_column='avg_sentiment_lm',
        model_name='lm'
    )

    predict_increase_by_source(
        db_path='data/snp500.db',
        table_name='snp500',
        sentiment_column='avg_sentiment_lm',
        model_name='lm',
        sources=["bloomberg", "businessinsider"]
    )

    predict_increase_by_source_upgraded(
        db_path='data/snp500.db',
        table_name='snp500',
        sentiment_column='avg_sentiment_lm',
        model_name='lm',
        sources=["bloomberg", "businessinsider"]
    )

run_gpt4o_zero_shot()