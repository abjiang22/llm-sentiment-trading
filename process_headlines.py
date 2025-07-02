import sqlite3
import re
import pandas as pd
from datetime import time as dtime
import pytz
from table_ops import reindex_table_by_column
import string

def exclude_market_hours(
    news_db_path:   str = "data/news.db",
    news_table_name: str = "master0",
    snp500_db_path: str = "data/snp500.db",
    snp500_table_name: str = "snp500",
    strict_weekdays: bool = False          # treat any Mon‑Fri not in S&P file as a trading day
):
    """
    Drop rows whose local (America/New_York) time lies within 09:30 ≤ t < 16:00
    on a trading day.  Works whether the table’s primary key is the hidden
    ROWID *or* an explicit `id INTEGER PRIMARY KEY`.  The original timestamp
    text is preserved; `id` keeps its INTEGER type.

    After execution the function prints both the number of rows removed and a
    post‑filter sanity check.
    """

    eastern   = pytz.timezone("America/New_York")
    mkt_open  = dtime(9, 30)
    mkt_close = dtime(16, 0)          # exclusive upper bound (t < 16:00)

    with sqlite3.connect(news_db_path) as nconn, sqlite3.connect(snp500_db_path) as sconn:
        # ────────────────── 1. trading‑day set ────────────────────────────
        trading_df = pd.read_sql(f"SELECT trade_date FROM {snp500_table_name}", sconn)
        trading_days = set(pd.to_datetime(trading_df.trade_date).dt.date)

        if strict_weekdays and trading_days:
            full_range = pd.date_range(min(trading_days), max(trading_days), freq="D")
            weekdays   = {d.date() for d in full_range if d.weekday() < 5}
            trading_days |= weekdays

        # ────────────────── 2. determine primary‑key column ───────────────
        pragma = nconn.execute(f"PRAGMA table_info({news_table_name})").fetchall()
        id_pk  = any(col[1] == "id" and col[5] == 1 for col in pragma)  # col[5] == pk flag

        pk_sel = "id AS rid, *" if id_pk else "rowid AS rid, *"
        df     = pd.read_sql(f"SELECT {pk_sel} FROM {news_table_name}", nconn)

        # ────────────────── 3. timestamp parsing & mask ───────────────────
        orig_ts   = df.published_at.astype(str)                          # keep raw text
        parsed_utc = pd.to_datetime(orig_ts, utc=True, errors="coerce")
        local_ts   = parsed_utc.dt.tz_convert(eastern)

        in_session = (
            local_ts.dt.date.isin(trading_days) &
            (local_ts.dt.time >= mkt_open) &
            (local_ts.dt.time <  mkt_close)
        )

        removed = df.loc[in_session, "rid"]
        kept_df = df.loc[~in_session].copy()

        # restore untouched timestamp text and enforce INTEGER id
        kept_df["published_at"] = orig_ts.loc[kept_df.index]
        if "id" in kept_df.columns:
            kept_df["id"] = (
                pd.to_numeric(kept_df["id"], downcast="integer", errors="coerce")
                .astype("Int64")
            )

        # ────────────────── 4. write back ────────────────────────────────
        dtype_map = {"published_at": "TEXT"}
        if "id" in kept_df.columns:
            dtype_map["id"] = "INTEGER"

        kept_df.drop(columns=["rid"]).to_sql(
            news_table_name,
            nconn,
            if_exists="replace",
            index=False,
            dtype=dtype_map
        )

        # ────────────────── 5. sanity check ───────────────────────────────
        chk = pd.read_sql(f"SELECT published_at FROM {news_table_name}", nconn)
        chk_ts = (
            pd.to_datetime(chk.published_at, utc=True, errors="coerce")
            .dt.tz_convert(eastern)
        )
        still_bad = chk.loc[
            chk_ts.dt.date.isin(trading_days) &
            (chk_ts.dt.time >= mkt_open) &
            (chk_ts.dt.time <  mkt_close)
        ]

    print(f"🗑️  removed {len(removed):,} market‑hour row(s)")
    if still_bad.empty:
        print("✅ post‑check: no market‑hour rows remain.")
    else:
        example = still_bad.published_at.iloc[0]
        print(f"⚠️  {len(still_bad):,} market‑hour rows still present!  "
              f"First example → {example}")

def keep_only_pre_market_articles(
    news_db_path:   str = "data/news.db",
    news_table_name: str = "master0",
    snp500_db_path: str = "data/snp500.db",
    snp500_table_name: str = "snp500",
    strict_weekdays: bool = False
):
    import sqlite3
    import pandas as pd
    import pytz
    from datetime import timedelta, time as dtime

    eastern = pytz.timezone("America/New_York")
    mkt_open_time = dtime(9, 30)

    with sqlite3.connect(news_db_path) as nconn, sqlite3.connect(snp500_db_path) as sconn:
        # ────────────────── 1. Load trading days ──────────────────────────────
        trading_df = pd.read_sql(f"SELECT trade_date FROM {snp500_table_name}", sconn)
        trading_days = set(pd.to_datetime(trading_df.trade_date).dt.date)

        if strict_weekdays and trading_days:
            full_range = pd.date_range(min(trading_days), max(trading_days), freq="D")
            weekdays   = {d.date() for d in full_range if d.weekday() < 5}
            trading_days |= weekdays

        # ────────────────── 2. Determine primary key and read data ────────────
        pragma = nconn.execute(f"PRAGMA table_info({news_table_name})").fetchall()
        id_pk  = any(col[1] == "id" and col[5] == 1 for col in pragma)

        pk_sel = "id AS rid, *" if id_pk else "rowid AS rid, *"
        df     = pd.read_sql(f"SELECT {pk_sel} FROM {news_table_name}", nconn)

        # ────────────────── 3. Timestamp conversion ───────────────────────────
        orig_ts     = df.published_at.astype(str)
        published_utc = pd.to_datetime(orig_ts, utc=True, errors="coerce")
        published_et  = published_utc.dt.tz_convert(eastern)

        # ────────────────── 4. Build trading day window mask ──────────────────
        keep_mask = pd.Series(False, index=df.index)

        for date in trading_days:
            open_dt = eastern.localize(pd.Timestamp.combine(date, mkt_open_time))
            lower_bound = open_dt - timedelta(hours=12)
            upper_bound = open_dt
            in_window = (published_et >= lower_bound) & (published_et < upper_bound)
            keep_mask |= in_window

        kept_df = df[keep_mask].copy()
        kept_df["published_at"] = orig_ts.loc[kept_df.index]

        if "id" in kept_df.columns:
            kept_df["id"] = (
                pd.to_numeric(kept_df["id"], downcast="integer", errors="coerce")
                .astype("Int64")
            )

        # ────────────────── 5. Write filtered data back ───────────────────────
        dtype_map = {"published_at": "TEXT"}
        if "id" in kept_df.columns:
            dtype_map["id"] = "INTEGER"

        kept_df.drop(columns=["rid"]).to_sql(
            news_table_name,
            nconn,
            if_exists="replace",
            index=False,
            dtype=dtype_map
        )

        print(f"✅ Kept {len(kept_df):,} article(s) within 12h before market open.")

def standardize_published_at(
    db_path: str = "data/news.db",
    table:   str = "master0",
    batch:   int = 10000
):
    """
    Rewrite EVERY non‑NULL published_at value in `table` to:
        YYYY-MM-DD HH:MM:SS+00:00   (UTC)
    Uses SQLite ROWID (never NULL) so it works even if your visible `id`
    column is empty, floats, or absent.
    """
    ISO_FMT = "%Y-%m-%d %H:%M:%S+00:00"

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # total rows to scan
        total = cur.execute(f"SELECT COUNT(*) FROM {table}"
                            ).fetchone()[0]
        print(f"Scanning {total:,} rows …")

        # stream through the table in blocks
        offset = 0
        fixed  = 0
        while True:
            rows = pd.read_sql_query(
                f"""
                SELECT rowid AS rid, published_at
                FROM   {table}
                LIMIT  {batch} OFFSET {offset}
                """,
                conn
            )
            if rows.empty:
                break

            # parse -> UTC -> canonical text
            parsed   = pd.to_datetime(rows.published_at,
                                      utc=True, errors="coerce")
            std_text = parsed.dt.strftime(ISO_FMT)

            # build [(new_ts, rid), …] where value really changes
            upd = [
                (new, int(rid))
                for new, old, rid in zip(std_text,
                                         rows.published_at,
                                         rows.rid)
                if pd.notna(new) and new != old
            ]
            if upd:
                cur.executemany(
                    f"UPDATE {table} SET published_at = ? "
                    f"WHERE ROWID = ?",
                    upd
                )
                fixed += len(upd)
            offset += batch
        conn.commit()
    print(f"✅ Standardised {fixed:,} / {total:,} timestamps.")


def remove_duplicates(db_path: str, table_name: str, group_by_cols: list):
    """
    Safely remove duplicates from a SQLite table by deleting rows with duplicate values
    in group_by_cols, keeping the one with the smallest ROWID.
    This method preserves the original schema (columns, types, indexes).
    """
    if not group_by_cols:
        raise ValueError("group_by_cols must contain at least one column name.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    group_cols_str = ", ".join(group_by_cols)

    # Step 1: Get original row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    original_count = cursor.fetchone()[0]

    # Step 2: Delete duplicates while keeping the first (lowest ROWID)
    delete_sql = f"""
        DELETE FROM {table_name}
        WHERE ROWID NOT IN (
            SELECT MIN(ROWID) FROM {table_name}
            GROUP BY {group_cols_str}
        )
    """
    cursor.execute(delete_sql)

    # Step 3: Get remaining row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    dedup_count = cursor.fetchone()[0]
    deleted_count = original_count - dedup_count

    conn.commit()
    conn.close()

    print(f"✅ Deleted {deleted_count} duplicate row(s) from '{table_name}' (based on {group_by_cols}).")

import sqlite3
import re
import pandas as pd

def clean_titles(db_path="data/news.db", table_name="master0", batch_size=1000):
    """Efficiently clean titles and update title_clean column without overwriting the table."""

    def clean_text(text):
        text = re.sub(r"[^\w\s]", "", str(text))     # Remove punctuation
        text = re.sub(r"\s+", " ", text)             # Normalize spaces
        return text.strip().lower()

    # Connect to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Add column if it doesn't exist
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cursor.fetchall()]
    if "title_clean" not in cols:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN title_clean TEXT")

    # Optional: add index for faster update if not exists
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_id ON {table_name}(id)")

    # Load only rows where cleaning is needed
    df = pd.read_sql_query(
        f"SELECT id, title FROM {table_name} WHERE title IS NOT NULL AND (title_clean IS NULL OR title_clean = '')",
        conn
    )
    if df.empty:
        print("✅ No titles to clean. All rows are up to date.")
        conn.close()
        return

    # Clean text
    df['title_clean'] = df['title'].apply(clean_text)

    # Batch update
    updates = list(zip(df['title_clean'], df['id']))
    print(f"🔧 Updating {len(updates)} rows in batches of {batch_size}...")

    for i in range(0, len(updates), batch_size):
        batch = updates[i:i+batch_size]
        cursor.executemany(
            f"UPDATE {table_name} SET title_clean = ? WHERE id = ?",
            batch
        )
        conn.commit()

    conn.close()
    print(f"✅ title_clean column updated for {len(updates)} rows in table '{table_name}'.")

def remove_invalid_titles(db_path="data/news.db", table_name="master0"):
    """
    Delete rows where `title` contains any non-ASCII printable characters.
    Keeps rows with only standard English letters, numbers, punctuation, and space.
    """
    conn = sqlite3.connect(db_path)

    # Load title values
    df = pd.read_sql_query(f"""
        SELECT rowid, title
        FROM {table_name}
        WHERE title IS NOT NULL AND title != ''
    """, conn)

    allowed = set(string.printable)

    def is_valid(text):
        return all(c in allowed for c in text) and any(c.isalnum() for c in text)

    bad_ids = [row["rowid"] for _, row in df.iterrows()
               if not is_valid(row["title"])]

    if bad_ids:
        conn.execute("BEGIN")
        conn.executemany(
            f"DELETE FROM {table_name} WHERE rowid = ?",
            [(i,) for i in bad_ids]
        )
        conn.commit()

    print(f"✅ Deleted {len(bad_ids)} row(s) with invalid characters in `title`.")
    conn.close()

def process_headlines(news_db_path="data/news.db", news_table_name="master0", snp500_db_path="data/snp500.db", snp500_table_name="snp500", group_by_cols=["title"], index_col="published_at"):
    keep_only_pre_market_articles(news_db_path, news_table_name, snp500_db_path, snp500_table_name)
    standardize_published_at(news_db_path, news_table_name, 10000)
    remove_duplicates(news_db_path, news_table_name, group_by_cols)
    clean_titles(news_db_path, news_table_name)
    remove_invalid_titles(news_db_path, news_table_name)
    reindex_table_by_column(news_db_path, news_table_name, index_col)

process_headlines(news_db_path="data/news.db", news_table_name="master0", snp500_db_path="data/snp500.db", snp500_table_name="snp500", group_by_cols=["title"], index_col="published_at")

#exclude_market_hours("data/news.db", "data/snp500.db")
#clean_titles(db_path="data/news.db", table_name="master0")R