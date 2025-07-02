import sqlite3
import re

def drop_columns_preserve_schema(db_path, table_name, columns_to_keep):
    """
    Drops all columns from the table not in columns_to_keep, preserving types, constraints, and indexes.
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # === Step 1: Extract full CREATE TABLE SQL ===
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    result = cursor.fetchone()
    if not result:
        raise ValueError(f"Table '{table_name}' does not exist in the database.")
    create_sql = result[0]

    # === Step 2: Parse column definitions ===
    col_defs = re.findall(r'\s*(\w+[^,]*),?', create_sql[create_sql.find('(')+1:create_sql.rfind(')')])
    col_defs_cleaned = []
    for col_def in col_defs:
        col_name = re.match(r'^`?(\w+)`?', col_def.strip()).group(1)
        if col_name in columns_to_keep:
            col_defs_cleaned.append(col_def.strip())

    if not col_defs_cleaned:
        raise ValueError("No columns to keep. You must preserve at least one column.")

    new_table_sql = f"CREATE TABLE {table_name}_new (\n  " + ",\n  ".join(col_defs_cleaned) + "\n);"
    cursor.execute("BEGIN TRANSACTION;")
    cursor.execute(new_table_sql)

    # === Step 3: Copy data into the new table ===
    cols_str = ', '.join(columns_to_keep)
    cursor.execute(f"INSERT INTO {table_name}_new ({cols_str}) SELECT {cols_str} FROM {table_name}")

    # === Step 4: Drop old indexes ===
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}'")
    indexes = cursor.fetchall()

    # === Step 5: Drop old table and rename new one ===
    cursor.execute(f"DROP TABLE {table_name}")
    cursor.execute(f"ALTER TABLE {table_name}_new RENAME TO {table_name}")

    # === Step 6: Recreate relevant indexes ===
    for (index_name,) in indexes:
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{index_name}'")
        index_sql = cursor.fetchone()[0]
        if index_sql:
            # Replace old table name in index SQL
            new_index_sql = index_sql.replace(f"ON {table_name}_new", f"ON {table_name}")
            # Check if index only uses preserved columns
            if all(col in columns_to_keep for col in re.findall(r'\b\w+\b', new_index_sql.split("(", 1)[1])):
                cursor.execute(new_index_sql)

    conn.commit()
    conn.close()
    print(f"✅ Dropped columns and preserved schema for '{table_name}'.")

# Example usage:
drop_columns_preserve_schema(
    db_path='data/snp500.db',
    table_name='snp500',
    columns_to_keep=['trade_id', 'trade_date', 'open_price', 'close_price', 'high_price', 'low_price', 'volume']
)
