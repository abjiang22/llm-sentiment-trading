import sqlite3
import pandas as pd

def drop_table(db_path, table_name):
    """
    Drops a table from a SQLite database if it exists.
    
    Parameters:
        db_path (str): Path to the SQLite database.
        table_name (str): Name of the table to drop.
    """

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("BEGIN TRANSACTION;")

        # Drop the table if it exists
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

        conn.commit()
        print(f"✅ Table '{table_name}' successfully dropped.")

    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to drop table '{table_name}': {e}")

    finally:
        conn.close()

def rename_table(db_path, old_table_name, new_table_name):
    """
    Renames a table in a SQLite database safely.

    Parameters:
        db_path (str): Path to the SQLite database file.
        old_table_name (str): Current table name.
        new_table_name (str): New table name you want to assign.
    """

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("BEGIN TRANSACTION;")

        # Perform the rename
        cursor.execute(f"ALTER TABLE {old_table_name} RENAME TO {new_table_name};")

        conn.commit()
        print(f"✅ Table '{old_table_name}' successfully renamed to '{new_table_name}'.")

    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to rename table '{old_table_name}': {e}")

    finally:
        conn.close()

def copy_table(db_path, source_table_name, new_table_name):
    """
    Creates a copy of a table including both schema and data.
    
    Parameters:
        db_path (str): Path to the SQLite database.
        source_table_name (str): Name of the table to copy.
        new_table_name (str): New table name for the copy.
    """

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("BEGIN TRANSACTION;")

        # Drop the new table if it already exists
        cursor.execute(f"DROP TABLE IF EXISTS {new_table_name}")

        # Create new table with same schema
        cursor.execute(f"CREATE TABLE {new_table_name} AS SELECT * FROM {source_table_name}")

        conn.commit()
        print(f"✅ Table '{source_table_name}' successfully copied to '{new_table_name}'.")

    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to copy table '{source_table_name}': {e}")

    finally:
        conn.close()

def drop_column_rebuild_table(db_path, table_name, column_to_drop):
    """
    Drops a column from a SQLite table by rebuilding it safely with transaction handling.
    """

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Load the table into a DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    # Check if the column exists
    if column_to_drop not in df.columns:
        print(f"⚠️ Column '{column_to_drop}' does not exist in table '{table_name}'.")
        conn.close()
        return

    # Remove the column in-memory
    df_new = df.drop(columns=[column_to_drop])

    # Prepare columns
    cols = list(df_new.columns)
    if 'id' in cols:
        cols.remove('id')  # Don't insert 'id', let it autoincrement

    try:
        cursor.execute("BEGIN TRANSACTION;")

        # Rename old table
        cursor.execute(f"ALTER TABLE {table_name} RENAME TO {table_name}_old;")

        # Create new table
        create_columns = []
        for col in df_new.columns:
            if col == 'id':
                create_columns.append("id INTEGER PRIMARY KEY AUTOINCREMENT")
            else:
                create_columns.append(f"{col} TEXT")

        cursor.execute(f'''
            CREATE TABLE {table_name} (
                {', '.join(create_columns)}
            );
        ''')

        # Insert data (without id)
        placeholders = ', '.join(['?' for _ in cols])
        insert_sql = f'''
            INSERT INTO {table_name} ({', '.join(cols)})
            VALUES ({placeholders});
        '''
        cursor.executemany(insert_sql, df_new[cols].values.tolist())

        # Drop the old table
        cursor.execute(f"DROP TABLE {table_name}_old;")

        # Commit
        conn.commit()
        print(f"✅ Column '{column_to_drop}' dropped and table '{table_name}' rebuilt successfully.")

    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to drop column '{column_to_drop}': {e}")

    finally:
        conn.close()

def reindex_table_by_column(db_path, table_name, specified_column):
    """
    Reindex a table by a specified column in ascending order and reassign id sequentially.
    Fully in-SQL version, no pandas. Schema-agnostic except assumes 'id' column exists.
    """

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("BEGIN TRANSACTION;")

        # Fetch original schema (excluding 'id')
        cursor.execute(f"PRAGMA table_info({table_name})")
        schema_info = cursor.fetchall()

        non_id_columns = [col[1] for col in schema_info if col[1] != 'id']

        if specified_column not in non_id_columns:
            raise ValueError(f"Column '{specified_column}' does not exist in table '{table_name}'.")

        # Create temporary table for sorted data
        temp_table = f"{table_name}_temp"

        # Drop temp table if exists
        cursor.execute(f"DROP TABLE IF EXISTS {temp_table}")

        # Dynamically create new table
        columns_def = ", ".join([f"{col} TEXT" for col in non_id_columns])
        cursor.execute(f"""
            CREATE TABLE {temp_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {columns_def}
            );
        """)

        # Insert sorted data into the temp table
        columns_list = ", ".join(non_id_columns)
        cursor.execute(f"""
            INSERT INTO {temp_table} ({columns_list})
            SELECT {columns_list}
            FROM {table_name}
            ORDER BY {specified_column} ASC;
        """)

        # Drop the original table
        cursor.execute(f"DROP TABLE {table_name}")

        # Rename temp table back to original table name
        cursor.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")

        conn.commit()
        print(f"✅ Table '{table_name}' successfully reordered by '{specified_column}' and reindexed by id.")

    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to reorder table '{table_name}': {e}")

    finally:
        conn.close()

def filter_table_by_date(db_path, table_name, date_column, start_date=None, end_date=None, new_table_name=None):
    """
    Filters a table's rows by date properly, using SQLite's date() function (no substring hacks).
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if new_table_name is None:
        new_table_name = f"{table_name}_filtered"

    try:
        cursor.execute("BEGIN TRANSACTION;")

        # Build the WHERE clause dynamically
        conditions = []
        params = []

        if start_date:
            conditions.append(f"date({date_column}) >= date(?)")
            params.append(start_date)
        if end_date:
            conditions.append(f"date({date_column}) <= date(?)")
            params.append(end_date)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        # Drop the new table if it exists
        cursor.execute(f"DROP TABLE IF EXISTS {new_table_name}")

        # Create the filtered table
        cursor.execute(f"""
            CREATE TABLE {new_table_name} AS
            SELECT * FROM {table_name}
            {where_clause}
        """, params)

        conn.commit()
        print(f"✅ Filtered table '{new_table_name}' created successfully using date() comparison.")

    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to filter table '{table_name}': {e}")

    finally:
        conn.close()


#rename_table('data/news.db', 'master0_filtered', 'master0')
#reindex_table_by_column('data/news.db', 'master0_filtered', 'published_at')

""" 
filter_table_by_date(
    db_path="data/news.db",
    table_name="master0",
    date_column="published_at",
    start_date="2018-12-25",
    end_date="2025-04-14",
    new_table_name="master0_filtered"
) """

#copy_table('data/news.db', 'master0', 'master0_4_21')
drop_table('data/news_backup.db', 'master0')
#drop_column_rebuild_table('data/news.db', 'master0', 'finbert_sentiment_score')