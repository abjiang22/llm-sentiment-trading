
"""
import sqlite3

# Connect to your database
conn = sqlite3.connect('data/news_5_1.db')
cursor = conn.cursor()

# Execute the query
cursor.execute('''
    SELECT source, COUNT(*) AS article_count
    FROM master0
    GROUP BY source
    ORDER BY article_count DESC
''')

# Fetch all results
results = cursor.fetchall()

# Display
for source, count in results:
    print(f"{source}: {count} articles")

# Close the connection
conn.close()
"""


import sqlite3

def filter_sources(db_path: str, table_name: str):
    """
    Keep only rows with source in the allowed list and delete all others.

    Args:
        db_path (str): Path to SQLite database.
        table_name (str): Name of the table to filter.
    """
    allowed_sources = ("bloomberg", "businessinsider", "financialpost", "wsj")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Delete rows that are NOT in the allowed list
    placeholders = ','.join('?' for _ in allowed_sources)
    query = f"""
        DELETE FROM {table_name}
        WHERE source NOT IN ({placeholders})
    """
    cursor.execute(query, allowed_sources)
    conn.commit()
    conn.close()

    print(f"✅ Table `{table_name}` filtered to only keep rows from: {', '.join(allowed_sources)}")

filter_sources("data/news_4_24_experiment.db", "master0")