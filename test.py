import sqlite3

# Connect to your database
conn = sqlite3.connect('data/news.db')
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