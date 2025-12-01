import sqlite3
import os

db_path = "data/ecommerce.db"

if not os.path.exists(db_path):
    print(f"Database file not found at {db_path}")
else:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables found:")
        for table in tables:
            print(table[0])
        conn.close()
    except Exception as e:
        print(f"Error accessing database: {e}")
