import sqlite3

# connect to your database
conn = sqlite3.connect('eco_scan.db')
cursor = conn.cursor()

print("\n=== Tables in eco_scan.db ===")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for t in tables:
    print("-", t[0])

print("\n=== Columns in each table ===")
for t in tables:
    print(f"\nTable: {t[0]}")
    cursor.execute(f"PRAGMA table_info({t[0]});")
    for col in cursor.fetchall():
        print(f"   {col[1]} ({col[2]})")

conn.close()
