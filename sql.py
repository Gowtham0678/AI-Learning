import sqlite3
connect=sqlite3.connect('company.db')
cursor=connect.cursor()

cursor.execute("SELECT * FROM Employees")
rows = cursor.fetchall()
for row in rows:
    print(row)
connect.commit()
connect.close()
print("Data retrieved successfully.")