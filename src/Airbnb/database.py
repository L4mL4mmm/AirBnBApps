import sqlite3
import os
from datetime import datetime

class Database:
    def __init__(self, db_name="airbnb_history.db"):
        self.db_name = os.path.join("Artifacts", db_name)
        self.init_db()

    def init_db(self):
        os.makedirs(os.path.dirname(self.db_name), exist_ok=True)
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT,
                property_type TEXT,
                room_type TEXT,
                accommodates INTEGER,
                price REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def insert_prediction(self, city, property_type, room_type, accommodates, price):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (city, property_type, room_type, accommodates, price)
            VALUES (?, ?, ?, ?, ?)
        ''', (city, property_type, room_type, accommodates, price))
        conn.commit()
        conn.close()

    def get_history(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10')
        rows = cursor.fetchall()
        conn.close()
        return rows
