"""
agent/tools/db_query.py

Executes read-only SQL queries against a local SQLite database.
Uses SQLite by default (zero setup) — swap DB_URL in .env for PostgreSQL.

For demo purposes, creates a sample database with fake data if none exists.
"""

import sqlite3
import os
import json
from agent.tools.base import BaseTool


DEMO_DB_PATH = "demo.db"


def _ensure_demo_db():
    """
    Create a demo SQLite database with sample data if it doesn't exist.
    This means the tool works out of the box with no setup.
    """
    if os.path.exists(DEMO_DB_PATH):
        return

    conn = sqlite3.connect(DEMO_DB_PATH)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL,
            stock INTEGER
        );

        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            product_id INTEGER,
            quantity INTEGER,
            total REAL,
            status TEXT,
            created_at TEXT
        );

        INSERT INTO products VALUES
            (1, 'Laptop Pro', 'Electronics', 1299.99, 45),
            (2, 'Wireless Mouse', 'Electronics', 29.99, 200),
            (3, 'Standing Desk', 'Furniture', 499.99, 30),
            (4, 'Monitor 4K', 'Electronics', 399.99, 75),
            (5, 'Office Chair', 'Furniture', 299.99, 50);

        INSERT INTO orders VALUES
            (1, 1, 2, 2599.98, 'completed', '2025-01-10'),
            (2, 2, 5, 149.95, 'completed', '2025-01-12'),
            (3, 3, 1, 499.99, 'pending', '2025-01-15'),
            (4, 4, 3, 1199.97, 'completed', '2025-01-18'),
            (5, 1, 1, 1299.99, 'cancelled', '2025-01-20'),
            (6, 5, 2, 599.98, 'completed', '2025-01-22');
    """)

    conn.commit()
    conn.close()


class DBQueryTool(BaseTool):
    name = "db_query"
    description = (
        "Run a read-only SQL SELECT query against the local database. "
        "Tables available: products (id, name, category, price, stock), "
        "orders (id, product_id, quantity, total, status, created_at). "
        "Only SELECT statements are allowed."
    )

    def __init__(self, db_path: str = DEMO_DB_PATH):
        self.db_path = db_path
        _ensure_demo_db()

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A valid SQL SELECT statement"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    async def execute(self, query: str) -> str:
        # safety check — only allow SELECT
        clean = query.strip().upper()
        if not clean.startswith("SELECT"):
            return "Error: Only SELECT queries are allowed."

        # block dangerous keywords even inside SELECT
        blocked = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "EXEC"]
        if any(word in clean for word in blocked):
            return "Error: Query contains disallowed keywords."

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return "Query returned no results."

            # format as readable JSON
            result = [dict(row) for row in rows]
            return json.dumps(result, indent=2)

        except sqlite3.Error as e:
            return f"Database error: {str(e)}"


