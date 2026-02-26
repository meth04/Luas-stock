# engine/portfolio_manager.py
import sqlite3
import os
from datetime import datetime, timedelta

class PortfolioManager:
    def __init__(self, db_path="database/portfolio.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        # Bảng số dư
        cursor.execute('''CREATE TABLE IF NOT EXISTS balance (
                            fund_id TEXT PRIMARY KEY, 
                            cash REAL)''')
        # Bảng danh mục (Inventory) xử lý T+2
        cursor.execute('''CREATE TABLE IF NOT EXISTS inventory (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            ticker TEXT,
                            quantity INTEGER,
                            buy_price REAL,
                            buy_date TEXT,
                            status TEXT)''') # status: 'PENDING' (T+0, T+1) hoặc 'AVAILABLE' (T+2)
        
        # Khởi tạo vốn 500 triệu nếu chưa có
        cursor.execute("INSERT OR IGNORE INTO balance VALUES ('MAIN_FUND', 500000000)")
        self.conn.commit()

    def update_settlement(self, current_date_str):
        """Cập nhật trạng thái T+2 dựa trên ngày hiện tại"""
        cursor = self.conn.cursor()
        curr_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        
        # Ở VN, hàng về sau 2 ngày làm việc. Giả định đơn giản: >= 2 ngày
        cursor.execute("SELECT id, buy_date FROM inventory WHERE status = 'PENDING'")
        for row in cursor.fetchall():
            buy_date = datetime.strptime(row[1], "%Y-%m-%d")
            if (curr_date - buy_date).days >= 2:
                cursor.execute("UPDATE inventory SET status = 'AVAILABLE' WHERE id = ?", (row[0],))
        self.conn.commit()

    def get_fund_status(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT cash FROM balance WHERE fund_id = 'MAIN_FUND'")
        cash = cursor.fetchone()[0]
        cursor.execute("SELECT ticker, SUM(quantity), AVG(buy_price), status FROM inventory GROUP BY ticker, status")
        stocks = cursor.fetchall()
        return {"cash": cash, "inventory": stocks}

    def execute_trade(self, action, ticker, price, quantity, date_str):
        cursor = self.conn.cursor()
        if action == "BUY":
            cost = price * quantity
            cursor.execute("UPDATE balance SET cash = cash - ? WHERE fund_id = 'MAIN_FUND'", (cost,))
            cursor.execute("INSERT INTO inventory (ticker, quantity, buy_price, buy_date, status) VALUES (?,?,?,?,'PENDING')",
                           (ticker, quantity, price, date_str))
        elif action == "SELL":
            # Logic bán hàng AVAILABLE
            revenue = price * quantity
            cursor.execute("UPDATE balance SET cash = cash + ? WHERE fund_id = 'MAIN_FUND'", (revenue,))
            # Xóa bớt số lượng trong kho (Ưu tiên lô cũ nhất)
            cursor.execute("DELETE FROM inventory WHERE ticker = ? AND status = 'AVAILABLE' LIMIT 1", (ticker,))
        self.conn.commit()