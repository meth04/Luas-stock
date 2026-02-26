from sqlalchemy.orm import Session
from .models import MarketDataDaily, MarketDataIntraday, AgentLog, SessionLocal
from datetime import datetime
import pandas as pd
import numpy as np

class DataRepository:
    def __init__(self):
        self.db: Session = SessionLocal()

    def close(self):
        self.db.close()

    def save_daily_data(self, ticker: str, df: pd.DataFrame):
        """
        Lưu DataFrame OHLCV + Foreign Flow vào DB.
        """
        if df.empty: return 0
        
        last_record = self.db.query(MarketDataDaily).filter(
            MarketDataDaily.ticker == ticker
        ).order_by(MarketDataDaily.date.desc()).first()
        
        last_date = last_record.date if last_record else datetime(2000, 1, 1)
        
        count = 0
        new_records = []
        
        for _, row in df.iterrows():
            row_date = row['date']
            if row_date > last_date:
                vol = int(row.get('volume', 0)) if not pd.isna(row.get('volume')) else 0
                buy_f = int(row.get('buy_foreign', 0)) if not pd.isna(row.get('buy_foreign')) else 0
                sell_f = int(row.get('sell_foreign', 0)) if not pd.isna(row.get('sell_foreign')) else 0
                
                record = MarketDataDaily(
                    ticker=ticker,
                    date=row_date,
                    open=float(row['open']), 
                    high=float(row['high']), 
                    low=float(row['low']), 
                    close=float(row['close']),
                    volume=vol,
                    buy_foreign=buy_f,
                    sell_foreign=sell_f
                )
                new_records.append(record)
                count += 1
        
        if new_records:
            self.db.add_all(new_records)
            self.db.commit()
            
        return count

    def save_agent_log(self, ticker: str, action: str, confidence: str, reason: str):
        """Lưu kết quả quyết định của Risk Manager"""
        try:
            log = AgentLog(
                ticker=ticker,
                action=action,
                confidence=confidence,
                reason=reason,
                timestamp=datetime.now()
            )
            self.db.add(log)
            self.db.commit()
            # print(f"💾 Đã lưu log cho {ticker}.")
        except Exception as e:
            print(f"⚠️ Lỗi lưu log: {e}")
            self.db.rollback()

    def get_price_history(self, ticker: str, days: int = 3650) -> pd.DataFrame:
        """
        Lấy dữ liệu lịch sử chuẩn hóa cho Quant Tool.
        Bao gồm cả dữ liệu Khối ngoại (buy_foreign, sell_foreign).
        """
        try:
            # Query lấy dữ liệu sắp xếp theo ngày tăng dần
            results = self.db.query(MarketDataDaily).filter(
                MarketDataDaily.ticker == ticker
            ).order_by(MarketDataDaily.date.asc()).all()

            if not results:
                return pd.DataFrame()

            # Chuyển đổi sang list dict
            data = [{
                'date': r.date,
                'open': r.open, 
                'high': r.high, 
                'low': r.low, 
                'close': r.close,
                'volume': r.volume,
                'buy_foreign': r.buy_foreign,
                'sell_foreign': r.sell_foreign
            } for r in results]
            
            df = pd.DataFrame(data)
            
            # Xử lý kiểu dữ liệu
            df['date'] = pd.to_datetime(df['date'])
            cols = ['open', 'high', 'low', 'close', 'volume', 'buy_foreign', 'sell_foreign']
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
            # Chỉ lấy số ngày yêu cầu
            if days > 0:
                df = df.tail(days)
                
            return df.reset_index(drop=True)
            
        except Exception as e:
            print(f"⚠️ Lỗi đọc DB {ticker}: {e}")
            return pd.DataFrame()