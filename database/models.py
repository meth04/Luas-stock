import sys
import os
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Date, Text, BigInteger
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

# Tạo thư mục data nếu chưa có
if not os.path.exists("data"):
    os.makedirs("data")

DB_URL = "sqlite:///data/vnstock.db"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 1. BẢNG DANH MỤC MÃ (Symbols) ---
class Symbol(Base):
    __tablename__ = 'symbols'
    ticker = Column(String(10), primary_key=True)
    company_name = Column(String(255))
    exchange = Column(String(10)) # HOSE, HNX
    industry = Column(String(100))

# --- 2. BẢNG DỮ LIỆU LỊCH SỬ NGÀY (OHLCV) ---
class MarketDataDaily(Base):
    __tablename__ = 'market_data_daily'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), index=True)
    date = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)
    buy_foreign = Column(BigInteger, default=0)  
    sell_foreign = Column(BigInteger, default=0) 

# --- 3. BẢNG DỮ LIỆU PHÚT (Realtime Snapshot) ---
class MarketDataIntraday(Base):
    __tablename__ = 'market_data_intraday'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    price = Column(Float)
    volume = Column(BigInteger)
    change_percent = Column(Float) # % Tăng giảm so với tham chiếu

# --- 4. BẢNG LỊCH SỬ KHUYẾN NGHỊ (Agent Logs) ---
class AgentLog(Base):
    __tablename__ = 'agent_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now)
    ticker = Column(String(10), index=True)
    action = Column(String(50)) # MUA / BÁN / QUAN SÁT
    confidence = Column(String(50)) # Tỷ trọng khuyến nghị
    reason = Column(Text) # Lý do cốt lõi
    full_report_path = Column(String(255)) # Đường dẫn file báo cáo chi tiết (nếu cần)

def init_db():
    """Hàm khởi tạo bảng"""
    Base.metadata.create_all(bind=engine)
    print("✅ Đã khởi tạo Database thành công tại data/vnstock.db")