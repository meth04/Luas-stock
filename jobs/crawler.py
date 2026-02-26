import time
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Fix path để import module từ thư mục gốc
sys.path.append(os.getcwd())

# Import nội bộ
from database.models import init_db
from database.repo import DataRepository

try:
    from vnstock import Vnstock
except ImportError:
    print("❌ Lỗi: Chưa cài đặt thư viện 'vnstock'.")
    print("👉 Vui lòng chạy: pip install -U vnstock")
    sys.exit(1)

class MarketCrawler:
    """
    Class chịu trách nhiệm tải dữ liệu thị trường (OHLCV + Foreign Flow)
    và lưu vào Database thông qua DataRepository.
    """
    
    def __init__(self):
        # Danh sách VN30 (Có thể mở rộng thêm nếu muốn)
        self.watchlist = [
            "ACB", "BCM", "BID", "CTG", "DGC", "FPT", "GAS", "GVR", "HDB", "HPG",
            "LPB", "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB",
            "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
        ]
        self.repo = DataRepository()

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa tên cột về định dạng thống nhất cho Database"""
        # 1. Đưa hết về chữ thường
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # 2. Map tên cột từ các nguồn khác nhau về chuẩn chung
        rename_map = {
            'time': 'date',
            'tradingdate': 'date',
            'datetime': 'date',
            'date_time': 'date',
            'vol': 'volume',
            'volume': 'volume',
            'nm_volume': 'volume', # Khớp lệnh
            'high': 'high',
            'low': 'low', 
            'open': 'open', 
            'close': 'close',
            'buy_foreign_quantity': 'buy_foreign',
            'sell_foreign_quantity': 'sell_foreign',
            'foreign_buy': 'buy_foreign',
            'foreign_sell': 'sell_foreign'
        }
        
        df = df.rename(columns=rename_map)
        
        # 3. Đảm bảo các cột bắt buộc phải có (nếu thiếu thì fill 0)
        required_cols = ['buy_foreign', 'sell_foreign', 'volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
                
        return df

    def _fetch_from_api(self, ticker: str) -> pd.DataFrame:
        """
        Gọi API Vnstock lấy dữ liệu 10 năm.
        Ưu tiên nguồn VCI vì có dữ liệu Khối ngoại đầy đủ.
        """
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            # Lấy 10 năm (3652 ngày) để phục vụ training model dài hạn
            start_date = (datetime.now() - timedelta(days=3652)).strftime('%Y-%m-%d')
            
            # --- NGUỒN 1: VCI (Ưu tiên) ---
            try:
                stock = Vnstock().stock(symbol=ticker, source='VCI')
                df = stock.quote.history(start=start_date, end=end_date, interval='1D')
            except Exception:
                df = pd.DataFrame()

            # --- NGUỒN 2: TCBS (Fallback nếu VCI lỗi) ---
            if df is None or df.empty:
                # print(f"⚠️ {ticker}: VCI thiếu dữ liệu, thử TCBS...")
                stock = Vnstock().stock(symbol=ticker, source='TCBS')
                df = stock.quote.history(start=start_date, end=end_date, interval='1D')
            
            # --- XỬ LÝ DỮ LIỆU ---
            if df is not None and not df.empty:
                # 1. Chuẩn hóa tên cột
                df = self._normalize_columns(df)
                
                # 2. Chuyển đổi kiểu dữ liệu
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Ép kiểu số cho các cột giá trị
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'buy_foreign', 'sell_foreign']
                for c in numeric_cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
                
                # Loại bỏ dòng nào không có ngày tháng hoặc giá = 0
                df = df.dropna(subset=['date'])
                df = df[df['close'] > 0]
                
                return df
            
            return pd.DataFrame()

        except Exception as e:
            print(f"⚠️ Lỗi API nghiêm trọng khi tải {ticker}: {str(e)}")
            return pd.DataFrame()

    def run_daily_update(self):
        """Hàm chính để chạy cập nhật hàng ngày"""
        print(f"\n🚀 BẮT ĐẦU CRAWL DATA & CẬP NHẬT DB ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"📋 Danh sách theo dõi: {len(self.watchlist)} mã (VN30)")
        print("⏳ Đang tải dữ liệu 10 năm (có thể mất vài phút)...")
        
        total_new_records = 0
        
        for i, ticker in enumerate(self.watchlist):
            print(f"   [{i+1}/{len(self.watchlist)}] Đang xử lý {ticker}...", end=" ")
            
            df = self._fetch_from_api(ticker)
            
            if not df.empty:
                # Lưu vào Database (repo sẽ tự check ngày trùng)
                count = self.repo.save_daily_data(ticker, df)
                if count > 0:
                    print(f"✅ Đã thêm {count} ngày mới.")
                else:
                    print(f"✅ Dữ liệu đã mới nhất.")
                
                total_new_records += count
            else:
                print("❌ Không tải được dữ liệu.")
            
            # Sleep 1.5s để tránh bị chặn IP (Rate Limit)
            time.sleep(1.5)

        print("-" * 60)
        print(f"✅ HOÀN TẤT CẬP NHẬT. Tổng cộng thêm: {total_new_records} bản ghi.")
        self.repo.close()

if __name__ == "__main__":
    # 1. Khởi tạo Database (Tạo bảng nếu chưa có)
    init_db()
    
    # 2. Chạy Crawler
    crawler = MarketCrawler()
    crawler.run_daily_update()