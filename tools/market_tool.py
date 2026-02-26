import pandas as pd
import numpy as np
import sys
import os
import time
from database.repo import DataRepository
from jobs.crawler import MarketCrawler

class MarketToolkit:
    _price_cache = {}

    @staticmethod
    def get_price_data(symbol: str, days: int = 730) -> pd.DataFrame:
        """Lấy dữ liệu giá có Cache (Giữ nguyên logic cũ)"""
        symbol = symbol.upper().strip()
        
        # Check Cache RAM
        if symbol in MarketToolkit._price_cache:
            last_time, cached_df = MarketToolkit._price_cache[symbol]
            if (pd.Timestamp.now() - last_time).total_seconds() < 3600:
                # Lọc data cần thiết
                return cached_df.tail(days + 50) # Lấy dư để tính chỉ báo

        repo = DataRepository()
        try:
            # 1. Query DB
            df = repo.get_price_history(symbol, days=days + 100) # Lấy dư 100 ngày để tính MA200
            
            # 2. Lazy Loading
            if df.empty:
                crawler = MarketCrawler()
                df_new = crawler._fetch_from_api(symbol)
                if not df_new.empty:
                    repo.save_daily_data(symbol, df_new)
                    df = repo.get_price_history(symbol, days=days + 100)
                time.sleep(1)

            # Cache lại
            if not df.empty:
                MarketToolkit._price_cache[symbol] = (pd.Timestamp.now(), df)
                return df.tail(days)
            return df
            
        except Exception as e:
            print(f"❌ Lỗi MarketTool: {e}", file=sys.stderr)
            return pd.DataFrame()
        finally:
            repo.close()

    @staticmethod
    def get_technical_report(symbol: str) -> str:
        """
        Phân tích kỹ thuật CHUYÊN SÂU (Advanced Technical Analysis)
        """
        # Lấy đủ dài để tính MA200 và Ichimoku
        df = MarketToolkit.get_price_data(symbol, days=365)
        if df.empty: return "⚠️ Không có dữ liệu giá."

        try:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # --- 1. TREND INDICATORS ---
            sma50 = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()
            
            # Ichimoku Cloud (Cơ bản: Conversion & Base Line)
            nine_period_high = high.rolling(window=9).max()
            nine_period_low = low.rolling(window=9).min()
            tenkan_sen = (nine_period_high + nine_period_low) / 2
            
            twenty_six_period_high = high.rolling(window=26).max()
            twenty_six_period_low = low.rolling(window=26).min()
            kijun_sen = (twenty_six_period_high + twenty_six_period_low) / 2

            # --- 2. MOMENTUM INDICATORS ---
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss.replace(0, 1e-10))
            rsi = 100 - (100 / (1 + rs))
            
            # Stochastic RSI (Nhạy hơn RSI thường)
            min_rsi = rsi.rolling(14).min()
            max_rsi = rsi.rolling(14).max()
            stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)

            # MACD
            k = close.ewm(span=12, adjust=False).mean()
            d = close.ewm(span=26, adjust=False).mean()
            macd = k - d
            signal = macd.ewm(span=9, adjust=False).mean()

            # --- 3. VOLATILITY & LEVELS ---
            # Bollinger Bands
            sma20 = close.rolling(20).mean()
            std = close.rolling(20).std()
            upper = sma20 + 2*std
            lower = sma20 - 2*std
            
            # Support & Resistance (Đơn giản: Đáy/Đỉnh 20 phiên)
            support_20d = low.rolling(20).min().iloc[-1]
            resistance_20d = high.rolling(20).max().iloc[-1]

            # --- TỔNG HỢP DỮ LIỆU HIỆN TẠI ---
            curr_price = close.iloc[-1]
            prev_price = close.iloc[-2]
            
            # Đánh giá Trend
            trend_long = "UPTREND" if curr_price > sma200.iloc[-1] else "DOWNTREND"
            trend_short = "BULLISH" if curr_price > sma50.iloc[-1] else "BEARISH"
            
            # Ichimoku Signal
            ichimoku_sig = "Tích cực" if tenkan_sen.iloc[-1] > kijun_sen.iloc[-1] else "Tiêu cực"

            # Oscillator Signals
            rsi_val = rsi.iloc[-1]
            stoch_val = stoch_rsi.iloc[-1]
            macd_val = macd.iloc[-1]
            sig_val = signal.iloc[-1]
            
            rsi_status = "QUÁ MUA (>70)" if rsi_val > 70 else "QUÁ BÁN (<30)" if rsi_val < 30 else "Trung tính"
            macd_status = "MUA (Cắt lên)" if macd_val > sig_val else "BÁN (Cắt xuống)"
            
            # Volume Analysis
            vol_mean = df['volume'].rolling(20).mean().iloc[-1]
            curr_vol = df['volume'].iloc[-1]
            vol_status = "Đột biến" if curr_vol > 1.5 * vol_mean else "Thấp" if curr_vol < 0.7 * vol_mean else "Trung bình"

            return f"""
            ### 📊 PHÂN TÍCH KỸ THUẬT NÂNG CAO: {symbol}
            
            **1. CẤU TRÚC GIÁ & XU HƯỚNG:**
            - Giá hiện tại: {curr_price:,.0f} VND ({trend_short} ngắn hạn / {trend_long} dài hạn)
            - Hỗ trợ gần nhất (20d): {support_20d:,.0f}
            - Kháng cự gần nhất (20d): {resistance_20d:,.0f}
            - Ichimoku (Tenkan/Kijun): {ichimoku_sig}
            
            **2. ĐỘNG LƯỢNG (MOMENTUM):**
            - RSI (14): {rsi_val:.2f} [{rsi_status}]
            - Stoch RSI: {stoch_val:.2f} (0-1) - {'Vùng đáy' if stoch_val < 0.2 else 'Vùng đỉnh' if stoch_val > 0.8 else 'Trung gian'}
            - MACD: {macd_status} (Histogram: {macd_val - sig_val:.2f})
            
            **3. BIẾN ĐỘNG & THANH KHOẢN:**
            - Bollinger Bands: Giá đang ở {'TRÊN' if curr_price > upper.iloc[-1] else 'DƯỚI' if curr_price < lower.iloc[-1] else 'GIỮA'} dải băng.
            - Volume: {curr_vol:,.0f} ({vol_status} so với TB 20 phiên)
            """
        except Exception as e:
            return f"❌ Lỗi tính toán: {e}"