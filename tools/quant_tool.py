import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
import warnings

# Tắt cảnh báo chia cho 0
warnings.filterwarnings('ignore')

# Import DataRepository từ project của bạn
# Giả sử cấu trúc thư mục là:
# project/
#   database/repo.py
#   tools/quant_tool.py
try:
    from database.repo import DataRepository
except ImportError:
    # Fallback cho trường hợp chạy trực tiếp module
    import sys
    sys.path.append(os.getcwd())
    from database.repo import DataRepository

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

class QuantConfig:
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "vn30_ranker_dart.json")
    FEATURE_PATH = os.path.join(MODEL_DIR, "rank_features.pkl")
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    TICKERS = [
        "ACB", "BCM", "BID", "CTG", "DGC", "FPT", "GAS", "GVR", "HDB", "HPG",
        "LPB", "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB",
        "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
    ]

# =============================================================================
# 2. CORE LOGIC
# =============================================================================

class TechnicalAnalysis:
    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss.replace(0, 1e-9))
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series, f=12, s=26, sig=9):
        k = series.ewm(span=f, adjust=False).mean()
        d = series.ewm(span=s, adjust=False).mean()
        macd = k - d
        signal = macd.ewm(span=sig, adjust=False).mean()
        return macd, signal

    @staticmethod
    def bollinger(series, n=20, k=2):
        sma = series.rolling(n).mean()
        std = series.rolling(n).std()
        return sma + k*std, sma, sma - k*std

    @staticmethod
    def mfi(high, low, close, vol, n=14):
        tp = (high + low + close) / 3
        mf = tp * vol
        pos = (mf.where(tp > tp.shift(), 0)).rolling(n).sum()
        neg = (mf.where(tp < tp.shift(), 0)).rolling(n).sum()
        return 100 - (100 / (1 + pos/(neg.replace(0, 1e-9))))

class FeatureEngineer:
    @staticmethod
    def create_base_features(df):
        df = df.copy().sort_values('date')
        
        # --- CLEAN DATA ---
        df = df[df['close'] > 0]
        # Tránh chia cho 0
        df['volume'] = df['volume'].replace(0, 1) 
        
        # 1. Base Price Indicators
        # Log Return
        df['Log_Ret'] = np.log(df['close'] / df['close'].shift(1).replace(0, np.nan))
        df['Log_Ret'] = df['Log_Ret'].fillna(0)
        
        # Volatility
        df['Vol_10'] = df['Log_Ret'].rolling(10).std()
        
        # RSI & MACD
        df['RSI'] = TechnicalAnalysis.rsi(df['close'], 14)
        macd, sig = TechnicalAnalysis.macd(df['close'])
        df['MACD_Div'] = (macd - sig)
        
        # Bollinger %B
        u, m, l = TechnicalAnalysis.bollinger(df['close'])
        df['BB_Pb'] = (df['close'] - l) / (u - l + 1e-9)
        df['BB_Width'] = (u - l) / (m + 1e-9)
        
        # Volume features
        df['Vol_Ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1)
        df['MFI'] = TechnicalAnalysis.mfi(df['high'], df['low'], df['close'], df['volume'])
        
        # --- NEW: FOREIGN FLOW FEATURES (DÒNG TIỀN KHỐI NGOẠI) ---
        # 1. Net Buy Ratio: (Mua - Bán) / Tổng Vol
        if 'buy_foreign' in df.columns and 'sell_foreign' in df.columns:
            net_foreign = df['buy_foreign'] - df['sell_foreign']
            df['Foreign_Net_Ratio'] = net_foreign / (df['volume'] + 1e-9)
            
            # 2. Cumulative Flow (Dòng tiền tích lũy 5 ngày)
            df['Foreign_Flow_5d'] = df['Foreign_Net_Ratio'].rolling(5).mean()
        else:
            # Fallback nếu thiếu cột
            df['Foreign_Net_Ratio'] = 0
            df['Foreign_Flow_5d'] = 0
        
        # 3. Market Regime
        df['Trend_Regime'] = (df['close'] > df['close'].rolling(50).mean()).astype(int)
        
        # 4. Interaction Features
        df['RSI_MFI_Div'] = df['RSI'] - df['MFI']
        df['Panic_Score'] = df['Vol_10'] * df['Vol_Ratio']
        
        # 5. Lag Returns
        for lag in [1, 3, 5, 10]:
            df[f'Ret_{lag}d'] = df['close'].pct_change(lag)
            
        # Clean Final NaN
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        return df

    @staticmethod
    def apply_cross_sectional_zscore(df_snapshot):
        # Cập nhật danh sách các cột cần tính Z-Score (bao gồm Foreign)
        cols_to_norm = [
            'RSI', 'MACD_Div', 'BB_Pb', 'BB_Width', 'Vol_Ratio', 'MFI',
            'RSI_MFI_Div', 'Panic_Score', 
            'Ret_1d', 'Ret_3d', 'Ret_5d', 'Ret_10d',
            'Foreign_Net_Ratio', 'Foreign_Flow_5d' 
        ]
        
        df_norm = df_snapshot.copy()
        
        for col in cols_to_norm:
            if col not in df_norm.columns: continue
            
            # Tính Z-Score Cross-Sectional
            mean_val = df_norm[col].mean()
            std_val = df_norm[col].std() + 1e-9
            
            df_norm[f'Z_{col}'] = (df_norm[col] - mean_val) / std_val
            df_norm[f'Z_{col}'] = df_norm[f'Z_{col}'].clip(-3, 3).fillna(0)
            
        df_norm['Z_Trend_Regime'] = df_norm['Trend_Regime']
        
        return df_norm
# =============================================================================
# 3. QUANT TOOLKIT
# =============================================================================

class QuantToolkit:
    def __init__(self):
        self.model = xgb.XGBRanker()
        self.features = []
        self.repo = DataRepository()
        self._load_model()

    def _load_model(self):
        if os.path.exists(QuantConfig.MODEL_PATH):
            try:
                self.model.load_model(QuantConfig.MODEL_PATH)
                self.features = joblib.load(QuantConfig.FEATURE_PATH)
            except Exception as e:
                print(f"⚠️ [Quant] Lỗi load model: {e}")
        else:
            print("⚠️ [Quant] Chưa có Model. Cần chạy train_model().")

    def get_market_ranking(self):
        if not self.features:
            return {"error": "Model chưa được huấn luyện."}

        snapshot = []
        
        for ticker in QuantConfig.TICKERS:
            # Lấy 100 ngày gần nhất để tính chỉ báo
            df = self.repo.get_price_history(ticker, days=100)
            
            if df.empty or len(df) < 60: continue
            
            df['date'] = pd.to_datetime(df['date'])
            df_feat = FeatureEngineer.create_base_features(df)
            
            if not df_feat.empty:
                latest = df_feat.iloc[-1:].copy()
                latest['ticker'] = ticker
                snapshot.append(latest)

        if not snapshot:
            return {"error": "Không đủ dữ liệu."}

        df_now = pd.concat(snapshot, ignore_index=True)
        df_now = FeatureEngineer.apply_cross_sectional_zscore(df_now)
        
        try:
            # Đảm bảo đủ feature
            missing_cols = set(self.features) - set(df_now.columns)
            if missing_cols:
                # Fill missing cols với 0 để không crash
                for c in missing_cols: df_now[c] = 0
            
            X = df_now[self.features]
            scores = self.model.predict(X)
            df_now['Rank_Score'] = scores
        except Exception as e:
            return {"error": f"Lỗi dự báo: {str(e)}"}

        df_sorted = df_now.sort_values('Rank_Score', ascending=False)
        min_s, max_s = df_sorted['Rank_Score'].min(), df_sorted['Rank_Score'].max()
        # Tránh chia cho 0 nếu min == max
        div = (max_s - min_s) if (max_s - min_s) > 1e-9 else 1.0
        df_sorted['Confidence'] = (df_sorted['Rank_Score'] - min_s) / div * 100

        top5 = df_sorted.head(5)
        bot5 = df_sorted.tail(5)

        return {
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "top_strong_buy": [
                {
                    "ticker": row['ticker'],
                    "price": row['close'],
                    "confidence": round(row['Confidence'], 1),
                    "reason": f"Z_Score={row['Rank_Score']:.2f}"
                } for _, row in top5.iterrows()
            ],
            "top_avoid_sell": [
                {
                    "ticker": row['ticker'],
                    "price": row['close'],
                    "confidence": round(row['Confidence'], 1),
                    "reason": "Weak Momentum"
                } for _, row in bot5.iterrows()
            ]
        }

    def train_model(self, days_history=3650):
            print(f"🧠 [Quant] Bắt đầu Train Model với dữ liệu {days_history} ngày...")
            
            data_frames = []
            for ticker in QuantConfig.TICKERS:
                df = self.repo.get_price_history(ticker, days=days_history)
                if not df.empty and len(df) > 100:
                    df['ticker'] = ticker
                    data_frames.append(df)
            
            if not data_frames:
                print("❌ DB rỗng. Hãy chạy crawler trước.")
                return

            full_df = pd.concat(data_frames, ignore_index=True)
            full_df['date'] = pd.to_datetime(full_df['date'])
            
            # 1. Feature Engineering
            processed_dfs = []
            for t in full_df['ticker'].unique():
                sub = full_df[full_df['ticker'] == t]
                processed_dfs.append(FeatureEngineer.create_base_features(sub))
            
            train_df = pd.concat(processed_dfs, ignore_index=True)

            # 2. Create Target
            future_close = train_df.groupby('ticker')['close'].shift(-3)
            train_df['Raw_Target'] = (future_close - train_df['close']) / train_df['close']
            train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()

            # 3. Create Rank Target
            def discretize(x):
                try: return pd.qcut(x, 5, labels=False, duplicates='drop')
                except: return np.zeros(len(x))
                
            train_df['Target_Rank'] = train_df.groupby('date')['Raw_Target'].transform(discretize)
            train_df['Target_Rank'] = train_df['Target_Rank'].fillna(2).astype(int)

            # 4. Z-Score Transformation (CẬP NHẬT LIST CỘT MỚI TẠI ĐÂY)
            cols_to_norm = [
                'RSI', 'MACD_Div', 'BB_Pb', 'BB_Width', 'Vol_Ratio', 'MFI',
                'RSI_MFI_Div', 'Panic_Score', 
                'Ret_1d', 'Ret_3d', 'Ret_5d', 'Ret_10d',
                'Foreign_Net_Ratio', 'Foreign_Flow_5d' 
            ]
            
            def to_zscore(x): return (x - x.mean()) / (x.std() + 1e-9)

            for col in cols_to_norm:
                train_df[f'Z_{col}'] = train_df.groupby('date')[col].transform(to_zscore)
                train_df[f'Z_{col}'] = train_df[f'Z_{col}'].clip(-3, 3).fillna(0)
            
            train_df['Z_Trend_Regime'] = train_df['Trend_Regime']

            # 5. Train XGBoost
            train_df = train_df.sort_values('date').reset_index(drop=True)
            features = [c for c in train_df.columns if c.startswith('Z_')]
            
            X = train_df[features]
            y = train_df['Target_Rank']
            qid = train_df.groupby('date').ngroup().values
            
            model = xgb.XGBRanker(
                booster='dart', objective='rank:ndcg',
                n_estimators=2000, learning_rate=0.011,
                max_depth=4, subsample=0.6, colsample_bytree=0.5,
                reg_lambda=50, reg_alpha=10,
                tree_method="hist", n_jobs=-1, random_state=42
            )
            
            print(f"🚀 Fitting DART Model trên {len(X)} mẫu...")
            model.fit(X, y, qid=qid)
            
            model.save_model(QuantConfig.MODEL_PATH)
            joblib.dump(features, QuantConfig.FEATURE_PATH)
            print(f"✅ Model đã lưu tại: {QuantConfig.MODEL_PATH}")
            self._load_model()

if __name__ == "__main__":
    tool = QuantToolkit()
    # Nếu chạy trực tiếp file này, nó sẽ thử train nếu chưa có model
    if not tool.features:
        tool.train_model()
    
    # Test
    res = tool.get_market_ranking()
    print(res)