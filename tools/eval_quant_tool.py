import sys
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from datetime import datetime, timedelta

# Fix path để import module từ thư mục gốc
sys.path.append(os.getcwd())

try:
    from database.repo import DataRepository
    from tools.quant_tool import FeatureEngineer, QuantConfig
except ImportError:
    print("❌ Lỗi Import. Hãy chạy script từ thư mục gốc của dự án.")
    sys.exit(1)

# Cấu hình giao diện biểu đồ
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

class QuantEvaluator:
    def __init__(self):
        self.repo = DataRepository()
        self.model = xgb.XGBRanker()
        self.features = []
        self._load_model()

    def _load_model(self):
        if os.path.exists(QuantConfig.MODEL_PATH):
            try:
                self.model.load_model(QuantConfig.MODEL_PATH)
                self.features = joblib.load(QuantConfig.FEATURE_PATH)
                print(f"✅ Đã load Model DART ({len(self.features)} features).")
            except Exception as e:
                print(f"❌ Lỗi load model: {e}")
                sys.exit(1)
        else:
            print("❌ Chưa có Model. Hãy chạy tools/quant_tool.py để train trước.")
            sys.exit(1)

    def load_test_data(self, test_days=365):
        """Lấy dữ liệu OOS (Out-of-Sample) để đánh giá"""
        print(f"📥 Đang tải dữ liệu {test_days} ngày gần nhất từ DB...")
        
        data_frames = []
        for ticker in QuantConfig.TICKERS:
            # Lấy dư 100 ngày để tính MA, RSI...
            df = self.repo.get_price_history(ticker, days=test_days + 100)
            if not df.empty and len(df) > 60:
                df['ticker'] = ticker
                data_frames.append(df)
        
        if not data_frames:
            print("❌ DB rỗng. Hãy chạy crawler trước.")
            return pd.DataFrame()

        full_df = pd.concat(data_frames, ignore_index=True)
        full_df['date'] = pd.to_datetime(full_df['date'])
        
        print("⚙️ Đang tính toán Feature SOTA (bao gồm Dòng tiền khối ngoại)...")
        
        # 1. Base Features
        processed_dfs = []
        for t in full_df['ticker'].unique():
            sub = full_df[full_df['ticker'] == t]
            processed_dfs.append(FeatureEngineer.create_base_features(sub))
        
        df_feat = pd.concat(processed_dfs, ignore_index=True)
        
        # 2. Tạo Target Thực tế để so sánh (T+3)
        future_close = df_feat.groupby('ticker')['close'].shift(-3)
        df_feat['Actual_Return'] = (future_close - df_feat['close']) / df_feat['close']
        df_feat = df_feat.dropna()
        
        # 3. Z-Score Transformation (Sử dụng logic mới nhất của QuantTool)
        df_test = FeatureEngineer.apply_cross_sectional_zscore(df_feat)
        
        # Lọc đúng ngày Test
        cutoff_date = df_test['date'].max() - timedelta(days=test_days)
        df_test = df_test[df_test['date'] >= cutoff_date]
        
        return df_test.sort_values('date')

    def run_evaluation(self):
        # Đánh giá 1 năm gần nhất (365 ngày)
        df = self.load_test_data(test_days=365)
        if df.empty: return

        print("\n🔮 Đang chạy dự báo trên tập Test...")
        
        # Đảm bảo đủ cột feature
        missing_cols = set(self.features) - set(df.columns)
        for c in missing_cols: df[c] = 0
            
        X = df[self.features]
        # XGBoost predict trả về margin scores
        df['Pred_Score'] = self.model.predict(X)

        # =========================================================================
        # 1. FEATURE IMPORTANCE (FIXED ERROR HERE)
        # =========================================================================
        print("\n📊 1. Phân tích Tầm quan trọng của Features (Feature Importance)...")
        
        # FIX LỖI: Dùng get_booster() trước khi gọi get_score()
        importance = self.model.get_booster().get_score(importance_type='gain')
        
        # Map feature names
        imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Gain'])
        imp_df = imp_df.sort_values('Gain', ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Gain', y='Feature', data=imp_df, palette='viridis')
        plt.title("Top 15 Yếu tố quan trọng nhất (XGBoost DART Feature Importance)")
        plt.tight_layout()
        plt.savefig('eval_feature_importance.png')
        print("   -> Đã lưu: eval_feature_importance.png")
        print("   (Hãy kiểm tra xem 'Z_Foreign...' có nằm trong Top không)")

        # =========================================================================
        # 2. IC ANALYSIS
        # =========================================================================
        print("\n📊 2. Đánh giá Chỉ số IC (Information Coefficient)...")
        ic_list = []
        for date, group in df.groupby('date'):
            if len(group) < 10: continue
            corr, _ = spearmanr(group['Pred_Score'], group['Actual_Return'])
            ic_list.append({'date': date, 'ic': corr})
        
        ic_df = pd.DataFrame(ic_list)
        mean_ic = ic_df['ic'].mean()
        ic_ir = mean_ic / (ic_df['ic'].std() + 1e-9)
        pos_ratio = (ic_df['ic'] > 0).mean()

        print(f"   ➤ Mean IC: {mean_ic:.4f} (Mục tiêu > 0.03)")
        print(f"   ➤ ICIR:    {ic_ir:.4f}   (Mục tiêu > 0.5)")
        print(f"   ➤ Positive Days: {pos_ratio:.1%}")

        plt.figure()
        plt.bar(ic_df['date'], ic_df['ic'], color='skyblue', alpha=0.8)
        plt.axhline(mean_ic, color='red', linestyle='--', label=f'Mean IC: {mean_ic:.3f}')
        plt.title(f"Daily Information Coefficient (IC)")
        plt.legend()
        plt.tight_layout()
        plt.savefig('eval_ic_chart.png')
        print("   -> Đã lưu: eval_ic_chart.png")

        # =========================================================================
        # 3. CUMULATIVE ALPHA (SPREAD)
        # =========================================================================
        print("\n📊 3. Đánh giá Lợi nhuận Alpha (Top vs Bottom)...")
        results = []
        for date, group in df.groupby('date'):
            if len(group) < 10: continue
            
            top5 = group.nlargest(5, 'Pred_Score')
            bot5 = group.nsmallest(5, 'Pred_Score')
            
            ret_top = top5['Actual_Return'].mean()
            ret_bot = bot5['Actual_Return'].mean()
            
            results.append({
                'date': date,
                'Top5_Ret': ret_top,
                'Bot5_Ret': ret_bot,
                'Spread': ret_top - ret_bot
            })
            
        res_df = pd.DataFrame(results)
        res_df['Cum_Top'] = res_df['Top5_Ret'].cumsum()
        res_df['Cum_Bot'] = res_df['Bot5_Ret'].cumsum()
        res_df['Cum_Spread'] = res_df['Spread'].cumsum()

        total_alpha = res_df['Spread'].sum() * 100
        print(f"   ➤ Tổng Alpha (Spread): {total_alpha:.2f}%")
        
        plt.figure()
        plt.plot(res_df['date'], res_df['Cum_Spread'], label='Alpha (Top - Bot)', color='green', linewidth=2.5)
        plt.plot(res_df['date'], res_df['Cum_Top'], label='Long Top 5 Only', color='blue', linestyle='--')
        plt.plot(res_df['date'], res_df['Cum_Bot'], label='Long Bottom 5 (Benchmark)', color='gray', linestyle=':')
        plt.title(f"Cumulative Return (Total Alpha = {total_alpha:.1f}%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig('eval_alpha_chart.png')
        print("   -> Đã lưu: eval_alpha_chart.png")

        # =========================================================================
        # 4. QUINTILE ANALYSIS
        # =========================================================================
        print("\n📊 4. Phân tích Nhóm (Quintile Analysis)...")
        # Chia 5 nhóm
        df['Quintile'] = pd.qcut(df['Pred_Score'], 5, labels=['Q1 (Weak)', 'Q2', 'Q3', 'Q4', 'Q5 (Strong)'])
        quintile_ret = df.groupby('Quintile')['Actual_Return'].mean() * 100
        
        print(quintile_ret)
        
        plt.figure()
        colors = ['#ff4d4d', '#ffa64d', '#ffff4d', '#99ff99', '#009900'] # Red to Green
        ax = quintile_ret.plot(kind='bar', color=colors, edgecolor='black', width=0.6)
        plt.title("Average Return by Quintile (Monotonicity Check)")
        plt.ylabel("Avg Return (%)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('eval_quintile_chart.png')
        print("   -> Đã lưu: eval_quintile_chart.png")
        
        print("\n✅ ĐÁNH GIÁ HOÀN TẤT.")

if __name__ == "__main__":
    evaluator = QuantEvaluator()
    evaluator.run_evaluation()