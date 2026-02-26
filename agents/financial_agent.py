import os
import json
from datetime import datetime

class FinancialAgent:
    """
    Agent quản lý phân tích cơ bản (Fundamental).
    Cơ chế: Write Once, Read Many (WORM).
    """
    CACHE_DIR = "data/financial_reports/analyzed"

    def __init__(self):
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)

    def _get_current_quarter(self):
        """Xác định quý hiện tại (VD: 2024_Q1)."""
        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1
        return f"{now.year}_Q{quarter}"

    def analyze(self, ticker: str) -> str:
        """
        1. Check xem đã có báo cáo quý này chưa.
        2. Nếu có -> Load file JSON trả về.
        3. Nếu chưa -> (Placeholder) Chạy RAG -> Lưu file -> Trả về.
        """
        ticker = ticker.upper()
        current_quarter = self._get_current_quarter()
        filename = f"{ticker}_{current_quarter}.json"
        filepath = os.path.join(self.CACHE_DIR, filename)

        # 1. Kiểm tra Cache
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"[CACHE HIT] Đã tìm thấy phân tích BCTC {ticker} {current_quarter}")
                return data['analysis']

        # 2. Nếu chưa có -> Chạy logic phân tích mới (Sau này lắp RAG vào đây)
        print(f"[CACHE MISS] Chưa có phân tích cho {ticker} {current_quarter}. Đang tạo mới...")
        
        # --- PLACEHOLDER CHO RAG SAU NÀY ---
        # analysis = await FinancialRAGTool.analyze(ticker)
        # Tạm thời trả về nội dung giả lập để test luồng
        analysis = f"""
        (Bản phân tích được lưu Cache vào {datetime.now()})
        - Tình hình tài chính {ticker} {current_quarter}:
        - Doanh thu: Chưa có dữ liệu RAG (Chờ tích hợp).
        - Lợi nhuận: Ổn định.
        - Nợ vay: Ở mức an toàn.
        """
        # -------------------------------------

        # 3. Lưu vào Cache
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "ticker": ticker,
                "period": current_quarter,
                "analysis": analysis,
                "created_at": str(datetime.now())
            }, f, ensure_ascii=False, indent=2)
            
        return analysis