import asyncio
import os
import sys
import json
from datetime import datetime

# --- CẤU HÌNH ĐƯỜNG DẪN HỆ THỐNG ---
# Giả sử file này nằm ở: vnstock/agents/financial_analysis.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Trỏ về thư mục vnstock
sys.path.append(project_root)

# Cấu hình đường dẫn cho LightRAG (đã chuyển ra thư mục gốc)
RAG_STORAGE_PATH = os.path.join(project_root, "rag_storage")
os.environ["WORKDIR"] = RAG_STORAGE_PATH

# --- HÀM LOGGING (Dùng stderr để không ảnh hưởng output của MCP) ---
def log_progress(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [FinancialAgent] {msg}", file=sys.stderr)

# --- IMPORT MODULES ---
try:
    from libs.rag_engine.retrieval import query_func
    from core.llm import call_llm
except ImportError as e:
    log_progress(f"❌ Lỗi Import: {e}. Hãy đảm bảo bạn đang đứng ở thư mục gốc 'vnstock'.")
    # Không exit ở đây để code có thể được import bởi tool khác

class DynamicFinancialAgent:
    def __init__(self, ticker: str, year: str, quarter: str, output_dir: str = "analysis_reports"):
        self.ticker = ticker.upper()
        self.year = str(year)
        self.quarter = quarter.upper()
        
        # Đường dẫn output: vnstock/analysis_reports
        self.output_dir = os.path.join(project_root, output_dir)
        
        # Kiểm tra nhanh xem Index có tồn tại không
        index_path = os.path.join(RAG_STORAGE_PATH, self.ticker, self.year, self.quarter)
        if not os.path.exists(index_path):
            log_progress(f"⚠️ CẢNH BÁO: Chưa tìm thấy dữ liệu Index tại: {index_path}")
            log_progress(f"👉 Kết quả phân tích có thể sẽ rỗng.")

        # Tạo thư mục output nếu chưa có
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_report_path(self):
        """Trả về đường dẫn file báo cáo: BID_2025_Q4.md"""
        filename = f"{self.ticker}_{self.year}_{self.quarter}.md"
        return os.path.join(self.output_dir, filename)

    async def analyze(self) -> str:
        """Quy trình chính: Check Cache -> RAG -> LLM -> Save"""
        report_path = self._get_report_path()
        
        # 1. KIỂM TRA CACHE
        if os.path.exists(report_path):
            log_progress(f"⚡ CACHE HIT: Tìm thấy báo cáo tại {report_path}")
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return content
            except Exception as e:
                log_progress(f"⚠️ Lỗi đọc cache: {e}. Sẽ tiến hành phân tích lại.")

        # 2. PHÂN TÍCH MỚI
        log_progress(f"🐢 CACHE MISS: Bắt đầu phân tích sâu cho {self.ticker} {self.quarter}/{self.year}...")
        
        # 2.1 Lấy bộ câu hỏi chuyên sâu
        industry, questions = self._get_deep_questions()
        
        # 2.2 Thu thập dữ liệu từ RAG
        evidence_chain = await self._gather_evidence(questions)
        
        if not evidence_chain:
            return f"❌ LỖI: Không tìm thấy bất kỳ dữ liệu nào cho {self.ticker} {self.quarter}/{self.year}. Vui lòng kiểm tra lại việc Index dữ liệu."

        # 2.3 Viết báo cáo (LLM)
        final_report = await self._write_final_report(industry, evidence_chain)

        # 3. LƯU FILE
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            log_progress(f"✅ Đã lưu báo cáo vào: {report_path}")
        except Exception as e:
            log_progress(f"❌ Lỗi lưu file: {e}")

        return final_report

    def _get_deep_questions(self) -> tuple[str, list[str]]:
        """
        Sinh bộ câu hỏi 15-20 câu tùy theo ngành nghề.
        """
        log_progress(f"🧠 [Router] Xác định ngành và lập kế hoạch điều tra...")
        
        # Danh sách mã ngành (Hardcode cho chính xác)
        banks = ["ACB", "BID", "CTG", "HDB", "LPB", "MBB",
                "SHB", "SSB", "STB", "TCB", "TPB", "VCB",
                "VIB", "VPB"]
        real_estate = ["BCM", "VHM", "VIC", "VRE"]
        
        industry = "GENERAL"
        if self.ticker in banks:
            industry = "BANK"
        elif self.ticker in real_estate:
            industry = "REAL_ESTATE"

        log_progress(f"   => Ngành: {industry}")

        # --- BỘ CÂU HỎI CHUNG (Dùng cho mọi công ty) ---
        base_questions = [
            f"Ý kiến kiểm toán viên về báo cáo tài chính quý {self.quarter} năm {self.year} của {self.ticker} là gì (chấp nhận toàn phần hay ngoại trừ)?",
            f"Tổng tài sản và Vốn chủ sở hữu của {self.ticker} tại cuối quý {self.quarter}/{self.year} là bao nhiêu?",
            f"Lưu chuyển tiền thuần từ hoạt động kinh doanh (CFO) của {self.ticker} quý {self.quarter} năm {self.year} dương hay âm? Giá trị cụ thể?",
            f"So sánh Lợi nhuận sau thuế và Dòng tiền kinh doanh của {self.ticker} trong quý {self.quarter} năm {self.year}?",
            f"Thuyết minh về các giao dịch với bên liên quan trọng yếu của {self.ticker} trong {self.quarter} {self.year} ?",
            f"Có khoản mục nào bất thường trong Báo cáo kết quả kinh doanh của {self.ticker} quý {self.quarter} năm {self.year} không?"
        ]

        specific_questions = []

        if industry == "BANK":
            specific_questions = [
                f"Dư nợ cho vay khách hàng của {self.ticker} tại thời điểm 31/12/{self.year}?",
                f"Tổng tiền gửi của khách hàng tại {self.ticker} tăng trưởng ra sao trong {self.quarter} năm {self.year}?",
                f"Tỷ lệ nợ xấu (Nợ nhóm 3, 4, 5) của {self.ticker} quý {self.quarter} năm {self.year} biến động thế nào?",
                f"Chi phí dự phòng rủi ro tín dụng của {self.ticker} quý {self.quarter} năm {self.year} là bao nhiêu? So với cùng kỳ năm trước?",
                f"Thu nhập lãi thuần (NII) của {self.ticker} trong quý {self.quarter} năm {self.year}?",
                f"Lãi thuần từ hoạt động dịch vụ của {self.ticker} quý {self.quarter} năm {self.year}?",
                f"Tỷ lệ bao phủ nợ xấu (Dự phòng/Nợ xấu) có xu hướng ra sao của {self.ticker} quý {self.quarter} năm {self.year}?",
                f"Thuyết minh về cơ cấu nợ vay theo nhóm nợ của {self.ticker} quý {self.quarter} năm {self.year}?",
                f"Lợi nhuận trước thuế của {self.ticker} hoàn thành bao nhiêu % kế hoạch trong quý {self.quarter} năm {self.year} (nếu có thông tin)?"
            ]
        
        elif industry == "REAL_ESTATE":
            specific_questions = [
                f"Giá trị Hàng tồn kho của {self.ticker} tại quý {self.quarter} {self.year}? Dự án nào chiếm tỷ trọng lớn?",
                f"Chi phí sản xuất kinh doanh dở dang tập trung ở các dự án nào của {self.ticker} trong {self.quarter} {self.year}?",
                f"Khoản mục 'Người mua trả tiền trước' (Doanh thu chưa thực hiện) của {self.ticker} trong quý {self.quarter} năm {self.year} là bao nhiêu?",
                f"Vay và nợ thuê tài chính (Ngắn hạn + Dài hạn) của {self.ticker} trong quý {self.quarter} năm {self.year} là bao nhiêu?",
                f"Hệ số Nợ vay / Vốn chủ sở hữu của {self.ticker} trong quý {self.quarter} năm {self.year} đang ở mức nào?",
                f"Doanh thu thuần từ bán hàng và cung cấp dịch vụ của {self.ticker} trong quý {self.quarter} năm {self.year}?",
                f"Lợi nhuận gộp và Biên lợi nhuận gộp của {self.ticker} trong quý {self.quarter} năm {self.year}?",
                f"Dòng tiền từ hoạt động đầu tư của {self.ticker} trong quý {self.quarter} năm {self.year} là dương hay âm? Và giá trị là bao nhiêu?",
                f"Tiền và tương đương tiền cuối kỳ của {self.ticker} trong quý {self.quarter} năm {self.year} còn bao nhiêu?"
            ]

        else: # GENERAL (Sản xuất, Bán lẻ...)
            specific_questions = [
                f"Doanh thu thuần của {self.ticker} {self.quarter} năm {self.year} tăng hay giảm so với cùng kỳ?",
                f"Giá vốn hàng bán và Lợi nhuận gộp của {self.ticker} {self.quarter} năm {self.year}?",
                f"Chi phí bán hàng và Chi phí quản lý doanh nghiệp {self.ticker} {self.quarter} năm {self.year} chiếm bao nhiêu % doanh thu?",
                f"Chi phí tài chính (đặc biệt là lãi vay) của {self.ticker} {self.quarter} năm {self.year} là bao nhiêu?",
                f"Các khoản Phải thu ngắn hạn của khách hàng của {self.ticker} {self.quarter} năm {self.year} là bao nhiêu? Có tăng mạnh không?",
                f"Hàng tồn kho (Nguyên vật liệu, Thành phẩm) của {self.ticker} {self.quarter} năm {self.year} biến động thế nào?",
                f"Vay và nợ thuê tài chính ngắn hạn/dài hạn của {self.ticker} trong quý {self.quarter} năm {self.year} là bao nhiêu?",
                f"Lãi cơ bản trên cổ phiếu (EPS) của {self.ticker} trong quý {self.quarter} năm {self.year} là bao nhiêu?",
                f"{self.ticker} trong quý {self.quarter} năm {self.year} có khoản chi phí xây dựng cơ bản dở dang nào lớn không?"
            ]

        full_list = base_questions + specific_questions
        log_progress(f"   => Đã lập danh sách {len(full_list)} câu hỏi điều tra.")
        return industry, full_list

    async def _gather_evidence(self, questions: list[str]) -> str:
        """
        Chạy RAG song song với Semaphore. Chỉ giữ lại những câu trả lời có dữ liệu thực tế.
        """
        log_progress(f"🕵️ [Executor] Bắt đầu thu thập dữ liệu (song song, max 5 luồng)...")
        
        semaphore = asyncio.Semaphore(5)
        
        async def _query_one(i: int, q: str):
            async with semaphore:
                if i % 3 == 0:
                    log_progress(f"   Processing {i+1}/{len(questions)}...")
                try:
                    contexts, ai_ans = await query_func(None, q, mode="hybrid")
                    
                    if not contexts or "không tìm thấy" in ai_ans.lower() or "không có thông tin" in ai_ans.lower():
                        return None

                    raw_ocr = "\n".join(contexts[:2]) 
                    
                    entry = f"""
                ---
                ❓ VẤN ĐỀ: {q}
                💡 TÓM TẮT AI: {ai_ans}
                📄 BẰNG CHỨNG GỐC: 
                {raw_ocr[:800]} ... (đã cắt bớt)
                """
                    return entry
                    
                except Exception as e:
                    log_progress(f"      ⚠️ Lỗi truy vấn câu hỏi '{q}': {e}")
                    return None

        tasks = [_query_one(i, q) for i, q in enumerate(questions)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        evidence_buffer = [r for r in results if isinstance(r, str)]

        valid_evidence = "\n".join(evidence_buffer)
        log_progress(f"✅ Thu thập xong. Tìm thấy dữ liệu cho {len(evidence_buffer)}/{len(questions)} câu hỏi.")
        return valid_evidence

    async def _write_final_report(self, industry: str, evidence: str) -> str:
        log_progress(f"✍️ [Analyst] Đang tổng hợp và viết báo cáo Markdown...")
        
        system_prompt = f"""
        ROLE: Bạn là Giám đốc Phân tích Đầu tư (Head of Research) chuyên về thị trường Việt Nam.
        Phong cách: Sắc sảo, Hoài nghi (Skeptical), Dựa trên số liệu (Data-driven).
        
        BỐI CẢNH:
        - Mã CK: {self.ticker}
        - Ngành: {industry}
        - Kỳ báo cáo: Quý {self.quarter} Năm {self.year}
        """

        user_prompt = f"""
        Dựa trên HỒ SƠ ĐIỀU TRA (EVIDENCE) dưới đây, hãy viết một báo cáo phân tích đầu tư chuyên sâu.
        
        LƯU Ý QUAN TRỌNG:
        1. Chỉ sử dụng thông tin có trong EVIDENCE. Nếu không có số liệu, KHÔNG ĐƯỢC BỊA ĐẶT.
        2. Nếu thông tin bị thiếu, hãy ghi chú là "Chưa có dữ liệu trong tài liệu cung cấp".
        
        HỒ SƠ ĐIỀU TRA:
        {evidence}

        YÊU CẦU CẤU TRÚC BÁO CÁO (MARKDOWN):
        
        # BÁO CÁO PHÂN TÍCH TÀI CHÍNH: {self.ticker} - {self.quarter}/{self.year}

        ## 1. Tổng quan & Chất lượng Báo cáo
        - Ý kiến kiểm toán (Có ngoại trừ hay nhấn mạnh gì không?)
        - Đánh giá sơ bộ về mức độ tin cậy của số liệu.

        ## 2. Sức khỏe Tài chính (Balance Sheet)
        - Phân tích Tài sản & Nguồn vốn.
        - Với Ngân hàng: Nhấn mạnh Nợ xấu (NPL), Trích lập dự phòng.
        - Với DN khác: Nhấn mạnh Tồn kho, Phải thu, Nợ vay/VCSH.
        
        ## 3. Hiệu quả Kinh doanh (P&L)
        - Doanh thu & Lợi nhuận tăng hay giảm? Tại sao?
        - Các chỉ số biên lợi nhuận (nếu tính được).

        ## 4. Soi Dòng tiền (Cash Flow - QUAN TRỌNG NHẤT)
        - So sánh Lợi nhuận sau thuế vs Dòng tiền kinh doanh (CFO).
        - Kết luận: Công ty đang "Lãi thật" hay "Lãi giấy"?

        ## 5. Rủi ro & Cơ hội
        - Red Flags (Cờ đỏ): Những điểm cần cảnh giác.
        - Cơ hội đầu tư: Những điểm sáng.
        
        ## 6. Kết luận
        - Quan điểm: Tích cực / Trung lập / Tiêu cực.
        """

        report_content = await call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="gemini-3-flash-preview", 
            temperature=0.3
        )
        return report_content

# if __name__ == "__main__":
#     if len(sys.argv) >= 4:
#         ticker = sys.argv[1]
#         year = sys.argv[2]
#         quarter = sys.argv[3]
        
#         print(f"🚀 Khởi động Financial Analyst Agent cho {ticker}...")
#         agent = DynamicFinancialAgent(ticker, year, quarter)
        
#         # Chạy Async
#         final_result = asyncio.run(agent.analyze())
        
#         print("\n" + "="*40)
#         print("RESULT:")
#         print(final_result)
#         print("="*40)
#     else:
#         print("Usage: python -m agents.financial_analysis <TICKER> <YEAR> <QUARTER>")