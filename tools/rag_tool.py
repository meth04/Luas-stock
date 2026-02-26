import asyncio
# Giả sử thư mục rag_engine nằm trong libs/
from libs.rag_engine.retrieval import query_func

class FinancialRAGTool:
    @staticmethod
    async def analyze_report_async(query: str, ticker: str = None) -> str:
        """
        Hỏi đáp với Báo cáo tài chính (Async).
        """
        try:
            # Nếu người dùng không nhập ticker trong query, thêm vào
            full_query = f"{query} của {ticker}" if ticker and ticker not in query else query
            
            # Gọi hàm query_func từ rag_engine
            # mode='hybrid' là tốt nhất cho BCTC
            contexts, answer = await query_func(None, full_query, mode="hybrid")
            
            if not answer:
                return "Không tìm thấy thông tin trong báo cáo tài chính."
                
            return f"**RAG ANALYSIS ({ticker}):**\n{answer}\n\n*Nguồn dữ liệu: {len(contexts)} trích đoạn văn bản.*"
        except Exception as e:
            return f"Lỗi hệ thống RAG: {e}"

    @staticmethod
    def analyze_report_sync(query: str, ticker: str = None) -> str:
        """Phiên bản Sync để dùng cho các Agent không hỗ trợ Async."""
        return asyncio.run(FinancialRAGTool.analyze_report_async(query, ticker))