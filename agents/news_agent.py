import sys
from core.llm import call_llm
from tools.search_tool import SearchToolkit

class NewsAgent:
    async def analyze(self, ticker: str) -> str:
        print(f"📰 [News Agent] Đang lọc tin đồn & sự kiện {ticker}...", file=sys.stderr)
        
        raw_news = await SearchToolkit.search_news(f"Tin tức sự kiện {ticker}", limit=5)
        
        system_prompt = "Bạn là Chuyên gia Phân tích Sự kiện (Event-Driven Analyst)."
        user_prompt = f"""
        Phân tích tin tức cho {ticker}:
        {raw_news}
        
        OUTPUT:
        1. **Phân loại tin:** (Tin Lợi nhuận / M&A / Lãnh đạo / Vĩ mô ngành / Tin đồn).
        2. **Đánh giá tác động:** Ngắn hạn (T+3) vs Dài hạn.
        3. **Sentiment Score:** Thang 1-10.
        """
        
        return await call_llm(system_prompt, user_prompt)

