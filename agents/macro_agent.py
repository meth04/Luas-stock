import sys
from core.llm import call_llm
from tools.search_tool import SearchToolkit

class MacroAgent:
    async def analyze(self) -> str:
        print("🌐 [Macro Agent] Đang phân tích dòng tiền vĩ mô (Async)...", file=sys.stderr)
        
        raw_news = await SearchToolkit.search_macro(limit=6) 
        
        system_prompt = "Bạn là Chuyên gia Chiến lược Vĩ mô (Macro Strategist) tại Hedge Fund."
        user_prompt = f"""
        Phân tích bối cảnh vĩ mô Việt Nam dựa trên tin tức:
        {raw_news}
        
        NHIỆM VỤ:
        Đừng chỉ tóm tắt. Hãy kết nối các điểm dữ liệu (Connect the dots).
        
        OUTPUT (Ngắn gọn, súc tích):
        1. **Dòng tiền & Lãi suất:** SBV đang nới lỏng hay thắt chặt? Lãi suất liên ngân hàng thế nào?
        2. **Tỷ giá & Ngoại khối:** Áp lực tỷ giá USD/VND tác động ra sao đến dòng vốn ngoại?
        3. **Sentiment:** Tích cực / Tiêu cực / Thận trọng.
        """
        
        return await call_llm(system_prompt, user_prompt, temperature=0.4)