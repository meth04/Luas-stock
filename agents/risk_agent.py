from core.llm import call_llm

class RiskAgent:
    async def make_decision(self, ticker: str, debate_transcript: str, quant_score: str) -> str:
        print(f"⚖️ [Risk Manager] Đang cân nhắc quyết định cuối cùng cho {ticker}...")
        
        system_prompt = "Bạn là Giám đốc Quỹ đầu tư (Portfolio Manager). Nhiệm vụ của bạn là bảo toàn vốn và kiếm lợi nhuận bền vững."
        
        user_prompt = f"""
        Bạn vừa nghe cuộc tranh luận giữa Bull và Bear về mã cổ phiếu {ticker}.
        
        BIÊN BẢN TRANH LUẬN:
        {debate_transcript}
        
        DỮ LIỆU ĐỊNH LƯỢNG (QUAN TRỌNG):
        {quant_score}
        
        YÊU CẦU RA QUYẾT ĐỊNH:
        Dựa trên tất cả thông tin, hãy đưa ra quyết định cuối cùng (Final Verdict).
        
        Output format (Markdown):
        1. **QUYẾT ĐỊNH:** [MUA MẠNH / MUA THĂM DÒ / GIỮ / BÁN / QUAN SÁT]
        2. **Tỷ trọng khuyến nghị:** (Ví dụ: 0%, 10%, 30% NAV)
        3. **Lý do cốt lõi:** Tổng hợp lý do chính khiến bạn ra quyết định này (cân nhắc giữa Rủi ro và Lợi nhuận).
        4. **Kế hoạch hành động:** Vùng giá mua/bán và điểm cắt lỗ (Stoploss).
        """
        
        return await call_llm(system_prompt, user_prompt)