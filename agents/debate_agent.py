from core.llm import call_llm

class DebateAgent:
    async def run_debate(self, ticker: str, full_report: str) -> str:
        print(f"⚔️ [Debate Agent] Khởi động cuộc tranh luận Bull vs Bear cho {ticker}...")

        # --- ROUND 1: BULL (Phe Bò) ---
        print("   -> Bull đang phát biểu...")
        bull_prompt = f"""
        Bạn là BULL TRADER (Nhà đầu tư giá lên). Bạn cực kỳ lạc quan, thích rủi ro và luôn tìm kiếm cơ hội tăng trưởng.
        
        DỮ LIỆU THỊ TRƯỜNG:
        {full_report}
        
        NHIỆM VỤ:
        Dựa trên dữ liệu trên, hãy đưa ra luận điểm đanh thép tại sao PHẢI MUA {ticker} NGAY LẬP TỨC.
        Hãy tập trung vào các tín hiệu tích cực (Technical Breakout, Tin tốt, Quant Score cao...).
        Bỏ qua các rủi ro nhỏ nhặt.
        """
        bull_arg = await call_llm("Bạn là Bull Trader hung hăng.", bull_prompt)

        # --- ROUND 2: BEAR (Phe Gấu) ---
        print("   -> Bear đang phản bác...")
        bear_prompt = f"""
        Bạn là BEAR TRADER (Nhà đầu tư giá xuống/Thận trọng). Bạn hoài nghi mọi thứ, lo sợ rủi ro và luôn bảo vệ tiền vốn.
        
        DỮ LIỆU THỊ TRƯỜNG:
        {full_report}
        
        LUẬN ĐIỂM CỦA PHE BÒ:
        "{bull_arg}"
        
        NHIỆM VỤ:
        Hãy phản bác lại luận điểm của Phe Bò. Chỉ ra những lỗ hổng chết người trong lập luận đó.
        Tại sao mua vào lúc này là TỰ SÁT? Hãy nhấn mạnh vào rủi ro vĩ mô, định giá cao hoặc tín hiệu kỹ thuật xấu.
        """
        bear_arg = await call_llm("Bạn là Bear Trader cực kỳ khó tính.", bear_prompt)

        # --- TỔNG HỢP ---
        debate_transcript = f"""
        === 🐂 LUẬN ĐIỂM PHE MUA (BULL) ===
        {bull_arg}

        === 🐻 LUẬN ĐIỂM PHE BÁN (BEAR) ===
        {bear_arg}
        """
        return debate_transcript