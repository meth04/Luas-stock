import sys
import asyncio
from core.llm import call_llm

class TechnicalAgent:
    async def analyze(self, ticker: str) -> str:
        print(f"📈 [Technical Agent] Đang soi chart {ticker}...", file=sys.stderr)
        
        # Lazy import: tránh block startup bằng heavy deps (pandas, numpy...)
        def _get_report():
            from tools.market_tool import MarketToolkit
            return MarketToolkit.get_technical_report(ticker)
        
        tech_data = await asyncio.to_thread(_get_report)
        
        system_prompt = "Bạn là Trader chuyên nghiệp theo trường phái Price Action & Indicator Confluence."
        user_prompt = f"""
        Phân tích kỹ thuật mã {ticker} dựa trên dữ liệu:
        {tech_data}
        
        YÊU CẦU:
        1. **Cấu trúc thị trường:** Giá đang ở Phase nào (Tích lũy, Tăng trưởng, Phân phối, Đè giá)?
        2. **Sự hợp lưu (Confluence):** Các chỉ báo (RSI, MACD, Ichimoku, Volume) có đồng thuận không hay mâu thuẫn?
        3. **Setup Giao dịch:**
           - Entry an toàn (Vùng hỗ trợ/Pullback).
           - Stoploss (Bắt buộc).
           - Take Profit (Kháng cự).
        """
        
        return await call_llm(system_prompt, user_prompt)


