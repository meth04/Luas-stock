import asyncio

class QuantAgent:
    async def analyze(self, ticker: str) -> str:
        print(f"🤖 [Quant Agent] Đang chạy mô hình xếp hạng DART cho {ticker}...")
        
        def _run_quant():
            # Lazy import: tránh block startup bằng heavy deps (xgboost, pandas, numpy...)
            from tools.quant_tool import QuantToolkit
            
            tool = QuantToolkit()
            if not tool.features: tool.train_model()
            
            result = tool.get_market_ranking()
            if "error" in result: return f"❌ Lỗi Quant: {result['error']}"
            
            target_info = "Neutral"
            score = 50.0
            for item in result.get("top_strong_buy", []):
                if item['ticker'] == ticker:
                    target_info = "STRONG BUY"
                    score = item['confidence']
                    break
            
            return f"""
            ### 🤖 DỰ BÁO ĐỊNH LƯỢNG
            - **Mã:** {ticker}
            - **Xếp hạng:** {target_info}
            - **Điểm:** {score:.1f}
            """

        # Quant train model rất nặng -> đẩy vào Thread
        quant_report = await asyncio.to_thread(_run_quant)
        
        return f"""
        BÁO CÁO ĐỊNH LƯỢNG (AI RANKING):
        -------------------------------
        {quant_report}
        -------------------------------
        *Phương pháp: Learning to Rank (So sánh sức mạnh tương đối trong VN30).*
        """


