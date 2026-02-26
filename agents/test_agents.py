import sys
import os
import asyncio

# Hack path để Python tìm thấy các module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.macro_agent import MacroAgent
from agents.news_agent import NewsAgent
from agents.technical_agent import TechnicalAgent
from agents.quant_agent import QuantAgent

def print_separator(title):
    print(f"\n{'='*60}\n 🕵️  REPORT: {title}\n{'='*60}")

def main():
    ticker = "HPG" # Mã cổ phiếu để test
    
    print(f"🚀 KHỞI ĐỘNG HỆ THỐNG TRADING AGENTS CHO MÃ: {ticker}\n")

    # 1. MACRO AGENT
    try:
        macro = MacroAgent()
        result = macro.analyze()
        print_separator("MACRO ECONOMICS")
        print(result)
    except Exception as e:
        print(f"❌ Lỗi Macro Agent: {e}")

    # 2. NEWS AGENT
    try:
        news = NewsAgent()
        result = news.analyze(ticker)
        print_separator(f"NEWS SENTIMENT ({ticker})")
        print(result)
    except Exception as e:
        print(f"❌ Lỗi News Agent: {e}")

    # 3. TECHNICAL AGENT
    try:
        tech = TechnicalAgent()
        result = tech.analyze(ticker)
        print_separator(f"TECHNICAL ANALYSIS ({ticker})")
        print(result)
    except Exception as e:
        print(f"❌ Lỗi Technical Agent: {e}")

    # 4. QUANT AGENT
    try:
        quant = QuantAgent()
        result = quant.analyze(ticker)
        print_separator(f"QUANT PREDICTION ({ticker})")
        print(result)
    except Exception as e:
        print(f"❌ Lỗi Quant Agent: {e}")

    print("\n✅ ĐÃ HOÀN THÀNH PHIÊN LÀM VIỆC CỦA CÁC AGENT.")

if __name__ == "__main__":
    main()