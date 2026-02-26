import sys
import os
import argparse
import asyncio
from datetime import datetime

# Hack path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import các Agent
from agents.macro_agent import MacroAgent
from agents.news_agent import NewsAgent
from agents.technical_agent import TechnicalAgent
from agents.quant_agent import QuantAgent
from core.mcp_client import FinancialMCPClient

# Cấu hình màu mè cho dễ nhìn
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def print_box(title, content):
    print(f"\n{Colors.OKBLUE}{'='*60}")
    print(f" 🛠️  DEBUG REPORT: {title.upper()}")
    print(f"{'='*60}{Colors.ENDC}")
    print(content)
    print(f"{Colors.OKBLUE}{'-'*60}{Colors.ENDC}\n")

async def debug_financial(ticker, year, quarter):
    print(f"{Colors.WARNING}>>> Đang test FINANCIAL AGENT qua MCP...{Colors.ENDC}")
    client = FinancialMCPClient()
    try:
        res = await client.call_tool(
            "analyze_financial_report", 
            {"ticker": ticker, "year": year, "quarter": quarter},
            timeout=600 # Set timeout dài cho debug
        )
        print_box(f"FINANCIAL {ticker} {quarter}/{year}", res)
    except Exception as e:
        print(f"{Colors.FAIL}❌ Lỗi Financial: {e}{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description="Công cụ Debug từng Agent cho hệ thống Vnstock AI")
    parser.add_argument("agent", type=str, help="Tên agent cần test: macro, news, tech, quant, financial")
    parser.add_argument("--ticker", type=str, default="HPG", help="Mã cổ phiếu (VD: HPG)")
    parser.add_argument("--year", type=str, default="2024", help="Năm báo cáo (cho Financial)")
    parser.add_argument("--quarter", type=str, default="Q4", help="Quý báo cáo (cho Financial)")

    args = parser.parse_args()
    agent_name = args.agent.lower()
    ticker = args.ticker.upper()

    print(f"🚀 Bắt đầu Debug Agent: {agent_name.upper()} | Ticker: {ticker}")
    start_time = datetime.now()

    try:
        if agent_name == "macro":
            agent = MacroAgent()
            res = agent.analyze()
            print_box("MACRO ECONOMICS", res)

        elif agent_name == "news":
            agent = NewsAgent()
            res = agent.analyze(ticker)
            print_box(f"NEWS SENTIMENT ({ticker})", res)

        elif agent_name == "tech":
            agent = TechnicalAgent()
            res = agent.analyze(ticker)
            print_box(f"TECHNICAL ANALYSIS ({ticker})", res)

        elif agent_name == "quant":
            agent = QuantAgent()
            res = agent.analyze(ticker)
            print_box(f"QUANT PREDICTION ({ticker})", res)

        elif agent_name == "financial":
            if os.name == 'nt':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            asyncio.run(debug_financial(ticker, args.year, args.quarter))

        else:
            print(f"{Colors.FAIL}❌ Không tìm thấy agent tên '{agent_name}'. Các lựa chọn: macro, news, tech, quant, financial{Colors.ENDC}")

    except Exception as e:
        print(f"{Colors.FAIL}❌ CRITICAL ERROR: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()

    print(f"⏱️  Thời gian chạy: {(datetime.now() - start_time).total_seconds():.2f}s")

if __name__ == "__main__":
    main()