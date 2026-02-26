import sys
import os
import time
import asyncio
from datetime import datetime
import re

# Hack path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.macro_agent import MacroAgent
from agents.news_agent import NewsAgent
from agents.technical_agent import TechnicalAgent
from agents.quant_agent import QuantAgent
from core.mcp_client import FinancialMCPClient
from core.llm import call_llm
from database.repo import DataRepository

# --- CẤU HÌNH ---
TARGET_TICKER = "BID"
REPORT_YEAR = "2025" 
REPORT_QUARTER = "Q4"

def print_header(title):
    print(f"\n{'='*60}\n 🚀 {title}\n{'='*60}")

def print_step(msg):
    print(f"   ⏱️  [{datetime.now().strftime('%H:%M:%S')}] {msg}")

# Wrapper chạy task song song
async def run_agent_task(name: str, coro):
    try:
        # Nếu là coroutine (async function) thì await
        if asyncio.iscoroutine(coro):
            result = await coro
        else:
            # Nếu là hàm thường (sync) thì chạy trong thread
            result = await asyncio.to_thread(coro)
            
        print(f"   ✅ {name} Agent: Hoàn tất.")
        return result
    except Exception as e:
        print(f"   ❌ {name} Agent: Lỗi ({e})")
        return f"Error: {e}"

# --- LOGIC DEBATE & RISK (ĐÃ SỬA LỖI ASYNC) ---
async def run_debate(ticker, full_report):
    print_step("Khởi động Phòng Tranh Biện...")
    sys_p = "Bạn là Trọng tài tài chính."
    user_p = f"""
    Tình huống: Tranh biện về {ticker}.
    DỮ LIỆU: {full_report}
    NHIỆM VỤ: Đối thoại giữa MR. BULL (Mua) và MR. BEAR (Bán).
    YÊU CẦU: Trích dẫn số liệu cụ thể từ báo cáo.
    OUTPUT: Kịch bản 4 lượt.
    """
    # SỬA LỖI Ở ĐÂY: Gọi trực tiếp await call_llm vì nó đã là async
    return await call_llm(sys_p, user_p, temperature=0.7)

async def run_risk_manager(ticker, debate, quant):
    print_step("Giám đốc Quản trị Rủi ro đang ra quyết định...")
    sys_p = "Bạn là Portfolio Manager."
    user_p = f"""
    MÃ: {ticker} | DEBATE: {debate} | QUANT: {quant}
    QUYẾT ĐỊNH CUỐI CÙNG:
    1. HÀNH ĐỘNG: [MUA/BÁN/QUAN SÁT]
    2. TỶ TRỌNG: % NAV
    3. LÝ DO CỐT LÕI
    4. VÙNG GIÁ
    """
    # SỬA LỖI Ở ĐÂY: Gọi trực tiếp await call_llm
    return await call_llm(sys_p, user_p, temperature=0.2)

def save_log(ticker, verdict):
    repo = DataRepository()
    try:
        action = "QUAN SÁT"
        confidence = "0%"
        action_match = re.search(r"HÀNH ĐỘNG:\*\*?\s*(.*?)\n", verdict, re.IGNORECASE)
        if action_match: action = action_match.group(1).strip()
        conf_match = re.search(r"TỶ TRỌNG:\*\*?\s*(.*?)\n", verdict, re.IGNORECASE)
        if conf_match: confidence = conf_match.group(1).strip()
        
        repo.save_agent_log(ticker, action, confidence, verdict[:1000])
        print("💾 Đã lưu lịch sử vào DB.")
    except: pass
    finally: repo.close()

# --- MAIN FLOW ---
async def main():
    total_start = time.time()
    print_header(f"HEDGE FUND AI SYSTEM - {TARGET_TICKER} ({REPORT_QUARTER}/{REPORT_YEAR})")

    mcp_client = FinancialMCPClient()
    
    macro = MacroAgent()
    news = NewsAgent()
    tech = TechnicalAgent()
    quant = QuantAgent()

    print_step("Bắt đầu thu thập dữ liệu đa nguồn (Parallel)...")

    # Tạo tasks
    task_macro = run_agent_task("MACRO", macro.analyze()) 
    task_news = run_agent_task("NEWS", news.analyze(TARGET_TICKER))
    task_tech = run_agent_task("TECHNICAL", tech.analyze(TARGET_TICKER))
    task_quant = run_agent_task("QUANT", quant.analyze(TARGET_TICKER))
    
    # Financial Agent chạy qua MCP
    task_financial = run_agent_task("FINANCIAL", mcp_client.call_tool(
        "analyze_financial_report", 
        {"ticker": TARGET_TICKER, "year": REPORT_YEAR, "quarter": REPORT_QUARTER}
    ))

    # Chạy song song
    results = await asyncio.gather(
        task_macro, task_news, task_tech, task_quant, task_financial,
        return_exceptions=True
    )

    clean_results = []
    for res in results:
        if isinstance(res, Exception):
            clean_results.append(f"Error: {str(res)}")
        else:
            clean_results.append(str(res))
            
    macro_res, news_res, tech_res, quant_res, fin_res = clean_results
    
    print_step(f"✅ Thu thập xong sau {time.time() - total_start:.2f}s")

    full_report = f"""
    === DATA REPORT: {TARGET_TICKER} ===
    [1] MACRO: {macro_res[:1000]}...
    [2] NEWS: {news_res[:1000]}...
    [3] TECHNICAL: {tech_res[:1000]}...
    [4] QUANT: {quant_res}
    [5] FINANCIAL: {fin_res[:2000]}...
    """

    # Debate & Risk
    debate_transcript = await run_debate(TARGET_TICKER, full_report)
    print_header("PHÒNG TRANH BIỆN")
    print(debate_transcript)
    
    final_verdict = await run_risk_manager(TARGET_TICKER, debate_transcript, quant_res)
    print_header("QUYẾT ĐỊNH CỦA GIÁM ĐỐC QUỸ")
    print(final_verdict)
    
    save_log(TARGET_TICKER, final_verdict)
    print_header(f"HOÀN TẤT: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())