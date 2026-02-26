<p align="center">
  <h1 align="center">🏦 VNStock AI Hedge Fund</h1>
  <p align="center">
    <strong>Multi-Agent Investment Decision System for the Vietnam Stock Market</strong>
  </p>
  <p align="center">
    <em>Powered by LLM Agents • RAG Pipeline • Quantitative Ranking • MCP Protocol</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LLM-GPT%20%7C%20DeepSeek%20%7C%20Gemini-green?style=for-the-badge" alt="LLM">
  <img src="https://img.shields.io/badge/RAG-LightRAG-orange?style=for-the-badge" alt="RAG">
  <img src="https://img.shields.io/badge/Quant-XGBoost-red?style=for-the-badge" alt="XGBoost">
  <img src="https://img.shields.io/badge/Protocol-MCP-purple?style=for-the-badge" alt="MCP">
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Agent Pipeline](#-agent-pipeline)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Agents in Detail](#-agents-in-detail)
- [Tools & Data Sources](#-tools--data-sources)
- [RAG Engine](#-rag-engine)
- [Database](#-database)
- [MCP Server](#-mcp-server)
- [Performance](#-performance)
- [License](#-license)

---

## 🌟 Overview

**VNStock AI Hedge Fund** is an autonomous investment decision system that simulates a professional hedge fund research process. It orchestrates **5 specialized AI agents** that run in parallel to collect and analyze multi-dimensional market data for VN30 stocks, then synthesizes their findings through a **Bull vs. Bear debate** before a **Risk Manager agent** delivers the final investment verdict.

### Key Features

- 🤖 **5 Parallel Agents** — Macro, News, Technical, Quant, and Financial analysts working concurrently
- 📊 **RAG-Powered Financial Analysis** — Deep analysis of financial statements using OCR + LightRAG retrieval
- 🧠 **Quantitative Ranking** — XGBoost Learning-to-Rank model scoring stocks across VN30
- ⚔️ **Bull vs. Bear Debate** — Adversarial argumentation grounded in real data
- ⚖️ **Risk Manager** — Portfolio Manager agent delivering actionable BUY/SELL/HOLD decisions
- ⚡ **Async Architecture** — Fully asynchronous pipeline with `asyncio` + `aiohttp` for maximum throughput
- 🗄️ **Persistent Storage** — SQLite database for market data, agent logs, and decision history
- 🔌 **MCP Protocol** — Model Context Protocol server exposing tools for external integration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MAIN ORCHESTRATOR                        │
│                          (main.py)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  MACRO   │ │   NEWS   │ │TECHNICAL │ │  QUANT   │  Parallel│
│  │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │  ────►   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
│       │             │            │             │                │
│       ▼             ▼            ▼             ▼                │
│  SearchToolkit  SearchToolkit  MarketToolkit  QuantToolkit      │
│  (Serper API)   (Serper API)   (vnstock DB)  (XGBoost)         │
│                                                                 │
│  ┌───────────────────────────────────────┐                     │
│  │         FINANCIAL Agent (MCP)         │  Parallel           │
│  │  DynamicFinancialAgent → LightRAG     │  ────►              │
│  └───────────────────────────────────────┘                     │
│                        │                                        │
│                        ▼                                        │
│              ┌─────────────────┐                               │
│              │   FULL REPORT   │                               │
│              └────────┬────────┘                               │
│                       │                                         │
│              ┌────────▼────────┐                               │
│              │  DEBATE AGENT   │  Sequential                   │
│              │  🐂 Bull vs 🐻 Bear │  ────►                    │
│              └────────┬────────┘                               │
│                       │                                         │
│              ┌────────▼────────┐                               │
│              │  RISK MANAGER   │                               │
│              │  Final Verdict  │                               │
│              └────────┬────────┘                               │
│                       │                                         │
│              ┌────────▼────────┐                               │
│              │   💾 Save DB    │                               │
│              └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Agent Pipeline

| Phase | Agents | Mode | Description |
|-------|--------|------|-------------|
| **1. Data Collection** | Macro, News, Technical, Quant, Financial | ⚡ Parallel | All 5 agents gather data concurrently |
| **2. Synthesis** | Debate Agent | 🔄 Sequential | Bull argues for buying, Bear argues for selling — both grounded in data from Phase 1 |
| **3. Decision** | Risk Manager | 🔄 Sequential | Portfolio Manager weighs debate + quant data → BUY/SELL/HOLD verdict |
| **4. Logging** | System | 🔄 Sequential | Decision saved to SQLite with action, confidence, and reasoning |

---

## 📁 Project Structure

```
vnstock/
├── main.py                          # 🚀 Main orchestrator — entry point
├── .env                             # 🔐 API keys & model configuration
├── requirements.txt                 # 📦 Python dependencies
│
├── agents/                          # 🤖 AI Agent modules
│   ├── macro_agent.py               #   Macro economics analyst
│   ├── news_agent.py                #   News & event-driven analyst
│   ├── technical_agent.py           #   Technical chart analyst
│   ├── quant_agent.py               #   Quantitative ranking (XGBoost)
│   ├── financial_agent.py           #   Financial report analyst (legacy)
│   ├── financial_analysis.py        #   Dynamic financial analyst (RAG-powered)
│   ├── debate_agent.py              #   Bull vs Bear debate engine
│   ├── risk_agent.py                #   Risk Manager / Portfolio Manager
│   └── state.py                     #   Shared state schema (TypedDict)
│
├── core/                            # ⚙️ Core infrastructure
│   ├── llm.py                       #   Async LLM client (aiohttp)
│   └── mcp_client.py                #   MCP Protocol client
│
├── tools/                           # 🔧 Data collection tools
│   ├── search_tool.py               #   Web search (Serper API)
│   ├── market_tool.py               #   Technical analysis (pandas)
│   ├── quant_tool.py                #   XGBoost ranking model
│   ├── chart_tool.py                #   Chart generation
│   └── rag_tool.py                  #   RAG query interface
│
├── servers/                         # 🔌 MCP Server
│   └── financial_server.py          #   FastMCP server exposing all tools
│
├── libs/                            # 📚 Libraries
│   └── rag_engine/                  #   LightRAG-based retrieval engine
│       ├── core.py                  #     RAG engine initialization
│       ├── retrieval.py             #     Query routing & answer refinement
│       ├── ingest.py                #     Document ingestion pipeline
│       ├── llm.py                   #     LLM integration for RAG
│       ├── embedding.py             #     Embedding model (BGE-M3)
│       ├── config.py                #     RAG configuration
│       ├── evaluate.py              #     RAGAS evaluation framework
│       └── visualize.py             #     Knowledge graph visualization
│
├── database/                        # 🗄️ Database layer (SQLAlchemy + SQLite)
│   ├── models.py                    #   ORM models (Symbol, OHLCV, AgentLog)
│   └── repo.py                      #   Data repository (CRUD operations)
│
├── jobs/                            # ⏰ Background jobs
│   └── crawler.py                   #   VN30 market data crawler (vnstock API)
│
├── models/                          # 🧠 Trained ML models
│   ├── vn30_ranker_dart.json        #   XGBoost DART ranker model
│   └── rank_features.pkl            #   Feature list for ranking
│
├── data/                            # 💾 SQLite database
│   └── vnstock.db                   #   Market data + agent decision logs
│
├── rag_storage/                     # 📄 RAG index storage
│   └── {TICKER}/{YEAR}/{QUARTER}/   #   Indexed financial reports per stock
│
├── analysis_reports/                # 📝 Cached analysis reports
│   └── {TICKER}_{YEAR}_{QUARTER}.md #   Markdown reports (Write Once, Read Many)
│
└── notebooks/                       # 📓 Jupyter notebooks
    └── quant_tool.ipynb             #   Quant model development notebook
```

---

## ✅ Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **LLM API Access** — Any OpenAI-compatible API endpoint (e.g., CLIProxy, OpenRouter, local LLM server)
- **Serper API Key** — For real-time web/news search ([serper.dev](https://serper.dev))
- **Windows / Linux / macOS** (Windows uses `WindowsSelectorEventLoopPolicy` automatically)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/vnstock-ai-hedge-fund.git
cd vnstock-ai-hedge-fund
```

### 2. Create a Virtual Environment

```bash
python -m venv vnstock
# Windows:
vnstock\Scripts\activate
# Linux/macOS:
source vnstock/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run may take a few minutes to download the embedding model (`BAAI/bge-m3`).

### 4. Initialize the Database

```bash
python -c "from database.models import init_db; init_db()"
```

### 5. Crawl Market Data (First Time)

```bash
python jobs/crawler.py
```

This will download **10 years of daily OHLCV data** for all 30 VN30 stocks from the VCI/TCBS API. This step takes approximately 5-10 minutes on the first run.

---

## ⚙️ Configuration

Create a `.env` file in the project root (or edit the existing one):

```ini
# ─── LLM API Configuration ───
CLIPROXY_BASE_URL=http://127.0.0.1:8317/v1   # OpenAI-compatible API endpoint
CLIPROXY_API_KEY=sk-your-api-key              # API key

# ─── Model Selection ───
ROUTE_MODEL=deepseek-v3.2                     # Metadata routing model
LLM_MODEL_NAME=deepseek-v3.2                  # General LLM model
JUDGE_MODEL_NAME=qwen3-max                    # Evaluation judge model
REASONER_MODEL=deepseek-r1                    # Deep reasoning model (RAG refinement)
GTP_MODEL=gpt-5.2                             # Default model for agents

# ─── Embedding ───
EMBEDDING_MODEL_NAME=BAAI/bge-m3              # Embedding model for RAG

# ─── RAG Configuration ───
WORKDIR=./rag_storage                         # RAG index storage path
MAX_REQUESTS_PER_MINUTE=100                   # Rate limit for embedding API
CHUNK_SIZE=4096                               # Document chunk size
CHUNK_OVERLAP=350                             # Overlap between chunks
EMBEDDING_TIMEOUT=600                         # Embedding request timeout (seconds)

# ─── Search API ───
SERPER_API_KEY=your-serper-api-key             # Serper.dev API key for web search
```

### Configurable Ticker

Edit `main.py` to change the target stock and reporting period:

```python
# --- CẤU HÌNH ---
TARGET_TICKER = "BID"        # VN30 stock ticker
REPORT_YEAR = "2025"         # Financial report year
REPORT_QUARTER = "Q4"        # Financial report quarter
```

---

## 📖 Usage

### Run the Full Pipeline

```bash
python main.py
```

This will:
1. **Launch 5 agents in parallel** to collect macro news, stock news, technical indicators, quant predictions, and financial reports
2. **Run the Bull vs. Bear debate** with data-backed arguments
3. **Generate a portfolio decision** with action, allocation, reasoning, and price targets
4. **Save the decision to the database**

### Expected Output

```
============================================================
 🚀 HEDGE FUND AI SYSTEM - BID (Q4/2025)
============================================================
   ⏱️  [09:18:04] Bắt đầu thu thập dữ liệu đa nguồn (Parallel)...
   ✅ FINANCIAL Agent: Hoàn tất.
   ✅ QUANT Agent: Hoàn tất.
   ✅ MACRO Agent: Hoàn tất.
   ✅ TECHNICAL Agent: Hoàn tất.
   ✅ NEWS Agent: Hoàn tất.
   ⏱️  [09:18:35] ✅ Thu thập xong sau 31.22s

============================================================
 🚀 PHÒNG TRANH BIỆN
============================================================
   ... Bull vs Bear debate transcript ...

============================================================
 🚀 QUYẾT ĐỊNH CỦA GIÁM ĐỐC QUỸ
============================================================
   1. **HÀNH ĐỘNG:** QUAN SÁT
   2. **TỶ TRỌNG:** 0-3% NAV
   3. **LÝ DO CỐT LÕI:** ...
   4. **VÙNG GIÁ:** ...

============================================================
 🚀 HOÀN TẤT: 85.23s
============================================================
```

### Run Individual Components

```bash
# Crawl/update market data
python jobs/crawler.py

# Test individual agents
python agents/test_agents.py

# Start MCP server (for external tool integration)
python servers/financial_server.py
```

---

## 🤖 Agents in Detail

### 1. Macro Agent (`agents/macro_agent.py`)

Analyzes the Vietnam macroeconomic environment by searching for the latest news on interest rates, exchange rates, monetary policy, and capital flows.

- **Data Source:** Serper API (real-time Google News)
- **Output:** SBV policy direction, USD/VND exchange pressure, market sentiment

### 2. News Agent (`agents/news_agent.py`)

Performs event-driven analysis by filtering and categorizing news for the target stock.

- **Data Source:** Serper API (stock-specific news)
- **Output:** News classification (M&A, earnings, leadership), short vs. long-term impact, sentiment score (1-10)

### 3. Technical Agent (`agents/technical_agent.py`)

Runs advanced technical analysis using Price Action and Indicator Confluence methodology.

- **Data Source:** Historical OHLCV data from SQLite (via `MarketToolkit`)
- **Indicators:** RSI, Stochastic RSI, MACD, Bollinger Bands, Ichimoku Cloud, SMA 50/200, Volume Analysis
- **Output:** Market structure phase, confluence signals, trade setup (entry/stoploss/take-profit)

### 4. Quant Agent (`agents/quant_agent.py`)

Runs a machine learning ranking model to score the target stock relative to the VN30 universe.

- **Model:** XGBoost DART Ranker (trained on 10 years of data)
- **Features:** 20+ technical and statistical features, cross-sectional z-score normalization
- **Output:** Relative ranking, confidence score, BUY/NEUTRAL/SELL classification

### 5. Financial Agent (`agents/financial_analysis.py`)

The most sophisticated agent — performs deep fundamental analysis of quarterly financial reports using RAG (Retrieval-Augmented Generation).

- **Data Source:** OCR-processed financial statements indexed in LightRAG
- **Process:**
  1. Determines stock industry (Bank / Real Estate / General)
  2. Generates 15 targeted investigation questions
  3. Runs parallel RAG queries with `Semaphore(5)` concurrency control
  4. Synthesizes findings into a structured Markdown report
- **Output:** Balance sheet analysis, P&L review, cash flow assessment (real vs. paper profit), risk flags, investment conclusion
- **Caching:** Write-Once-Read-Many (WORM) — reports are cached in `analysis_reports/`

### 6. Debate Agent (`agents/debate_agent.py`)

Orchestrates an adversarial debate between a Bull (optimistic) trader and a Bear (pessimistic) trader, both grounded in actual data from the 5 agents above.

### 7. Risk Manager (`agents/risk_agent.py`)

Acts as the Portfolio Manager, weighing the debate arguments against quantitative data to deliver the final investment decision.

---

## 🔧 Tools & Data Sources

| Tool | Module | Type | Description |
|------|--------|------|-------------|
| **SearchToolkit** | `tools/search_tool.py` | Async | Real-time news search via Serper API |
| **MarketToolkit** | `tools/market_tool.py` | Sync (threaded) | Technical indicators from historical OHLCV data |
| **QuantToolkit** | `tools/quant_tool.py` | Sync (threaded) | XGBoost ranking model for VN30 stocks |
| **ChartTool** | `tools/chart_tool.py` | Sync | Interactive HTML chart generation |

---

## 📄 RAG Engine

The RAG engine (`libs/rag_engine/`) is built on top of [LightRAG](https://github.com/HKUDS/LightRAG) and provides:

- **Multi-modal Retrieval:** Hybrid search (keyword + semantic + knowledge graph)
- **Smart Routing:** Automatic ticker/year/quarter detection from natural language queries
- **Financial-Aware Query Expansion:** Industry-specific search query generation (banking, real estate, manufacturing)
- **Answer Refinement:** Strict auditor-grade answer extraction with no hallucination policy
- **Multi-Index:** Separate indexes per `{ticker}/{year}/{quarter}` for precise retrieval

### Ingesting Financial Reports

```bash
# Place OCR-processed PDF text files in the appropriate directory
# Then run the ingestion pipeline
python -m libs.rag_engine.ingest
```

---

## 🗄️ Database

The system uses **SQLite** via SQLAlchemy ORM with 4 tables:

| Table | Purpose |
|-------|---------|
| `symbols` | Stock metadata (ticker, company name, exchange, industry) |
| `market_data_daily` | Historical OHLCV + foreign flow data |
| `market_data_intraday` | Real-time price snapshots |
| `agent_logs` | Decision history (action, confidence, reasoning) |

Database file: `data/vnstock.db`

---

## 🔌 MCP Server

The project includes an **MCP (Model Context Protocol) server** (`servers/financial_server.py`) that exposes all tools as standardized MCP tools:

| MCP Tool | Description |
|----------|-------------|
| `get_macro_news` | Fetch macro economic news |
| `get_stock_news` | Fetch stock-specific news |
| `get_technical_report` | Run technical analysis |
| `get_price_history` | Get historical price data |
| `run_quant_prediction` | Run XGBoost ranking |
| `analyze_financial_report` | Run full RAG-based financial analysis |

### Starting the MCP Server

```bash
python servers/financial_server.py
```

The server communicates via **stdio** and can be integrated with any MCP-compatible client (e.g., Claude Desktop, custom integrations).

---

## ⚡ Performance

The system is optimized for maximum throughput with several key techniques:

| Optimization | Technique |
|-------------|-----------|
| **Async I/O** | `aiohttp` for non-blocking LLM API calls |
| **Parallel Agents** | `asyncio.gather` runs 5 agents concurrently |
| **Parallel RAG** | `Semaphore(5)` limits concurrent RAG queries |
| **Connection Pooling** | Single shared `aiohttp.ClientSession` with `TCPConnector(limit=10)` |
| **Lazy Imports** | Heavy dependencies (xgboost, pandas) loaded on-demand, not at startup |
| **Report Caching** | WORM cache for financial reports in `analysis_reports/` |
| **Model Routing** | Fast models for agent tasks, reasoning models for RAG refinement |

### Typical Runtime (with cached financial report)

| Phase | Time |
|-------|------|
| Data Collection (5 agents parallel) | ~20-30s |
| Debate (2 sequential LLM calls) | ~15-20s |
| Risk Decision (1 LLM call) | ~10-15s |
| **Total** | **~50-70s** |

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` and ensure your virtual environment is activated |
| `SERPER_API_KEY` error | Add your Serper API key to `.env` |
| Empty technical analysis | Run `python jobs/crawler.py` to populate market data |
| RAG returns no data | Ensure financial reports are ingested in `rag_storage/{TICKER}/{YEAR}/{QUARTER}/` |
| `asyncio` event loop error on Windows | The system auto-applies `WindowsSelectorEventLoopPolicy` — ensure Python 3.10+ |

---

## 📄 License

This project is for educational and research purposes. Use at your own risk — the system's investment recommendations are AI-generated and should not be taken as financial advice.

---

<p align="center">
  <em>Built with ❤️ for the Vietnam investment community</em>
</p>
