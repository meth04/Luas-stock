import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("CLIPROXY_API_KEY")
    BASE_URL = os.getenv("CLIPROXY_BASE_URL", "http://127.0.0.1:8317/v1")
    
    LLM_MODEL = os.getenv("LLM_MODEL_NAME", "qwen3-max")
    JUDGE_MODEL = os.getenv("JUDGE_MODEL_NAME", "qwen3-max")
    REASONER_MODEL = os.getenv("REASONER_MODEL", "deepseek-r1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    GPT_MODEL=os.getenv("GPT_MODEL", "gpt-5.2")
    RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
    RERANK_TOP_K = 20
    
    BASE_WORKDIR = os.getenv("WORKDIR", "./rag_storage")
    MAX_RPM = int(os.getenv("MAX_REQUESTS_PER_MINUTE", 50))
    
    CHUNK_SIZE = 4096 
    CHUNK_OVERLAP = 300

settings = Config()