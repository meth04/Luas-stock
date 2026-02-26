import os
import numpy as np
import re
import json
import sys
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from .config import settings
from .llm import openai_complete_if_cache
from .embedding import bge_m3_embedding

def safe_to_dict(item):
    """Chuẩn hóa dữ liệu đầu vào thành Dict"""
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return {"content": item, "id": str(hash(item))}
    if isinstance(item, (list, tuple)) and len(item) > 0:
        content = str(item[0])
        score = item[1] if len(item) > 1 else 0.0
        return {"content": content, "score": score, "id": str(hash(content))}
    return {"content": str(item), "id": str(hash(str(item)))}

def extract_json_list(text):
    """
    Dùng Regex để tìm mảng JSON [ ... ] bất kể LLM nói nhảm gì xung quanh.
    """
    try:
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None
    except Exception:
        return None


async def llm_rerank_func(query: str, **kwargs) -> list:
    documents = kwargs.get('documents') or kwargs.get('nodes') or []
    if not documents: return []
    
    # 1. Chuẩn hóa đầu vào
    normalized_docs = [safe_to_dict(doc) for doc in documents]
    
    docs_to_process = normalized_docs
    
    doc_list_str = ""
    for idx, doc in enumerate(docs_to_process):
        content = doc.get('content', '')
        # Chỉ lấy 500 ký tự đầu để LLM đọc nhanh
        preview = content[:500].replace("\n", " ") 
        doc_list_str += f"ID_{idx}: {preview}\n"

    # Prompt được tối ưu để trả về JSON thuần nhất có thể
    prompt = f"""
    ROLE: Financial Data Filter.
    QUERY: "{query}"
    
    DOCUMENTS:
    {doc_list_str}
    
    TASK: Select relevant DOC_IDs that contain specific financial figures (numbers, tables) answering the QUERY.
    OUTPUT: A JSON list of integers only. No markdown. No explanation.
    Example: [0, 5, 12]
    """

    try:
        response = await openai_complete_if_cache(
            prompt, 
            model_name=settings.REASONER_MODEL, 
            temperature=0.0,
            max_tokens=100
        )
        
        selected_ids = extract_json_list(response)
        
        if not selected_ids:
            # Nếu Regex không bắt được, thử fallback đơn giản
            if "[]" in response: return []
            # Nếu lỗi, in ra để debug
            print(f"⚠️ Rerank Parsing Failed. Raw response: {response[:100]}...", file=sys.stderr)
            return normalized_docs[:settings.RERANK_TOP_K]

        # 3. Map lại ID sang Document
        final_results = []
        for idx in selected_ids:
            if isinstance(idx, int) and 0 <= idx < len(docs_to_process):
                final_results.append(docs_to_process[idx])
        
        if not final_results:
            print("⚠️ LLM Rerank returned empty list. Using fallback.", file=sys.stderr)
            return normalized_docs[:settings.RERANK_TOP_K]
            
        print(f"✅ LLM Rerank: Selected {len(final_results)} chunks.", file=sys.stderr)
        return final_results

    except Exception as e:
        print(f"❌ Rerank Exception: {e}. Fallback used.", file=sys.stderr)
        return normalized_docs[:settings.RERANK_TOP_K]


def get_rag_engine(ticker: str, year: str, quarter: str):
    # Cấu trúc scaleable: rag_storage/CTG/2025/Q3
    work_dir = os.path.join(settings.BASE_WORKDIR, ticker.upper(), year, quarter.upper())
    
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    return LightRAG(
        working_dir=work_dir,
        llm_model_func=openai_complete_if_cache,
        llm_model_name=settings.LLM_MODEL,
        llm_model_max_async=12,
        default_embedding_timeout=600,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=bge_m3_embedding
        ),
        # rerank_model_func=llm_rerank_func,
        chunk_token_size=settings.CHUNK_SIZE,
        chunk_overlap_token_size=settings.CHUNK_OVERLAP,
        entity_extract_max_gleaning=1,
    )