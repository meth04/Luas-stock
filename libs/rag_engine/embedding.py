import asyncio
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from .config import settings

_embedding_model = None

def load_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL} (CUDA enabled)...", file=sys.stderr)
        _embedding_model = SentenceTransformer(
            settings.EMBEDDING_MODEL, 
            trust_remote_code=True,
            device="cuda" 
        )
    return _embedding_model

async def bge_m3_embedding(texts: list[str]) -> np.ndarray:
    model = load_embedding_model()
    loop = asyncio.get_event_loop()
    
    embeddings = await loop.run_in_executor(
        None, 
        lambda: model.encode(texts, normalize_embeddings=True, batch_size=2)
    )
    return embeddings