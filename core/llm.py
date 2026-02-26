import os
import json
import aiohttp
from dotenv import load_dotenv

load_dotenv()

# Lấy cấu hình từ .env
API_KEY = os.getenv("CLIPROXY_API_KEY")
BASE_URL = os.getenv("CLIPROXY_BASE_URL")

# Nếu URL kết thúc bằng /v1/, hãy bỏ nó đi để nối chuỗi cho chuẩn
if BASE_URL.endswith("/"):
    BASE_URL = BASE_URL.rstrip("/")
if BASE_URL.endswith("/v1"):
    BASE_URL = BASE_URL.replace("/v1", "")

# Single shared session để reuse TCP connections (tránh TLS handshake lặp lại)
_session = None

async def _get_session():
    global _session
    if _session is None or _session.closed:
        connector = aiohttp.TCPConnector(limit=10, keepalive_timeout=60)
        timeout = aiohttp.ClientTimeout(total=300)
        _session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _session

async def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-5.2", temperature: float = 0.3) -> str:
    """
    Hàm gọi LLM sử dụng aiohttp (truly async, non-blocking).
    Dùng shared session để tối ưu connection reuse.
    """
    url = f"{BASE_URL}/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # Payload chuẩn JSON
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "stream": False
    }

    try:
        session = await _get_session()
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                return f"❌ Lỗi API ({response.status}): {text}"
                
            data = await response.json()
            
            # Trích xuất nội dung trả lời
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
            else:
                return f"❌ Phản hồi API không đúng định dạng: {data}"
            
    except Exception as e:
        return f"❌ Lỗi kết nối LLM: {str(e)}"