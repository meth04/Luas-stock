import os
import json
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()

class SearchToolkit:
    """
    Công cụ tìm kiếm sử dụng Serper Dev API (Async Version).
    """
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")

    @staticmethod
    async def _call_serper_async(query: str, type: str = "search", limit: int = 5) -> str:
        if not SearchToolkit.SERPER_API_KEY:
            return "❌ Lỗi: Chưa cấu hình SERPER_API_KEY trong file .env"

        endpoint = "https://google.serper.dev/news" if type == "news" else "https://google.serper.dev/search"
        
        payload = {
            "q": query,
            "gl": "vn",
            "hl": "vi",
            "num": limit,
            "tbs": "qdr:w" if type == "news" else None
        }
        
        headers = {
            'X-API-KEY': SearchToolkit.SERPER_API_KEY,
            'Content-Type': 'application/json'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        return f"❌ Lỗi Serper API ({response.status}): {text}"
                    
                    results = await response.json()
                    
                    items = results.get("news", []) if type == "news" else results.get("organic", [])
                    
                    if not items:
                        return f"Không tìm thấy kết quả nào cho: {query}"

                    summary = []
                    for item in items:
                        title = item.get('title', 'No Title')
                        link = item.get('link', '#')
                        snippet = item.get('snippet', '')
                        date = item.get('date', '')
                        source = item.get('source', 'Google')
                        
                        summary.append(f"- [{date}] **{title}** ({source})\n  _{snippet}_\n  Link: {link}")
                    
                    return "\n\n".join(summary)

        except Exception as e:
            return f"❌ Lỗi kết nối: {str(e)}"

    @staticmethod
    async def search_news(query: str, limit: int = 5) -> str:
        """
        Tìm tin tức theo từ khóa (Async).
        """
        if len(query.split()) < 2:
            search_query = f"tin tức tài chính mới nhất về cổ phiếu {query}"
        else:
            search_query = query
            
        return await SearchToolkit._call_serper_async(search_query, type="news", limit=limit)

    @staticmethod
    async def search_macro(limit: int = 5) -> str:
        """Tìm tin tức vĩ mô (Async)."""
        query = "tin tức vĩ mô kinh tế Việt Nam lãi suất tỷ giá ngân hàng nhà nước mới nhất"
        return await SearchToolkit._call_serper_async(query, type="news", limit=limit)