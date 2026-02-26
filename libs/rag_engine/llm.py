import asyncio
from openai import AsyncOpenAI
from .config import settings

class RateLimiter:
    def __init__(self, max_calls, period=60.0):
        self.max_calls = max_calls
        self.period = period
        self.semaphore = asyncio.Semaphore(max_calls)
        self.timestamps = []

    async def acquire(self):
        async with self.semaphore:
            now = asyncio.get_event_loop().time()
            self.timestamps = [t for t in self.timestamps if now - t < self.period]
            if len(self.timestamps) >= self.max_calls:
                sleep_time = self.period - (now - self.timestamps[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            self.timestamps.append(asyncio.get_event_loop().time())

limiter = RateLimiter(settings.MAX_RPM)

async def openai_complete_if_cache(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    client = AsyncOpenAI(api_key=settings.API_KEY, base_url=settings.BASE_URL)
    model_name = kwargs.get("model") or kwargs.get("model_name") or settings.LLM_MODEL
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    await limiter.acquire()
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 4096) 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return ""