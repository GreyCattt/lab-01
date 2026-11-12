import time

import httpx
import pytest

from findex.crawler import fetch


async def mock_handler(request: httpx.Request):
    if "fail_once" in str(request.url):
        # Імітуємо логіку, де перший запит падає з 429, а потім працює (для спрощення в тесті просто віддаємо 429)
        return httpx.Response(429, content=b"Too Many Requests", headers={"Retry-After": "1"})
    return httpx.Response(200, content=b"<html>Success</html>")

@pytest.mark.asyncio
async def test_fetch_success():
    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        stats = {"pending_requests": 0, "errors": 0}
        result = await fetch(client, "https://example.com/ok", stats)
        
        assert result["status"] == 200
        assert result["content"] == b"<html>Success</html>"

@pytest.mark.asyncio
async def test_fetch_retry_429():
    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        stats = {"pending_requests": 0, "errors": 0}
        
        start = time.monotonic()
        result = await fetch(client, "https://example.com/fail_once", stats)
        elapsed = time.monotonic() - start
        
        assert elapsed >= 3.0
        assert result["status"] == 429