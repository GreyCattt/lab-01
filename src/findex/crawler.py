import asyncio
import logging
import random
import time
import urllib.parse
from collections.abc import AsyncGenerator
from typing import Any
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("findex.crawler")

def canonicalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path if parsed.path else "/"
    
    query = urllib.parse.parse_qsl(parsed.query)
    query.sort()
    encoded_query = urllib.parse.urlencode(query)
    
    return urllib.parse.urlunparse((scheme, netloc, path, parsed.params, encoded_query, ""))

class RobotsCache:
    def __init__(self, client: httpx.AsyncClient, user_agent: str) -> None:
        self.client = client
        self.user_agent = user_agent
        self.cache: dict[str, RobotFileParser] = {}
        self.lock = asyncio.Lock()

    async def is_allowed(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        async with self.lock:
            if base_url not in self.cache:
                parser = RobotFileParser()
                try:
                    resp = await self.client.get(f"{base_url}/robots.txt", timeout=5.0)
                    if resp.status_code == 200:
                        parser.parse(resp.text.splitlines())
                    else:
                        # Якщо 404, просто передаємо правило "дозволено все"
                        parser.parse(["User-agent: *", "Disallow:"])
                except httpx.RequestError:
                    # Якщо взагалі не вдалося з'єднатися, теж дозволяємо
                    parser.parse(["User-agent: *", "Disallow:"])        
                    
                self.cache[base_url] = parser

        return self.cache[base_url].can_fetch(self.user_agent, url)

async def fetch(client: httpx.AsyncClient, url: str, stats: dict[str, int]) -> dict[str, Any]:
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries + 1):
        start_time = time.monotonic()
        stats["pending_requests"] += 1
        
        try:
            async with asyncio.timeout(10.0):
                response = await client.get(url)
                elapsed = time.monotonic() - start_time
                
                if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    retry_after = response.headers.get("Retry-After")
                    delay = int(retry_after) if retry_after and retry_after.isdigit() else (base_delay * (2 ** attempt) + random.uniform(0, 1))
                    
                    logger.warning(f"URL: {url} | STATUS: {response.status_code} | RETRY: {attempt+1} | DELAY: {delay:.2f}s")
                    stats["pending_requests"] -= 1
                    await asyncio.sleep(delay)
                    continue
                
                logger.info(f"URL: {url} | STATUS: {response.status_code} | BYTES: {len(response.content)} | TIME: {elapsed:.2f}s")
                stats["pending_requests"] -= 1
                return {"status": response.status_code, "content": response.content}

        except (TimeoutError, httpx.RequestError):
            elapsed = time.monotonic() - start_time
            stats["pending_requests"] -= 1
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"URL: {url} | STATUS: TIMEOUT/ERROR | RETRY: {attempt+1} | DELAY: {delay:.2f}s")
                await asyncio.sleep(delay)
                continue
            
            logger.error(f"URL: {url} | STATUS: FAILED | TIME: {elapsed:.2f}s")
            stats["errors"] += 1
            return {"status": "ERROR", "content": b""}
            
    return {"status": "ERROR", "content": b""}

async def crawl_worker(
    queue: asyncio.Queue[str], 
    client: httpx.AsyncClient, 
    global_sem: asyncio.Semaphore, 
    host_sems: dict[str, asyncio.Semaphore], 
    robots: RobotsCache, 
    results_queue: asyncio.Queue[dict[str, str]], 
    seen: set[str], 
    stats: dict[str, int]
) -> None:
    while True:
        url = await queue.get()
        stats["active_tasks"] = stats.get("active_tasks", 0) + 1
        host = urllib.parse.urlparse(url).netloc

        if host not in host_sems:
            host_sems[host] = asyncio.Semaphore(1)

        async with global_sem, host_sems[host]:
            if await robots.is_allowed(url):
                result = await fetch(client, url, stats)

                if isinstance(result["status"], int) and result["status"] == 200:
                    content: bytes = result["content"]
                    soup = BeautifulSoup(content.decode('utf-8', errors='ignore'), "html.parser")
                    text = soup.get_text(separator=' ', strip=True)

                    for a in soup.find_all("a", href=True):
                        href = a['href']
                        if isinstance(href, str):
                            next_url = canonicalize_url(urllib.parse.urljoin(url, href))
                            if next_url not in seen and next_url.startswith("http"):
                                seen.add(next_url)
                                await queue.put(next_url)

                    await results_queue.put({"url": url, "text": text})

            await asyncio.sleep(1.0)
        
        queue.task_done()
        stats["active_tasks"] -= 1
        stats["queue_depth"] = queue.qsize()

async def crawl(
    start_urls: list[str], 
    max_pages: int, 
    concurrency: int, 
    stats: dict[str, int]
) -> AsyncGenerator[dict[str, str], None]:
    if not logger.handlers:
        try:
            handler = logging.FileHandler("crawl.log", encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        except OSError:
            pass

    queue: asyncio.Queue[str] = asyncio.Queue()
    results_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()
    seen: set[str] = set()
    global_sem = asyncio.Semaphore(concurrency)
    host_sems: dict[str, asyncio.Semaphore] = {}
    
    headers = {"User-Agent": "FindexCrawler/1.0 (contact: greycat@example.com)"}
    
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        robots = RobotsCache(client, headers["User-Agent"])
        
        for url in start_urls:
            canon = canonicalize_url(url)
            seen.add(canon)
            queue.put_nowait(canon)
            
        stats["queue_depth"] = queue.qsize()

        async with asyncio.TaskGroup() as tg:
            workers = [tg.create_task(crawl_worker(queue, client, global_sem, host_sems, robots, results_queue, seen, stats)) for _ in range(concurrency)]
            
            pages_crawled = 0
            while pages_crawled < max_pages:
                try:
                    async with asyncio.timeout(1.0):
                        doc = await results_queue.get()
                        yield doc
                        pages_crawled += 1
                except TimeoutError:
                   if queue.empty() and stats.get("active_tasks", 0) == 0:
                        break
            
            for w in workers:
                w.cancel()