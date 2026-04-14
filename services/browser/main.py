from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from playwright.async_api import Browser, async_playwright

_log = logging.getLogger("browser")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class FetchRequest(BaseModel):
    url: str
    timeout: int = Field(default=20, ge=1, le=60)
    wait_until: str = Field(default="networkidle")


class FetchResponse(BaseModel):
    ok: bool
    url: str
    html: str = ""
    status: int = 0
    error: str | None = None


_browser: Browser | None = None
_browser_lock = asyncio.Lock()
_playwright = None


async def _get_browser() -> Browser:
    global _browser, _playwright
    async with _browser_lock:
        if _browser is not None and _browser.is_connected():
            return _browser
        _log.info("launching chromium")
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        return _browser


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global _browser, _playwright
    if _browser is not None:
        try:
            await _browser.close()
        except Exception:
            pass
    if _playwright is not None:
        try:
            await _playwright.stop()
        except Exception:
            pass


app = FastAPI(title="mst-ag-browser", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "service": "browser"}


@app.post("/fetch", response_model=FetchResponse)
async def fetch(req: FetchRequest) -> FetchResponse:
    try:
        browser = await _get_browser()
    except Exception as e:
        _log.error("browser launch failed: %s", e, exc_info=True)
        return FetchResponse(ok=False, url=req.url, error=f"browser launch failed: {e}")

    context = None
    page = None
    try:
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="en-AU",
        )
        page = await context.new_page()
        response = await page.goto(
            req.url,
            wait_until=req.wait_until if req.wait_until in
                ("load", "domcontentloaded", "networkidle", "commit") else "networkidle",
            timeout=req.timeout * 1000,
        )
        status = response.status if response else 0
        html = await page.content()
        return FetchResponse(ok=True, url=req.url, html=html, status=status)
    except Exception as e:
        _log.warning("fetch failed url=%s: %s", req.url, e)
        return FetchResponse(ok=False, url=req.url, error=str(e))
    finally:
        try:
            if page is not None:
                await page.close()
        except Exception:
            pass
        try:
            if context is not None:
                await context.close()
        except Exception:
            pass
