"""Single source of truth for "fetch a URL and return its content".

Wraps:
  - requests with conditional GET (If-None-Match / If-Modified-Since)
  - per-host rate-limit + cool-off
  - robots.txt with per-host opt-out
  - api_connections-based authentication when connection_id is provided
  - content-type-aware loaders (HTML / PDF / RSS / sitemap / JSON / CSV / YouTube)
  - optional Playwright headless fallback (per-host opt-in)
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

import requests

from tools.harvest import connectors, host_config, loaders, rate_limit, robots

_log = logging.getLogger("harvest.fetcher")

_DEFAULT_TIMEOUT_S = 30
_USER_AGENT = "mst-harness/1.0 (+contact)"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB cap on fetched body


def _cfg(key: str, default):
    """Read a harvest config key from features.harvest with a fallback."""
    try:
        from infra.config import get_feature
        return get_feature("harvest", key, default)
    except Exception:
        return default


def _ua() -> str:
    return str(_cfg("user_agent", _USER_AGENT)) or _USER_AGENT


def _default_timeout() -> int:
    try:
        return int(_cfg("fetch_timeout_s", _DEFAULT_TIMEOUT_S))
    except Exception:
        return _DEFAULT_TIMEOUT_S


def _max_bytes() -> int:
    try:
        return int(_cfg("max_body_bytes", _MAX_BYTES))
    except Exception:
        return _MAX_BYTES


@dataclass
class FetchResult:
    url: str
    final_url: str = ""
    status_code: int = 0
    content_type: str = ""
    text: str = ""                       # text loaders' output (post-conversion)
    raw_text: str = ""                   # original body decoded as text (HTML / RSS / etc.)
    raw_bytes: bytes = b""               # original body bytes (PDF / binary loaders)
    bytes_len: int = 0
    etag: str = ""
    last_modified: str = ""
    content_hash: str = ""
    kind: str = ""                       # html / pdf / rss / sitemap / json / csv / youtube / text
    error: str = ""
    headers: dict = field(default_factory=dict)
    skipped_reason: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300 and bool(self.text or self.raw_text or self.raw_bytes)


def _hash(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:32]


def _looks_js_only(text: str) -> bool:
    if not text or len(text) > 4096:
        return False
    lower = text.lower()
    js_signals = ("enable javascript", "noscript", "you need to enable", "_next/static")
    return any(sig in lower for sig in js_signals)


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _add_query(url: str, extra: dict) -> str:
    if not extra:
        return url
    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q.update({k: str(v) for k, v in extra.items() if v is not None})
    return urlunparse(p._replace(query=urlencode(q)))


def _resolve_host_options(url: str, *, policy_default_rate_s: float | None,
                          policy_respects_robots: bool,
                          headless_fallback: bool,
                          connection_id: int | None) -> dict:
    """Merge per-host overrides into the call options."""
    host_cfg = host_config.get(_host(url))
    return {
        "rate_s": (
            host_cfg.get("rate_limit_per_host_s")
            if host_cfg.get("rate_limit_per_host_s") is not None
            else policy_default_rate_s
        ),
        "respects_robots": (
            bool(host_cfg["respect_robots"])
            if "respect_robots" in host_cfg
            else policy_respects_robots
        ),
        "headless": (
            bool(host_cfg["headless_fallback"])
            if "headless_fallback" in host_cfg
            else headless_fallback
        ),
        "connection_id": (
            int(host_cfg["connection_id"])
            if isinstance(host_cfg.get("connection_id"), int)
            else connection_id
        ),
    }


def fetch(url: str, *,
          timeout_s: int | None = None,
          if_none_match: Optional[str] = None,
          if_modified_since: Optional[str] = None,
          extra_headers: Optional[dict] = None,
          policy_default_rate_s: float | None = None,
          policy_respects_robots: bool = True,
          headless_fallback: bool = False,
          connection_id: int | None = None,
          org_id: int | None = None,
          method: str = "GET") -> FetchResult:
    """Fetch a URL with rate-limit, robots check, conditional GET, optional
    auth, and content-type loader dispatch.

    Returns a FetchResult. ``skipped_reason`` is set if the fetch was
    deliberately skipped (rate-limited, robots-disallowed, cool-off,
    or a 304 Not Modified).
    """
    if timeout_s is None:
        timeout_s = _default_timeout()
    fr = FetchResult(url=url)
    opts = _resolve_host_options(
        url,
        policy_default_rate_s=policy_default_rate_s,
        policy_respects_robots=policy_respects_robots,
        headless_fallback=headless_fallback,
        connection_id=connection_id,
    )

    # YouTube: special-case before the HTTP path — we want the transcript.
    # Still apply per-host rate limiting so a channel-wide harvest doesn't
    # hammer the transcript service.
    if loaders.detect_kind(url, "", "") == "youtube":
        if not rate_limit.acquire(url, policy_default_rate_s=opts["rate_s"]):
            fr.skipped_reason = "rate_limited_or_cool_off"
            return fr
        fr.kind = "youtube"
        fr.final_url = url
        text = loaders.load_youtube_transcript(url)
        if text:
            fr.text = text
            fr.raw_text = text
            fr.bytes_len = len(text)
            fr.status_code = 200
            fr.content_type = "text/plain"
            fr.content_hash = _hash(text)
            rate_limit.record_success(url)
        else:
            fr.error = "youtube_transcript_unavailable"
            rate_limit.record_failure(url)
        return fr

    # Robots
    if not robots.can_fetch(url, policy_respects=opts["respects_robots"], user_agent=_USER_AGENT):
        fr.skipped_reason = "robots_disallow"
        return fr

    # Rate limit / cool-off
    if not rate_limit.acquire(url, policy_default_rate_s=opts["rate_s"]):
        fr.skipped_reason = "rate_limited_or_cool_off"
        return fr

    # Auth + base headers
    headers: dict[str, str] = {"User-Agent": _ua(), "Accept": "*/*"}
    auth_query: dict = {}
    if opts["connection_id"]:
        ah, aq, dh, _base = connectors.resolve(opts["connection_id"], org_id=org_id)
        headers.update(dh or {})
        headers.update(ah or {})
        auth_query.update(aq or {})

    if if_none_match:
        headers["If-None-Match"] = if_none_match
    if if_modified_since:
        headers["If-Modified-Since"] = if_modified_since
    if extra_headers:
        headers.update(extra_headers)

    target_url = _add_query(url, auth_query)

    try:
        if method.upper() == "HEAD":
            resp = requests.head(target_url, headers=headers, timeout=timeout_s, allow_redirects=True)
        else:
            resp = requests.get(
                target_url, headers=headers, timeout=timeout_s, allow_redirects=True,
                stream=True,
            )
    except requests.RequestException as e:
        rate_limit.record_failure(url)
        fr.error = f"request_error: {type(e).__name__}: {str(e)[:200]}"
        return fr

    fr.status_code = resp.status_code
    fr.final_url = str(resp.url) or url
    fr.content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    fr.etag = resp.headers.get("ETag") or ""
    fr.last_modified = resp.headers.get("Last-Modified") or ""
    fr.headers = dict(resp.headers)

    if resp.status_code == 304:
        fr.skipped_reason = "not_modified"
        rate_limit.record_success(url)
        return fr

    if method.upper() == "HEAD":
        if 200 <= resp.status_code < 400:
            rate_limit.record_success(url)
        else:
            rate_limit.record_failure(url)
        return fr

    if resp.status_code >= 400:
        rate_limit.record_failure(url)
        fr.error = f"http_{resp.status_code}"
        if resp.status_code == 403 and opts["headless"]:
            return _try_headless(fr, timeout_s)
        return fr

    # Read body with size cap
    try:
        chunks: list[bytes] = []
        total = 0
        max_bytes = _max_bytes()
        for chunk in resp.iter_content(chunk_size=8192, decode_unicode=False):
            if chunk:
                chunks.append(chunk)
                total += len(chunk)
                if total >= max_bytes:
                    fr.error = "body_too_large"
                    break
        body = b"".join(chunks)
        fr.bytes_len = len(body)
        fr.raw_bytes = body
        # Best-effort decode for text-shaped content
        encoding = resp.encoding or "utf-8"
        try:
            fr.raw_text = body.decode(encoding, errors="replace")
        except (LookupError, TypeError):
            fr.raw_text = body.decode("utf-8", errors="replace")
    except Exception as e:
        rate_limit.record_failure(url)
        fr.error = f"read_error: {type(e).__name__}: {str(e)[:200]}"
        return fr

    # JS-only fallback (HTML only)
    if opts["headless"] and "html" in (fr.content_type or "") and _looks_js_only(fr.raw_text):
        result = _try_headless(fr, timeout_s)
        if result.ok:
            return _finalise(result, url)
        # fall through to use the raw HTML we already have

    return _finalise(fr, url)


def _finalise(fr: FetchResult, requested_url: str) -> FetchResult:
    """Run loader dispatch + record success."""
    fr.kind = loaders.detect_kind(
        fr.final_url or requested_url,
        fr.content_type,
        fr.raw_text[:512] if fr.raw_text else "",
    )
    # For PDFs the bytes path is preferred; for everything else use raw_text.
    fr.text = loaders.to_text(
        kind=fr.kind,
        url=fr.final_url or requested_url,
        body_text=fr.raw_text,
        body_bytes=fr.raw_bytes if fr.kind == "pdf" else None,
    )
    if not fr.text and fr.raw_text:
        fr.text = fr.raw_text  # fall back so HTML pages still work
    fr.content_hash = _hash(fr.text)
    rate_limit.record_success(requested_url)
    return fr


def _try_headless(fr: FetchResult, timeout_s: int) -> FetchResult:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        fr.error = (fr.error or "") + " (headless_unavailable: playwright not installed)"
        return fr

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                ctx = browser.new_context(user_agent=_ua())
                page = ctx.new_page()
                page.goto(fr.url, timeout=timeout_s * 1000, wait_until="domcontentloaded")
                fr.raw_text = page.content()
                fr.bytes_len = len(fr.raw_text.encode("utf-8", errors="ignore"))
                fr.status_code = 200
                fr.content_type = fr.content_type or "text/html"
                fr.error = ""
            finally:
                browser.close()
    except Exception as e:
        fr.error = (fr.error or "") + f" (headless_error: {type(e).__name__}: {str(e)[:120]})"
    return fr


def head(url: str, **kwargs) -> FetchResult:
    """Convenience: HEAD request for broken-link sweep etc."""
    return fetch(url, method="HEAD", **kwargs)
