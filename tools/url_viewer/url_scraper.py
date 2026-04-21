from __future__ import annotations

import asyncio
import logging
import re
import time
from urllib.parse import urlparse

from infra.config import get_feature
from infra.nocodb_client import NocodbClient
from tools.contract import ToolName, ToolResult
from tools.dispatcher import register_executor
from tools.scraper.pathfinder import PathfinderScraper
from tools.search.urls import _is_safe_url
from tools._org import resolve_org_id

_log = logging.getLogger("tools.url_scraper")

_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.I)


def _extract_urls(text: str) -> list[str]:
	seen: set[str] = set()
	out: list[str] = []
	for m in _URL_RE.findall(text or ""):
		url = m.rstrip(".,;:!?)\"]}'")
		if not url or url in seen:
			continue
		seen.add(url)
		out.append(url)
	return out


def _cfg_int(key: str, default: int) -> int:
	raw = get_feature("url_scraper", key, default)
	try:
		val = int(raw)
		return val if val > 0 else default
	except Exception:
		return default


def _derive_target_name(url: str) -> str:
	parts = urlparse(url)
	path = (parts.path or "/").strip("/").replace("/", " - ") or (parts.netloc or url)
	return f"{parts.netloc}: {path}"[:255]


def _scrape_target_exists(client: NocodbClient, url: str, org_id: int) -> bool:
	try:
		data = client._get("scrape_targets", params={
			"where": f"(url,eq,{url})~and(org_id,eq,{org_id})",
			"limit": 1,
		})
		return bool(data.get("list"))
	except Exception:
		return False


def _save_scrape_target(client: NocodbClient, url: str, org_id: int, source_url: str, depth: int) -> bool:
	if not url or not _is_safe_url(url):
		return False
	if _scrape_target_exists(client, url, org_id):
		return False
	payload = {
		"org_id": org_id,
		"url": url,
		"name": _derive_target_name(url),
		"category": "auto",
		"active": 1,
		"frequency_hours": 24,
		"depth": depth,
		"discovered_from": source_url or "",
		"auto_crawled": 1,
		"consecutive_failures": 0,
		"consecutive_unchanged": 0,
		"chunk_count": 0,
	}
	try:
		client._post("scrape_targets", payload)
		return True
	except Exception:
		_log.warning("url_scraper save scrape_target failed  url=%s", url[:120], exc_info=True)
		return False


@register_executor(ToolName.URL_SCRAPER)
async def execute(params: dict, emit) -> ToolResult:
	raw_urls = params.get("urls") or []
	if isinstance(raw_urls, str):
		raw_urls = [raw_urls]
	query_text = str(params.get("query") or "")
	org_id = resolve_org_id(params.get("_org_id") or params.get("org_id"))

	urls: list[str] = []
	for u in raw_urls:
		urls.extend(_extract_urls(str(u)))
	if query_text:
		urls.extend(_extract_urls(query_text))

	deduped: list[str] = []
	seen: set[str] = set()
	for u in urls:
		if u in seen:
			continue
		seen.add(u)
		deduped.append(u)

	max_urls = _cfg_int("max_urls_per_turn", 3)
	max_links_per_url = _cfg_int("max_links_per_url", 30)
	deduped = deduped[:max_urls]
	if not deduped:
		return ToolResult(tool=ToolName.URL_SCRAPER, action_index=0, ok=False, data="No valid URL provided")

	scraper = PathfinderScraper(timeout=_cfg_int("scrape_timeout_s", 60))
	client = NocodbClient() if org_id else None
	t0 = time.time()
	blocks: list[str] = []
	ok_count = 0
	saved_targets = 0

	for url in deduped:
		if not _is_safe_url(url):
			blocks.append(f"SOURCE: {url}\nSkipped unsafe URL")
			continue

		if client and _save_scrape_target(client, url, org_id, "", 0):
			saved_targets += 1

		emit({"type": "tool_status", "phase": "url_scrape", "url": url})
		res = await asyncio.to_thread(scraper.scrape, url)
		if res.get("status") == "ok" and (res.get("text") or "").strip():
			ok_count += 1
			final_url = res.get("canonical") or res.get("final_url") or url
			text = (res.get("text") or "")[:4000]
			if client and final_url and _save_scrape_target(client, final_url, org_id, url, 0):
				saved_targets += 1
			for link in (res.get("links") or [])[:max_links_per_url]:
				if client and _save_scrape_target(client, link, org_id, final_url, 1):
					saved_targets += 1
			blocks.append(f"SOURCE: {final_url}\n{text}")
		else:
			err = (res.get("error") or "scrape_failed")[:120]
			blocks.append(f"SOURCE: {url}\nScrape failed: {err}")

	elapsed = round(time.time() - t0, 2)
	summary = f"\n\nURL scrape summary: sources_ok={ok_count}/{len(deduped)}"
	if client:
		summary += f", scrape_targets_saved={saved_targets}"
	elif deduped:
		summary += ", scrape_targets_saved=0 (missing org_id)"
	return ToolResult(
		tool=ToolName.URL_SCRAPER,
		action_index=0,
		ok=ok_count > 0,
		data="\n\n".join(blocks) + summary,
		elapsed_s=elapsed,
	)
