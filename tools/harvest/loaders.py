"""Content-type-aware loaders. Convert any fetched body into plain text.

All loaders are best-effort and soft-import their heavy dependencies so
the harness still runs if pdfminer / youtube-transcript-api / feedparser
aren't installed — those formats just become no-ops with a warning.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
from typing import Iterable

_log = logging.getLogger("harvest.loaders")

_YOUTUBE_HOSTS = ("youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com")
_YOUTUBE_ID_RE = re.compile(r"(?:v=|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})")


def _cfg_int(key: str, default: int) -> int:
    try:
        from infra.config import get_feature
        return int(get_feature("harvest", key, default) or default)
    except Exception:
        return default


def _collate_pdf_pages(
    pages: Iterable[str],
    *,
    max_pages: int,
    max_chars: int,
) -> tuple[str, bool, int]:
    parts: list[str] = []
    chars = 0
    consumed = 0
    truncated = False
    for i, page_text in enumerate(pages):
        if i >= max_pages or chars >= max_chars:
            truncated = True
            break
        text = page_text or ""
        remaining = max_chars - chars
        if len(text) > remaining:
            text = text[:remaining]
            truncated = True
        parts.append(text)
        chars += len(text)
        consumed += 1
    return "\n\n".join(parts).strip(), truncated, consumed


def detect_kind(url: str, content_type: str = "", body_head: str = "") -> str:
    """Return one of: html, pdf, rss, sitemap, json, csv, youtube, text."""
    u = (url or "").lower()
    ct = (content_type or "").lower()
    head = (body_head or "")[:512].lstrip().lower()

    # YouTube URLs are detected purely by URL — content-type is HTML
    if any(h in u for h in _YOUTUBE_HOSTS):
        return "youtube"
    if "application/pdf" in ct or u.endswith(".pdf"):
        return "pdf"
    if "rss" in ct or "atom" in ct or head.startswith("<rss") or head.startswith("<feed"):
        return "rss"
    if "xml" in ct and ("urlset" in head or "sitemapindex" in head or u.endswith(".xml") and "sitemap" in u):
        return "sitemap"
    if "json" in ct or u.endswith(".json"):
        return "json"
    if "csv" in ct or u.endswith(".csv"):
        return "csv"
    if "html" in ct or head.startswith("<!doctype html") or head.startswith("<html"):
        return "html"
    if not ct and not head:
        return "html"
    return "text"


# ── individual loaders ────────────────────────────────────────────────────

def load_pdf(body_bytes: bytes) -> str:
    """PDF → plain text.

    Extract incrementally page-by-page so long PDFs don't monopolize the
    process for one monolithic parse. Between pages, cooperative background
    tasks yield to active chat turns.
    """
    if not body_bytes:
        return ""
    max_pages = max(1, _cfg_int("pdf_max_pages", 20))
    max_chars = max(1000, _cfg_int("pdf_max_chars", 20000))

    def _yield_checkpoint() -> None:
        try:
            from workers.tool_queue import yield_to_chat
            yield_to_chat(max_wait_s=30.0, poll_s=0.5)
        except Exception:
            return

    try:
        import pdfplumber  # type: ignore
    except ImportError:
        pdfplumber = None
    try:
        if pdfplumber is not None:
            page_texts: list[str] = []
            with pdfplumber.open(io.BytesIO(body_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    if i >= max_pages:
                        break
                    page_texts.append(page.extract_text() or "")
                    _yield_checkpoint()
            text, truncated, consumed = _collate_pdf_pages(page_texts, max_pages=max_pages, max_chars=max_chars)
            _log.info(
                "pdf extracted incrementally pages=%d chars=%d truncated=%s",
                consumed,
                len(text),
                truncated,
            )
            return text
    except Exception as e:
        _log.warning("pdfplumber parse failed: %s", e)

    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except ImportError:
        _log.warning("pdfplumber/pdfminer not installed — PDF loader is a no-op")
        return ""
    try:
        text = extract_text(io.BytesIO(body_bytes)) or ""
        if len(text) > max_chars:
            text = text[:max_chars]
        return text
    except Exception as e:
        _log.warning("pdf parse failed: %s", e)
        return ""


def load_rss(text: str) -> str:
    """RSS/Atom → flattened text of all entries (title + summary)."""
    if not text:
        return ""
    try:
        import feedparser  # type: ignore
    except ImportError:
        return text  # caller will treat as plain text
    parsed = feedparser.parse(text)
    chunks: list[str] = []
    for e in (parsed.entries or [])[:200]:
        title = getattr(e, "title", "") or ""
        summary = getattr(e, "summary", "") or ""
        link = getattr(e, "link", "") or ""
        published = getattr(e, "published", "") or ""
        chunks.append(f"## {title}\n{published}\n{link}\n{summary}".strip())
    return "\n\n".join(chunks)


def load_json(body_bytes_or_text) -> str:
    """JSON → pretty-printed text. Truncated at 64KB."""
    if not body_bytes_or_text:
        return ""
    if isinstance(body_bytes_or_text, bytes):
        text = body_bytes_or_text.decode("utf-8", errors="replace")
    else:
        text = body_bytes_or_text
    try:
        obj = json.loads(text)
        pretty = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
        return pretty[:65536]
    except (json.JSONDecodeError, TypeError):
        return text[:65536]


def load_csv(body_text: str, *, max_rows: int = 200) -> str:
    """CSV → markdown-flavoured text representation (header + first N rows).

    For large CSVs the LLM doesn't need every row — head is sufficient.
    """
    if not body_text:
        return ""
    out_lines: list[str] = []
    try:
        reader = csv.reader(io.StringIO(body_text))
        rows = list(reader)
    except Exception as e:
        _log.warning("csv parse failed: %s", e)
        return body_text[:65536]
    if not rows:
        return ""
    header = rows[0]
    body = rows[1: 1 + max_rows]
    out_lines.append("| " + " | ".join(header) + " |")
    out_lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in body:
        out_lines.append("| " + " | ".join(r) + " |")
    if len(rows) > max_rows + 1:
        out_lines.append(f"\n_(truncated — {len(rows) - 1 - max_rows} more rows)_")
    return "\n".join(out_lines)


def load_youtube_transcript(url: str) -> str:
    """YouTube URL → transcript text. Soft-imports youtube-transcript-api.

    Note: the URL is fetched separately by this loader (the standard fetcher
    only gets the page HTML for a YouTube URL).
    """
    m = _YOUTUBE_ID_RE.search(url or "")
    if not m:
        return ""
    video_id = m.group(1)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    except ImportError:
        _log.warning("youtube-transcript-api not installed — YouTube loader is a no-op")
        return ""
    try:
        # Try common languages; the API returns a list of {text, start, duration}
        for langs in (["en"], ["en-US"], None):
            try:
                if langs:
                    parts = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
                else:
                    parts = YouTubeTranscriptApi.get_transcript(video_id)
                return "\n".join((p.get("text") or "").strip() for p in parts if p.get("text"))
            except Exception:
                continue
    except Exception as e:
        _log.warning("youtube transcript fetch failed: %s", e)
    return ""


# ── dispatch ──────────────────────────────────────────────────────────────

def to_text(*, kind: str, url: str, body_text: str = "",
            body_bytes: bytes | None = None) -> str:
    """Single dispatch entry. Returns plain text suitable for the
    extractor / summariser."""
    if kind == "pdf" and body_bytes:
        return load_pdf(body_bytes)
    if kind == "rss":
        return load_rss(body_text)
    if kind == "sitemap":
        return body_text  # caller handles via walker.extract_sitemap_urls
    if kind == "json":
        return load_json(body_bytes if body_bytes is not None else body_text)
    if kind == "csv":
        return load_csv(body_text)
    if kind == "youtube":
        return load_youtube_transcript(url)
    # html / text — return as-is
    return body_text
