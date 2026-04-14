from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

_log = logging.getLogger("web_search.urls")


SCRAPE_BLOCKLIST = {
    "reddit.com",
    "news.com.au",
    "medium.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "nytimes.com",
    "wsj.com",
    "ft.com",
    "goodreads.com",
}


BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
BROWSER_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

_INJECTION_PATTERNS = [
    re.compile(r"<\s*/?\s*(?:system|assistant|user|\|im_start\||\|im_end\|)[^>]*>", re.I),
    re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.I),
    re.compile(r"(?:^|\n)\s*#{2,}\s*(?:system|instruction|human|user|assistant|ai)\s*[:.]", re.I),
    re.compile(
        r"(?:^|\n)\s*(?:human|user)\s*:\s.{0,2000}?(?:^|\n)\s*(?:assistant|ai|gpt|chatbot)\s*:\s?",
        re.I | re.DOTALL | re.MULTILINE,
    ),
    re.compile(
        r"\b(?:ignore|disregard|forget|erase) (?:all |the |your )?(?:previous|prior|above|earlier|original|preceding) "
        r"(?:instructions?|messages?|context|prompts?|rules?|system)",
        re.I,
    ),
    re.compile(r"\bnew (?:instructions?|rules?|persona|role|system prompt)\s*:", re.I),
    re.compile(
        r"\b(?:system|admin|root|developer) (?:prompt|message|override|mode|instructions?|directives?)\s*:",
        re.I,
    ),
    re.compile(
        r"\b(?:override|bypass|skip|disable|turn off|switch off) "
        r"(?:all |your |the )?(?:previous |prior |safety |content )?"
        r"(?:instructions?|rules?|guidelines?|filters?|safety|restrictions?)",
        re.I,
    ),
    re.compile(
        r"\b(?:do not|don'?t|stop) (?:follow|obey|listen to|respect) "
        r"(?:your |the )?(?:previous|original|system|prior) "
        r"(?:instructions?|prompt|rules?|guidelines?)",
        re.I,
    ),
    re.compile(
        r"\b(?:reveal|show|print|output|repeat|leak|disclose) "
        r"(?:your |the )?(?:system ?prompt|system instructions?|hidden prompt|initial instructions?)",
        re.I,
    ),
    re.compile(
        r"\b(?:from now on|henceforth|going forward|starting now|"
        r"from this (?:point|moment)|in this (?:conversation|chat|session|task))"
        r"[\s,]*"
        r"(?:you (?:are|will be|will now|must now|shall|have become)|"
        r"act as|pretend (?:you are|to be)|roleplay as|become)",
        re.I,
    ),
    re.compile(
        r"\b(?:jailbreak|dan mode|developer mode|god mode|admin mode|"
        r"unrestricted mode|sudo mode|do anything now)\b",
        re.I,
    ),
]


def _strip_injection_patterns(text: str) -> str:
    for pat in _INJECTION_PATTERNS:
        text = pat.sub("[redacted]", text)
    return text


def _is_blocklisted(url: str) -> bool:
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return False
    if host.startswith("www."):
        host = host[4:]
    return any(host == d or host.endswith("." + d) for d in SCRAPE_BLOCKLIST)


def _is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"):
        return False
    if host.startswith("10.") or host.startswith("192.168.") or host.startswith("169.254."):
        return False
    if host.startswith("172."):
        # RFC 1918 private range is 172.16.0.0 – 172.31.255.255 only
        try:
            second_octet = int(host.split(".")[1])
            if 16 <= second_octet <= 31:
                return False
        except (IndexError, ValueError):
            return False
    if host.endswith(".local") or host.endswith(".internal"):
        return False
    return True


def _sanitise_url(url: str) -> str:
    parsed = urlparse(url)
    clean = parsed._replace(fragment="")
    return clean.geturl()
