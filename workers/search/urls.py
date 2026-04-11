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
}


BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
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
    re.compile(r"<\s*/?\s*(?:system|assistant|user|s|\|im_start\||\|im_end\|)[^>]*>", re.I),
    re.compile(r"ignore (?:all )?(?:previous|prior|above|earlier) (?:instructions|messages|context|prompts?)", re.I),
    re.compile(r"disregard (?:all )?(?:prior|previous|above|earlier) (?:instructions|messages|context|prompts?)", re.I),
    re.compile(r"(?:you are|act as|pretend (?:you are|to be)|roleplay as|behave as) (?:a |an )?[a-z ]{3,30}", re.I),
    re.compile(r"new (?:instructions?|rules?|persona|role):", re.I),
    re.compile(r"(?:system|admin|root|developer) (?:prompt|message|override|mode):", re.I),
    re.compile(r"(?:forget|override|bypass|skip) (?:all |your )?(?:previous |prior )?(?:instructions|rules|guidelines|safety)", re.I),
    re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.I),
    re.compile(r"human:|assistant:|###\s*(?:system|instruction|human|assistant)", re.I),
    re.compile(r"(?:do not|don'?t) (?:follow|obey|listen to) (?:your |the )?(?:previous|original|system)", re.I),
    re.compile(r"(?:reveal|show|print|output|repeat) (?:your |the )?(?:system ?prompt|instructions|rules)", re.I),
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
        # RFC 1918: 172.16.0.0 – 172.31.255.255
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
