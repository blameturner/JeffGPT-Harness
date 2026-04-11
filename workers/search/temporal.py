from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from config import CHAT_TIMEZONE

_log = logging.getLogger("web_search.temporal")


def now_in_chat_tz() -> datetime:
    try:
        return datetime.now(ZoneInfo(CHAT_TIMEZONE))
    except Exception:
        _log.warning("invalid CHAT_TIMEZONE=%r, falling back to UTC", CHAT_TIMEZONE)
        from datetime import timezone
        return datetime.now(timezone.utc)


def build_temporal_context(now: datetime | None = None) -> str:
    now = now or now_in_chat_tz()
    human = now.strftime("%A, %d %B %Y, %H:%M %Z").strip()
    return (
        f"Current date and time: {human}.\n"
        f"ISO: {now.isoformat()}.\n"
        "When the user says 'today', 'this week', 'this season', 'recent', "
        "or 'latest', resolve it relative to this date. Do NOT claim "
        "something is in the future if the date above shows it is in the "
        "past or present."
    )


def build_prompt_date_header(now: datetime | None = None) -> str:
    now = now or now_in_chat_tz()
    return f"Today is {now.strftime('%A, %-d %B %Y')} ({now.strftime('%Y-%m-%d')})."