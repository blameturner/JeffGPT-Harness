"""Out-of-app push surfacing for high-priority home items.

Hooked by the insight producer and (later) the question queue so that
important outputs don't wait for the user to open the dashboard.

V1 supports SMTP only (using ``external.smtp_server.send_email``). Slack
and webhook deliveries can be added without changing the call-site.

Every call is best-effort — a failed push must never break the caller.
"""
from __future__ import annotations

import logging
import os

from infra.config import get_feature

_log = logging.getLogger("home.surfacing")


def _cfg(key: str, default):
    return get_feature("surfacing", key, default)


def _enabled() -> bool:
    return bool(_cfg("enabled", False))


def push_insight(org_id: int, title: str, summary: str, insight_id: int | None) -> None:
    if not _enabled():
        _log.debug("surfacing disabled; insight %s not pushed", insight_id)
        return
    to_addr = os.getenv("SURFACING_EMAIL_TO") or _cfg("email_to", "")
    if not to_addr:
        _log.debug("surfacing: no email_to configured")
        return
    try:
        from external.smtp_server import send_email
        send_email(
            to=to_addr,
            subject=f"[JeffGPT] New briefing: {title[:80]}",
            body=f"{summary}\n\nOpen the dashboard to read the full briefing (insight #{insight_id}).",
            from_addr=os.getenv("SURFACING_EMAIL_FROM", "jeffgpt@localhost"),
            smtp_host=os.getenv("SMTP_HOST", "smtp.example.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            username=os.getenv("SMTP_USERNAME"),
            password=os.getenv("SMTP_PASSWORD"),
            use_tls=True,
        )
        _log.info("surfacing: insight %s pushed via email to %s", insight_id, to_addr)
    except Exception:
        _log.warning("surfacing: insight %s email push failed", insight_id, exc_info=True)


def push_question(org_id: int, question_text: str, question_id: int) -> None:
    """Placeholder — kept symmetric with ``push_insight`` so question producers
    can opt in later without changing the call site."""
    if not _enabled():
        return
    _log.debug("surfacing: question %s push noop (not implemented yet)", question_id)
