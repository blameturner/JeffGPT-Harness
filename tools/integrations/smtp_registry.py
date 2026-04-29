"""SMTP account registry — register + self-test SMTP credentials.

Two entry points:
  - register_smtp(...) : insert a row into `smtp_accounts`
  - test_smtp(id)      : send a self-test email; if IMAP details are present,
                         poll the inbox for the message and confirm receipt.
"""
from __future__ import annotations

import logging
import smtplib
import ssl
import time
import uuid
from email.message import EmailMessage

from infra.nocodb_client import NocodbClient
from tools._org import resolve_org_id

_log = logging.getLogger("integrations.smtp")

TABLE = "smtp_accounts"
SECRETS_TABLE = "secrets"

IMAP_POLL_INTERVAL_S = 3
IMAP_MAX_WAIT_S = 60


def _resolve_secret(client: NocodbClient, org_id: int, name: str) -> str | None:
    if not name or SECRETS_TABLE not in client.tables:
        return None
    rows = client._get_paginated(SECRETS_TABLE, params={
        "where": f"(org_id,eq,{org_id})~and(name,eq,{name})",
        "limit": 1,
    })
    if not rows:
        return None
    return rows[0].get("value") or rows[0].get("value_encrypted")


# ---------- register ----------

def register_smtp(
    name: str,
    host: str,
    port: int,
    username: str,
    password_secret_ref: str,
    from_email: str,
    org_id: int = 1,
    use_tls: bool = True,
    use_starttls: bool = True,
    imap_host: str = "",
    imap_port: int = 993,
    imap_username: str = "",
    imap_password_secret_ref: str = "",
    description: str = "",
) -> dict:
    """Insert a new smtp_accounts row. Returns the inserted row."""
    client = NocodbClient()
    if TABLE not in client.tables:
        raise RuntimeError(f"{TABLE} table missing — see docs/new-tables.md")
    payload = {
        "org_id": resolve_org_id(org_id),
        "name": name.strip(),
        "host": host,
        "port": int(port),
        "username": username,
        "password_secret_ref": password_secret_ref,
        "from_email": from_email,
        "use_tls": bool(use_tls),
        "use_starttls": bool(use_starttls),
        "imap_host": imap_host,
        "imap_port": int(imap_port) if imap_port else 0,
        "imap_username": imap_username or username,
        "imap_password_secret_ref": imap_password_secret_ref or password_secret_ref,
        "description": description,
        "verification_status": "unverified",
    }
    row = client._post(TABLE, payload)
    _log.info("smtp registered  name=%s id=%s", name, row.get("Id"))
    return row


# ---------- test ----------

def _send_test_email(account: dict, password: str, marker: str) -> str:
    """Send the test message. Returns the Message-ID header value."""
    msg = EmailMessage()
    msg["Subject"] = f"SMTP self-test {marker}"
    msg["From"] = account["from_email"]
    msg["To"] = account["from_email"]
    message_id = f"<smtp-test-{marker}@{account.get('host', 'local')}>"
    msg["Message-ID"] = message_id
    msg.set_content(
        f"This is an automated SMTP self-test.\n"
        f"Marker: {marker}\n"
        f"Account: {account.get('name')}\n"
    )

    host = account["host"]
    port = int(account["port"])
    use_tls = bool(account.get("use_tls"))
    use_starttls = bool(account.get("use_starttls"))

    if use_tls and not use_starttls:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=ctx, timeout=30) as s:
            s.login(account["username"], password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.ehlo()
            if use_starttls:
                s.starttls(context=ssl.create_default_context())
                s.ehlo()
            s.login(account["username"], password)
            s.send_message(msg)
    return message_id


def _poll_imap_for_marker(account: dict, password: str, marker: str) -> bool:
    """Return True if a message containing the marker arrives within timeout."""
    import imaplib
    host = account.get("imap_host") or ""
    if not host:
        return False
    port = int(account.get("imap_port") or 993)
    user = account.get("imap_username") or account["username"]

    deadline = time.time() + IMAP_MAX_WAIT_S
    while time.time() < deadline:
        try:
            with imaplib.IMAP4_SSL(host, port, timeout=15) as imap:
                imap.login(user, password)
                imap.select("INBOX")
                typ, data = imap.search(None, f'(SUBJECT "{marker}")')
                if typ == "OK" and data and data[0]:
                    return True
        except Exception as e:
            _log.debug("imap poll attempt failed: %s", e)
        time.sleep(IMAP_POLL_INTERVAL_S)
    return False


def test_smtp(account_id: int) -> dict:
    """Send a self-test email and (if IMAP configured) confirm receipt."""
    client = NocodbClient()
    if TABLE not in client.tables:
        raise RuntimeError(f"{TABLE} table missing")
    rows = client._get_paginated(TABLE, params={"where": f"(Id,eq,{account_id})", "limit": 1})
    if not rows:
        raise ValueError(f"smtp account {account_id} not found")
    account = rows[0]
    org_id = resolve_org_id(account.get("org_id"))

    smtp_password = _resolve_secret(client, org_id, account.get("password_secret_ref") or "")
    if not smtp_password:
        return _record_result(client, account_id, "failed", "smtp password secret not found", "")

    marker = uuid.uuid4().hex[:12]
    try:
        message_id = _send_test_email(account, smtp_password, marker)
    except Exception as e:
        return _record_result(client, account_id, "failed", f"send error: {type(e).__name__}: {e}", "")

    received = False
    if account.get("imap_host"):
        imap_secret = account.get("imap_password_secret_ref") or account.get("password_secret_ref")
        imap_password = _resolve_secret(client, org_id, imap_secret or "") or smtp_password
        try:
            received = _poll_imap_for_marker(account, imap_password, marker)
        except Exception as e:
            _log.warning("imap poll failed: %s", e)

    if account.get("imap_host"):
        status = "verified" if received else "send_only"
        note = "send + receipt confirmed" if received else "sent OK; receipt not seen within timeout"
    else:
        status = "send_only"
        note = "sent OK; no IMAP configured for receipt check"

    return _record_result(client, account_id, status, note, message_id)


def _record_result(client: NocodbClient, account_id: int, status: str, note: str, message_id: str) -> dict:
    update = {
        "verification_status": status,
        "verification_note": note,
        "verified_at": _iso_now(),
        "last_test_message_id": message_id,
    }
    try:
        client._patch(TABLE, account_id, update)
    except Exception:
        _log.warning("smtp result write failed", exc_info=True)
    _log.info("smtp test  id=%s status=%s", account_id, status)
    return {"account_id": account_id, "status": status, "note": note, "message_id": message_id}


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
