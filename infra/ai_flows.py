"""Single-shot AI flows for project workspaces. Each flow does ONE bounded
model call via `shared.models.model_call(function_name, prompt)` — function
configs live in `config.json` under `features.code_v2.models`.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from infra.config import get_function_config
from shared.models import model_call

_log = logging.getLogger("ai_flows")


def _truncate(text: str, function_name: str) -> str:
    try:
        cap = int(get_function_config(function_name).get("max_input_chars") or 8000)
    except KeyError:
        cap = 8000
    if len(text) <= cap:
        return text
    head = text[: cap // 2]
    tail = text[-cap // 2 :]
    return head + "\n\n…[truncated]…\n\n" + tail


def _extract_json(raw: str) -> Any:
    """Best-effort JSON extraction from a possibly-fenced model response."""
    if not raw:
        return None
    s = raw.strip()
    # Strip ``` / ```json fences
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", s, re.DOTALL)
    if fence:
        s = fence.group(1)
    # Find first { ... last }
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        s = s[start : end + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


# -------- 1. AI code review on diff --------
_CODE_REVIEW_PROMPT = """You are reviewing a code diff. Return JSON only, no prose:

{
  "summary": "<2-3 sentence overview>",
  "concerns": [
    {"path": "/src/x.py", "line": 42, "severity": "info|warning|error|security",
     "comment": "<specific, actionable>"}
  ],
  "suggested_followups": ["<short bullet>", "..."]
}

DIFF:
%s
"""


def review_diff(unified_diff: str) -> dict:
    prompt = _CODE_REVIEW_PROMPT % _truncate(unified_diff, "code_review")
    raw, tokens = model_call("code_review", prompt)
    parsed = _extract_json(raw) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    return {
        "summary": parsed.get("summary") or "",
        "concerns": parsed.get("concerns") or [],
        "suggested_followups": parsed.get("suggested_followups") or [],
        "tokens": tokens,
        "raw": raw if not parsed else "",
    }


# -------- 2. File summary --------
_FILE_SUMMARY_PROMPT = """Summarise this file in ONE paragraph (≤80 words).
Cover: purpose, public API surface, notable dependencies. No code. No headers.

PATH: %s

CONTENT:
%s
"""


def summarise_file(path: str, content: str) -> tuple[str, int]:
    prompt = _FILE_SUMMARY_PROMPT % (path, _truncate(content, "file_summary"))
    raw, tokens = model_call("file_summary", prompt)
    summary = (raw or "").strip().strip("`").strip()
    # collapse multi-paragraph noise to a single paragraph
    summary = re.sub(r"\s*\n+\s*", " ", summary)[:600]
    return summary, tokens


# -------- 3. README maintenance --------
_README_PROMPT = """Generate /README.md for this project. Output only the markdown,
no fenced wrapping. Be terse. Sections (omit any that have no signal):
- A one-line description.
- ## What it does (1 paragraph).
- ## Layout (bullet list of top-level dirs/files with one-line each).
- ## Recent decisions (bullet list of ADR titles, latest first).
- ## Open work (one-liner: TODO count, lint issue count if present).

PROJECT: %s
DESCRIPTION: %s
PATH MANIFEST (path, kind, size):
%s
RECENT ADRS (title — decision):
%s
OPEN WORK: %s
"""


def regenerate_readme(name: str, description: str, manifest: list[dict], adrs: list[dict], open_work: dict) -> tuple[str, int]:
    manifest_text = "\n".join(f"- {f.get('path')} ({f.get('kind') or '?'}, {f.get('size_bytes') or 0}B)" for f in manifest[:80])
    adr_text = "\n".join(f"- {a.get('title')} — {a.get('decision')}" for a in adrs[:10]) or "(none)"
    open_text = f"{open_work.get('open_todos', 0)} TODOs, {open_work.get('issue_count', 0)} open issues"
    prompt = _README_PROMPT % (name, description or "", _truncate(manifest_text, "readme_maintain"), adr_text, open_text)
    raw, tokens = model_call("readme_maintain", prompt)
    body = (raw or "").strip()
    # Strip a leading fence if model wrapped it
    body = re.sub(r"^```(?:markdown|md)?\s*\n", "", body)
    body = re.sub(r"\n```\s*$", "", body)
    return body, tokens


# -------- 4. FAQ maintenance --------
_FAQ_PROMPT = """You maintain /FAQ.md. Given the existing FAQ and a NEW Q/A,
return the updated full FAQ. Rules:
- Dedupe near-duplicate questions; keep the most recent answer when contradicted.
- Sort entries by recency (newest first).
- Each entry as `### Q: <question>` then a paragraph answer.
- Output the full markdown body, nothing else.

EXISTING FAQ:
%s

NEW Q: %s
NEW A: %s
"""


def update_faq(existing: str, question: str, answer: str) -> tuple[str, int]:
    prompt = _FAQ_PROMPT % (_truncate(existing or "(empty)", "faq_maintain"), question, _truncate(answer, "faq_maintain"))
    raw, tokens = model_call("faq_maintain", prompt)
    body = (raw or "").strip()
    body = re.sub(r"^```(?:markdown|md)?\s*\n", "", body)
    body = re.sub(r"\n```\s*$", "", body)
    return body, tokens


# -------- 5. Smart paste classification --------
_SMART_PASTE_PROMPT = """Classify the pasted content. Return JSON only:

{
  "kind": "code|doc|note|url|data",
  "language": "<language slug or empty>",
  "suggested_path": "/src/...|/notes/...|/docs/...",
  "reason": "<one sentence>"
}

PASTED:
%s
"""


def classify_paste(text: str) -> dict:
    prompt = _SMART_PASTE_PROMPT % _truncate(text, "smart_paste")
    raw, tokens = model_call("smart_paste", prompt)
    parsed = _extract_json(raw) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    return {
        "kind": parsed.get("kind") or "note",
        "language": parsed.get("language") or "",
        "suggested_path": parsed.get("suggested_path") or "/notes/paste.md",
        "reason": parsed.get("reason") or "",
        "tokens": tokens,
    }


# -------- 6. Playbook generation --------
_PLAYBOOK_PROMPT = """Generate a migration playbook for this goal. Return JSON only:

{
  "goal": "<echoed goal>",
  "steps": [
    {"title": "<short>", "description": "<what changes>", "scope_paths": ["/src/..."], "risk": "low|medium|high"}
  ]
}

Keep steps small (≤1 file's worth of work each). 3-8 steps.

GOAL: %s

PATH MANIFEST:
%s
"""


def generate_playbook(goal: str, manifest: list[dict]) -> dict:
    manifest_text = "\n".join(f"- {f.get('path')}" for f in manifest[:60])
    prompt = _PLAYBOOK_PROMPT % (goal, _truncate(manifest_text, "playbook_generate"))
    raw, tokens = model_call("playbook_generate", prompt)
    parsed = _extract_json(raw) or {}
    if not isinstance(parsed, dict):
        parsed = {"goal": goal, "steps": []}
    parsed.setdefault("goal", goal)
    parsed.setdefault("steps", [])
    parsed["tokens"] = tokens
    return parsed


# -------- 7. Spec-first regeneration --------
_SPEC_REGEN_PROMPT = """Regenerate the target file to align with the spec. Preserve
user-written sections that the spec doesn't dictate. Output only the new file
content, no fences, no commentary.

SPEC PATH: %s
SPEC:
%s

TARGET PATH: %s
CURRENT CONTENT:
%s
"""


def regenerate_from_spec(spec_path: str, spec_content: str, target_path: str, current_content: str) -> tuple[str, int]:
    prompt = _SPEC_REGEN_PROMPT % (
        spec_path, _truncate(spec_content, "spec_regen"),
        target_path, _truncate(current_content or "(new file)", "spec_regen"),
    )
    raw, tokens = model_call("spec_regen", prompt)
    body = (raw or "").strip()
    body = re.sub(r"^```\w*\s*\n", "", body)
    body = re.sub(r"\n```\s*$", "", body)
    return body, tokens
