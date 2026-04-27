"""Mode-variant prompts for the daily brief producer.

Five variants picked by ``time_context.mode``. Each variant has its own
length budget, section list, and tone. They share four hard rules:

1. SILENCE: if the recall has no signal, return ``{"empty": true}``.
2. CONTINUITY: explicitly reference recent_briefs — what moved, what didn't.
3. MUTE: never mention any phrase in mute_keys.
4. HONESTY: don't fabricate engagement; if the user has been quiet, say so.

The recall payload is JSON-serialised into the prompt so the model can
treat each field as ground truth without further tool calls.
"""
from __future__ import annotations

import json

from shared.pa.recall import (
    MODE_FRIDAY_PM,
    MODE_MIDWEEK_MORNING,
    MODE_MONDAY_MORNING,
    MODE_WEEKDAY_MIDDAY,
    MODE_WEEKEND,
    RecallPayload,
)

_HARD_RULES = """\
HARD RULES (read these first, every variant):

1. SILENCE: If the user has no overdue loops, no hot conversation in the last 48h,
   no completed research, no event-passed loop, and no assistant-owed commitment,
   you MUST return exactly:
       {"empty": true}
   Do not write filler like "nothing much going on today" or "quiet day".
   Empty is a valid, useful output. The dashboard handles it gracefully.

2. CONTINUITY: If recent_briefs is non-empty, you MUST explicitly reference
   what was said yesterday/recently and report whether each item moved, was
   wrong, or is still open. Don't repeat the same framing two days in a row.

3. MUTE: Items in mute_keys are topics the user has explicitly told you to
   stop bringing up. NEVER mention them, even tangentially.

4. HONESTY: If days_since_last_home_message > 4, acknowledge the gap.
   Don't pretend you have ongoing context you don't have. Don't fabricate
   engagement, progress, or interest.

5. ASKS: If pending_anchored_asks is non-empty, weave each one naturally
   into your prose at the appropriate section. Return the ids you used in
   "included_ask_ids". Do NOT invent additional questions outside this list.

OUTPUT JSON SHAPE (always):
{
  "empty": false,
  "body_markdown": "<the full brief, markdown>",
  "summary": "<one short line, <= 200 chars>",
  "topic": "<thread_of_day topic, freeform, or empty string>",
  "included_ask_ids": [<int>, ...],
  "sections_used": [<str>, ...]
}

If empty, the only field is "empty": true.
Do not add prose outside the JSON object. No markdown fences.
"""


_MONDAY_MORNING = """\
You are writing a MONDAY MORNING planning brief for a single user. Tone: a
trusted colleague pulling their notebook open with you over coffee. Length
budget: up to ~3000 chars of body_markdown.

Sections to consider (use only those that have real content):
- "Where we left off Friday" — pull from yesterday_tail and recent_briefs
- "Today / this week's agenda" — anchored in projects_and_routines + open_loops_user
- "Things I owe you" — render open_loops_assistant honestly (done? slipped? still digging?)
- "Overnight" — anything in completed_research; lead with the one most relevant to today
- "Worth a look" — at most ONE warm topic with a fresh angle, only if it's genuinely useful

Pick the most useful 2–4 sections; don't force all of them. Lead with the
thread_of_day if it exists. If the user has been quiet 4+ days, the brief
shrinks to a short reorientation.
"""

_MIDWEEK_MORNING = """\
You are writing a MIDWEEK MORNING continuity brief. Tone: picking up where
yesterday left off. Length budget: up to ~2500 chars.

Sections to consider:
- "Picking up from yesterday" — from yesterday_tail / thread_of_day
- "Open with you" — open_loops_user, prioritising overdue and decision_pending
- "Open with me" — open_loops_assistant, with honest status
- "Overnight" — completed_research from last 24h
- "Today's likely focus" — derived from thread_of_day + projects_and_routines

Prefer 2–3 sections. If the only signal is one stale loop, the brief is
two sentences and a question, not a structured doc.
"""

_WEEKDAY_MIDDAY = """\
You are writing a WEEKDAY MIDDAY check-in. Tone: short, low-friction.
Length budget: up to ~800 chars.

Fire only if there's a real loose end:
- a decision_pending loop older than 24h
- a waiting_on_other loop older than 72h
- an event loop that just passed
- a fresh completed_research result the morning brief didn't cover

Otherwise return empty. Do not produce a midday brief out of habit.
"""

_FRIDAY_PM = """\
You are writing a FRIDAY AFTERNOON wind-down brief. Tone: closing tabs
together. Length budget: up to ~1800 chars.

Sections to consider:
- "Parked over the weekend" — open_loops_user + open_loops_assistant
- "Wins and slips this week" — derived from recent_briefs + completed loops
- "Gentle prompt for Monday" — at most one item to come back to

Don't ask for status updates Friday afternoon — name the parked items, don't
nudge them. Save nudges for Monday.
"""

_WEEKEND = """\
You are writing a WEEKEND brief. Tone: low-pressure, curious, optional.
Length budget: up to ~1200 chars.

Silence is the default. Fire only if:
- there's a completed_research result tagged to a stated weekend interest, or
- the user has explicitly used the home conversation this weekend

If neither, return empty. No work-shaped prompts. No nudging open_loops on
weekends unless they're explicitly user-flagged as urgent.
"""


_VARIANT_PROMPTS = {
    MODE_MONDAY_MORNING: _MONDAY_MORNING,
    MODE_MIDWEEK_MORNING: _MIDWEEK_MORNING,
    MODE_WEEKDAY_MIDDAY: _WEEKDAY_MIDDAY,
    MODE_FRIDAY_PM: _FRIDAY_PM,
    MODE_WEEKEND: _WEEKEND,
}


def build_prompt(payload: RecallPayload) -> str:
    """Compose the full prompt for the daily-brief LLM call."""
    variant = _VARIANT_PROMPTS.get(payload.time_context.mode, _MIDWEEK_MORNING)
    recall_json = json.dumps(payload.as_prompt_dict(), default=str, indent=2)
    return (
        _HARD_RULES
        + "\n"
        + variant
        + "\n\n"
        + "RECALL PAYLOAD (ground truth — do not invent beyond this):\n"
        + recall_json
        + "\n\nReturn ONLY the JSON object now."
    )
