# Personal Assistant Redesign — Design Spec

Date: 2026-04-27
Branch: enrichment

## Problem

The current home assistant surfaces feel like a chatbot, not a colleague. Symptoms:

- Asks abstract questions about topics the user once searched ("Are you still looking at CuDF?") with no signal of real interest.
- Conflates unrelated topics by chasing graph paths between warm entities.
- Surfaces are short one-line questions; no long-form recall, no briefing, no continuity day-to-day.
- No awareness of weekday vs weekend, morning vs afternoon, or session shape.
- No notion of the assistant's own commitments — only tracks what the user said.
- Research, when it happens, is disconnected from morning surfacing.

Root cause: the proactive layer (`shared/pa/moves.py` + `shared/pa/picker.py`) is **graph-shape-driven**, not **episodic-state-driven**. `serendipity` pulls low-degree concepts; `connect` walks shortest paths between warm topics; both surface things based on graph structure rather than evidence the user cares.

## Principle

> **Every surface must replace work the user would otherwise do, or recall something they would otherwise lose. If neither applies, it stays silent.**

This is enforced inside producer prompts (hard "return empty if nothing useful" rule) and a post-check silence gate.

## Constraint

No DB schema changes. All new functionality reuses the existing tables:

- `assistant_questions`, `insights`, `conversations`, `messages`, `research_plans`
- `pa_open_loops`, `pa_warm_topics`, `pa_user_facts`, `pa_assistant_moves`

## Architecture

Two scheduled producers share one read-only recall layer.

```
                  ┌─────────────────────────┐
                  │  build_recall(org, now) │
                  │  (shared/pa/recall.py)  │
                  └──────────┬──────────────┘
                             │ (pure read; existing tables only)
              ┌──────────────┴───────────────┐
              ▼                              ▼
    ┌───────────────────┐         ┌────────────────────────┐
    │  anchored_asks    │         │     daily_brief        │
    │  (T-5 min)        │ ──────▶ │  (T, scheduled by mode)│
    │  writes:          │  recall │  writes:               │
    │  assistant_       │  reads  │  insights row          │
    │  questions rows   │  asks   │  + home conv message   │
    └───────────────────┘         └────────────────────────┘
```

Two supporting passes:

- `assistant_commitment_extractor` runs in `workers/post_turn.py` after each turn, mines the assistant's reply for first-person commitments, writes `pa_open_loops` rows tagged `source_ref="assistant_commitment:msg_<id>"`.
- `research_seeder` runs ~22:00 nightly, picks 1–2 task-kind warm topics or pending decisions, calls `create_research_plan`. Results land in `insights`/`research_plans` overnight; recall surfaces them next morning.

## The recall layer

`shared/pa/recall.py` exports one function:

```python
def build_recall(org_id: int, now: datetime) -> RecallPayload: ...
```

`RecallPayload` is a frozen dataclass with these fields. Every field is sourced from existing tables.

| Field                  | Source                                               | Filters / shape                                                              |
| ---------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------- |
| `time_context`         | `now` + computed                                     | weekday, part_of_day (`morning`/`midday`/`afternoon`/`evening`), is_weekend, days_since_last_home_message, mode (`monday_morning`/`midweek_morning`/`weekday_midday`/`friday_pm`/`weekend`) |
| `yesterday_tail`       | `messages` joined to `conversations`                 | last 5 user+assistant msgs per conversation touched in last 36h, kind != `home` (home itself is the surface). Capped at 5 conversations by recency. |
| `thread_of_day`        | derived from `yesterday_tail`                        | conversation with the highest `(msgs_24h × 2 + msgs_72h)` score, plus its title and last 8 msgs. Used as the brief's lead anchor. |
| `open_loops_user`      | `pa_open_loops` where `source_ref` not assistant-side | grouped by intent, with `staleness_hours`, `is_overdue` computed from `due_at`/`CreatedAt`. |
| `open_loops_assistant` | `pa_open_loops` where `source_ref` startswith `assistant_commitment:` | same shape; surfaced separately in the brief as "things I owe you". |
| `projects_and_routines`| `pa_user_facts`                                      | `kind in {routine, project, relationship, constraint}` and `confidence != "deleted"`. |
| `warm_topics`          | `pa_warm_topics`                                     | `last_touched_at >= now - 7d` AND warmth ≥ 0.3. Anything older is excluded. |
| `pending_anchored_asks`| `assistant_questions`                                | `status = pending` and created in last 6h (so brief sees just-produced asks). |
| `recent_briefs`        | `insights`                                           | `trigger = "daily_brief"` AND `created_at >= now - 7d`. Title + summary only (not body). |
| `completed_research`   | `research_plans` joined to `insights`                | `status = completed` AND completed in last 24h. |
| `mute_keys`            | `pa_user_facts`                                      | `kind = "preference"` AND `key LIKE "mute:%"` — used to filter all other fields. |
| `engagement_blocks`    | `pa_assistant_moves`                                 | entities or loops with 2+ `engaged=0` in last 14d → filtered out of warm_topics, open_loops, asks. |

The recall layer applies `mute_keys` and `engagement_blocks` as the final pass before returning. Producers consume the filtered payload and never re-apply the filter.

## Producer 1 — `daily_brief`

**Location**: `tools/daily_brief/agent.py`, mirroring `tools/digest/agent.py` and `tools/insight/agent.py` shape.

**Schedule** (in `scheduler.py`):

| Slot                    | Days       | Local time |
| ----------------------- | ---------- | ---------- |
| `monday_morning`        | Mon        | 06:30      |
| `midweek_morning`       | Tue–Thu    | 06:30      |
| `friday_pm`             | Fri        | 16:30      |
| `weekday_midday`        | Mon–Fri    | 14:00      |
| `weekend`               | Sat, Sun   | 09:30      |

**Inputs**: `build_recall(org_id, now)` only.

**Prompt**: variant selected by `time_context.mode`. Each variant has its own length budget, section list, and tone:

- `monday_morning` — long, planning-shaped, asks about the week ahead. Up to ~3000 chars.
- `midweek_morning` — continuity brief: yesterday's tails, today's open loops, anything I owe you, anything I dug into overnight. Up to ~2500 chars.
- `weekday_midday` — short check-in. Only fires if a real loose end exists.
- `friday_pm` — wind-down + weekend handoff. Names what's parked.
- `weekend` — curiosity-mode, low-stakes. Silence is the default; fires only on completed research or a stated weekend interest. Capped at ~1200 chars.

All variants share the same hard rules in the prompt:

1. **Silence rule**: if the recall payload contains no overdue loops AND no hot conversation in last 48h AND no completed research AND no event-passed loop AND no assistant-owed loops, return `{"empty": true}`. Do not write filler.
2. **Continuity rule**: if `recent_briefs` is non-empty, explicitly reference what was said yesterday and report whether it moved, was wrong, or is still open.
3. **Mute rule**: never mention any topic appearing in `mute_keys`.
4. **Honesty rule**: do not fabricate engagement. If the user has been quiet for 4+ days, the brief acknowledges that, asks one open question, and does not pretend to have ongoing context.
5. **Asks rule**: if `pending_anchored_asks` is non-empty, weave them naturally into the brief and return their ids in `included_ask_ids`. Do not invent new questions.

**Output JSON**:
```json
{
  "empty": false,
  "body_markdown": "...",
  "summary": "<= 200 chars",
  "topic": "<thread_of_day topic or freeform>",
  "included_ask_ids": [12, 14],
  "sections_used": ["where_we_left_off", "todays_agenda", "i_owe_you", "overnight"]
}
```

**Persistence** (after silence-gate post-check):

1. Insert `insights` row: `trigger="daily_brief"`, `topic`, `body_markdown`, `summary`, `status="published"`, `surfaced_at=now`, `sources=[{kind: "loop", id: ...}, {kind: "research_plan", id: ...}]`.
2. Insert message into the home conversation (existing `messages` table) with `role="assistant"`, `content=body_markdown`, plus a footer line referencing the insight id.
3. Asks in `included_ask_ids` are left in place. The existing `mark_answered` + `dispatch_followup` plumbing resolves them when the user replies in the home chat.

**Silence-gate post-check** (defence in depth): if the LLM returns non-empty but the recall payload was structurally empty by the silence rule, the producer downgrades to empty and logs a warning. Prevents prompt drift from breaking the principle.

**Model role**: new `model_profiles` entry `daily_brief` — long-context model, `max_tokens=3500`, `temperature=0.4`.

## Producer 2 — `anchored_asks`

**Location**: `tools/anchored_asks/agent.py`.

**Schedule**: 5 minutes before each `daily_brief` slot. Runs deterministically from rules — no LLM required for the rule application; only LLM-backed step is rephrasing (optional, see below).

**Triggers** (all on `pa_open_loops`, all use existing fields):

| Intent              | Trigger condition                                                  | Question shape                                            | followup_action                       |
| ------------------- | ------------------------------------------------------------------ | --------------------------------------------------------- | ------------------------------------- |
| `event`             | `due_at < now` AND no `resolution_note`                            | "How did *<text>* go?"                                    | (none — answer flows back via chat)   |
| `decision_pending`  | age > 24h AND `nudge_count < 2`                                    | "Decision on *<text>* — want me to dig in?"               | `enqueue:research:<text>` (existing)  |
| `waiting_on_other`  | age > 72h AND `nudge_count < 2`                                    | "Still waiting on *<text>*? Any movement?"                | (none)                                |
| `todo`              | `due_at < now` AND `nudge_count < 2`                               | "Did *<text>* get done?"                                  | (none)                                |

**Caps and filters**:

- Max 3 asks created per run.
- Existing `nudge_count` cap of 2 per loop preserved (already enforced by `closure` move's logic; reused).
- Skip any loop whose `text` contains a phrase matching a `mute:*` user_fact key.
- Skip any loop whose entity has appeared in 2+ `pa_assistant_moves` with `engaged=0` in the last 14d.

**Persistence**:

- Each ask becomes one `assistant_questions` row via `queue_question_deduped` with `context_ref=f"loop:{loop_id}:{trigger}"`.
- Ask `suggested_options` follow the patterns already in `shared/pa/extractor.py:_queue_loop_question` (Research / Defer / Done; or Got the answer / Still waiting / Drop it).
- Each ask produces a `pa_assistant_moves` log entry with `move_kind="closure"` (reusing the existing enum value — semantically the same: closing a loop), `mode="proactive"`, `input_refs={"loop_id": ..., "trigger": ...}`. The `closure` *enum value* is preserved; the `closure` *picker function* in `moves.py` is sidelined.

**Coordination with `daily_brief`**:

- `anchored_asks` runs first. The asks are visible in the brief's recall via `pending_anchored_asks`.
- If the brief absorbs an ask (lists its id in `included_ask_ids`), the ask remains queued for answer plumbing — the brief just rendered the question inside its prose.
- If the brief is silenced, the asks remain visible in the dashboard's question queue as a fallback. The dashboard always has *some* path for the user to engage if there's a real outstanding item.

## Supporting pass — assistant commitment extraction

**Location**: `shared/pa/assistant_extractor.py`, called from `workers/post_turn.py` after the existing user-side extractor.

**LLM call**: small JSON-only role `pa_assistant_commitments` (`temperature=0.1`, `max_tokens=200`). Mines the assistant's reply for first-person commitments matching patterns like:

- "I'll check / look into / dig into / pull together / draft …"
- "Let me run / find / verify …"
- "I'll have that for you by …"

**Output**: writes `pa_open_loops` rows with `intent="todo"`, `text=<commitment phrase, ≤ 12 words>`, `source_ref=f"assistant_commitment:msg_{assistant_msg_id}"`, `when_hint=<extracted timeframe or "">`.

**Recall integration**: the recall layer separates loops by `source_ref` prefix into `open_loops_user` and `open_loops_assistant`. The brief renders the latter under "things I owe you" with honest status reporting (done, in progress, slipped, dropped).

## Supporting pass — research seeder

**Location**: `tools/research_seeder/agent.py`. New cron in `scheduler.py` at 22:00 local.

**Selection** (rule-based, no LLM):

1. `pa_warm_topics` with `kind="task"` AND `last_touched_at >= now - 24h` AND no row in `research_plans` with matching `topic` in last 7d.
2. `pa_open_loops` with `intent="decision_pending"` AND created in last 24h AND no existing `research_plan` linked.

Rank by `warmth × engagement_bias(org_id, "news_watch")` (reusing the existing engagement signal). Pick top N (config: `home.research_seeder.max_topics_per_night`, default 2).

**Action**: call `tools.research.research_planner.create_research_plan(topic, org_id=org_id)` — the existing entry point. By morning the plan completes (existing pipeline) and writes an `insights` row. The brief's `completed_research` recall field picks it up.

## Mute mechanism

User says: "stop bringing up CuDF" / "I'm not interested in X anymore."

The existing `pa_extractor` runs on every chat turn and writes `new_facts`. Add to its prompt: when the user expresses disinterest in a topic, write a `kind="preference"` fact with `key="mute:<topic_slug>"` and `value="<reason or empty>"`.

The recall layer reads all `mute:*` facts and applies them as a final filter to `warm_topics`, `open_loops_user`, `open_loops_assistant`, and `pending_anchored_asks` — anything whose text or entity matches (case-insensitive substring) is dropped.

The brief prompt also receives `mute_keys` so it can avoid mentioning muted topics in free-form prose.

## Engagement compounding

Recall's `engagement_blocks` field is computed per call:

```
For each (loop_id or entity) referenced in pa_assistant_moves with engaged=0 in last 14d:
  if count >= 2: add to engagement_blocks
```

Recall filters `warm_topics`, `open_loops_*`, `pending_anchored_asks` by this set. The dashboard's existing engagement-tracking endpoint already writes `engaged=0` when the user dismisses a question; no UI work needed.

## Kill list

Files modified, not deleted, so the picker framework remains for future moves.

| Item                                            | Action                                                                                |
| ----------------------------------------------- | ------------------------------------------------------------------------------------- |
| `shared/pa/picker.py:_MOVES`                    | Empty out. `pick_proactive_move` becomes a no-op (returns `None`). The framework remains for future moves. |
| `shared/pa/moves.py:serendipity`                | Keep code, add `# NOTE: sidelined 2026-04-27` comment, unwire from `_MOVES`.         |
| `shared/pa/moves.py:connect`                    | Same — sidelined.                                                                     |
| `shared/pa/moves.py:news_watch`                 | Same — capability folds into `daily_brief`'s "worth a look" section.                  |
| `shared/pa/moves.py:closure`                    | Sidelined — replaced by deterministic `anchored_asks`.                                |
| `shared/pa/extractor.py:_queue_loop_question`   | **Delete.** Eager loop-question queueing at extraction time is the abstract-question failure mode. `anchored_asks` is the single owner of question creation. |
| `tools/digest/agent.py` question-queueing path  | Remove its `queue_question_deduped` call site. Digest keeps its RAG/digest role.      |
| `tools/insight/agent.py` question-queueing path | Remove its `queue_question_deduped` call site. Insights keep their long-form role.    |

## Configuration (`config.json` only)

```jsonc
{
  "home": {
    "daily_brief": {
      "enabled": true,
      "schedule": {
        "monday_morning": "30 6 * * 1",
        "midweek_morning": "30 6 * * 2-4",
        "friday_pm": "30 16 * * 5",
        "weekday_midday": "0 14 * * 1-5",
        "weekend": "30 9 * * 0,6"
      },
      "max_body_chars": 3500,
      "silence_gate": {
        "require_one_of": ["overdue_loops", "hot_conversation_48h", "completed_research_24h", "event_passed", "assistant_owed_loops"]
      }
    },
    "anchored_asks": {
      "enabled": true,
      "max_per_run": 3,
      "lead_minutes_before_brief": 5
    },
    "research_seeder": {
      "enabled": true,
      "max_topics_per_night": 2,
      "schedule": "0 22 * * *"
    },
    "assistant_commitments": {
      "enabled": true
    }
  },
  "pa": {
    "models": {
      "pa_assistant_commitments": { "role": "exp_rwkv_r",   "temperature": 0.1, "max_tokens": 200,  "max_input_chars": 4000, "frequency_penalty": 0.2 },
      "daily_brief":              { "role": "t1_secondary", "temperature": 0.4, "max_tokens": 3500, "max_input_chars": 16000 }
    }
  }
}
```

All toggles ship `false` initially; we enable per-org via existing feature-flag pattern (`is_feature_enabled`).

## What's intentionally out of scope

- Calendar / email integration (no infra).
- Voice / tone personalisation beyond mode switching.
- Cross-org pollination.
- New UI components — the home conversation and existing question queue are the surfaces.
- New tables — explicitly forbidden by the constraint.

## Acceptance criteria

1. **Useful or silent**: a recall payload with no overdue loops, no hot conversation in 48h, no completed research, no event-passed loop, and no assistant-owed loops produces no `insights` row and no home message. Verified by integration test with a synthetic empty-state org.
2. **Continuity**: with two `recent_briefs` from yesterday and a passed event loop, today's brief explicitly references yesterday's framing AND asks how the event went. Verified by snapshot test on the producer's prompt.
3. **Mute honoured**: with a `pa_user_facts` row of `kind=preference, key="mute:cudf"`, recall returns no warm_topics, no loops, and no asks containing "cudf". Verified by unit test on `build_recall`.
4. **Engagement compounding**: a topic with 2+ `engaged=0` moves in last 14d is excluded from recall. Verified by unit test.
5. **Assistant commitments**: an assistant turn containing "I'll check the X report tomorrow" produces a `pa_open_loops` row with `source_ref` starting `assistant_commitment:`. Verified by integration test on the post-turn flow.
6. **No abstract questions**: `serendipity` and `connect` are unwired from the picker; `_queue_loop_question` is removed from `pa_extractor`. Verified by code search.
7. **Brief absorbs asks**: an `anchored_asks` run produces 2 questions, the brief includes them, and answering one in chat resolves the loop. Verified by end-to-end test.

## Open implementation choices (defer to plan)

- Whether `weekday_midday` and `friday_pm` ship in v1 or are gated until the morning brief is proven.
- Whether `anchored_asks` rephrasing is rule-based templates (cheaper, predictable) or LLM-rephrased (more natural). Recommend templates in v1.
- Locale / timezone handling for the schedule — assume server local time matches user TZ for v1; revisit if multi-TZ users emerge.
