Plan: Chief of Staff Agent Platform                                                                                                                                                                                        
                                                        
 Status: Brainstorm in progress — paused partway. Resume by re-reading this file and continuing from "Open decisions."

 Context

 Build a personal Chief of Staff agent for JeffGPT-Harness. Chief is the front door, orchestrator, and quality gate — not just another scheduled agent. Subordinates are specialist tool-bundles (existing ones: research,
 discover, scraper, code; plus future user-defined ones) that Chief delegates to. "OpenClaw-style" personal agent hub in this harness.

 Today the harness has: scheduler.py (APScheduler over NocoDB agent_schedules), workers/user_agents/agent.py (configurable Agent loaded from NocoDB), /run + /run/stream HTTP surface, Huey tool queue, ChromaDB memory,
 FalkorDB graph. Most of the plumbing for this feature exists; the new parts are Chief's persona + delegation tool + review loop + email I/O.

 Core loop (agreed)

 Trigger (email / chat / cron)
    ↓
 Chief receives task
    ↓
 Chief picks subordinate(s) + composes per-invocation prompt
    ↓
 Subordinate runs → returns result
    ↓
 Chief reviews (quality check + editorial pass)
    ↓
    ├── Good → format & send in user's channel
    ├── Not good → re-prompt (≤ 2 retries)
    └── Stuck → clarifying reply (if user-initiated) OR ship-with-note (if cron-initiated)

 Use-case framing (agreed)

 Primary: Personal Chief of Staff (1A) — Chief handles plain-language tasks from user and orchestrates specialists. Example:

 ▎ User emails Chief: "please review this email and suggest a reply."
 ▎ Chief → delegates to a drafting subordinate with a crafted prompt → reviews draft → emails user the suggestion.

 Secondary (same platform, no new primitives): B (knowledge/research), D (outward-facing monitoring), plus user-defined agents created in UI that Chief can direct and tune.

 Not in scope for first cut: full inbox/calendar access (no Gmail/Calendar API unless later decided), deep code-ops agents (C), outbound email to third parties from subordinates.

 Design decisions so far (defaults — confirm when resuming)

 - Review mode: quality check + editorial pass. Chief always polishes subordinate output into user's voice before sending.
 - Iteration budget: max 2 re-prompts (3 runs total), then escalate.
 - Escalation rule: user-initiated (email/chat) → ask clarifying question; cron-initiated → ship best attempt with note.
 - Prompt authority: per-invocation task prompts are default. Persistent persona edits to subordinates (NocoDB writes) are a separate explicit verb, only when user asks Chief to.
 - Subordinate authority (v1): subordinates never send external email. All outbound to user/world comes from Chief.

 Open decisions (resume here)

 Triggers & channels

 - a) Email ingress — lean: inbound-webhook service (Postmark / Mailgun / SES / ImprovMX) → FastAPI route. Alternatives: self-hosted SMTP, IMAP-poll existing inbox.
 - b) CRON-triggered output destination — lean: persist as chat thread + email pointer. Alternatives: email only, chat only, push notification (ntfy/Pushover).
 - c) Threading — lean: email Message-ID / In-Reply-To threading, keyed to a conversation row Chief can recall.
 - d) Identity — lean: single outbound identity (chief@). Subordinates have no external voice.

 Not yet asked

 - e) Trust/permissions tiering — read-only vs. send-email vs. modify-other-agents. What does Chief do autonomously vs. needs approval? Is there an "approval queue" surface?
 - f) Concurrency & conversation state — multiple tasks in flight at once? How is state scoped (per-conversation, per-user, global Chief memory)?
 - g) Control-plane tools for Chief — concrete tool list: list_agents, read_recent_outputs, invoke_agent(name, task), review_output(result), update_agent_persona(name, prompt), schedule_agent(name, cron),
 ask_user(question), send_reply(channel, body). Refine this list.
 - h) Chief persona — tone, defaults, when to be terse vs. thorough. Skip for now; write when implementing.
 - i) Subordinate library — first cut: wrap existing research / discover / scraper / code as "specialists." Do we need a new abstraction, or do they stay as-is and Chief calls them via the existing /run interface?

 Sub-projects & build order

 1. Chief core + delegation + review loop (this plan) — Chief agent, invoke_agent + review_output tools, chat-in / chat-out only.
 2. CRON-triggered Chief — point agent_schedules entries at Chief with standing instructions; decide output destination (open decision b).
 3. Email ingress + egress — inbound webhook, outbound via existing SMTP lib or service. Threading, identity.
 4. Control-plane writes — Chief can edit subordinate personas, toggle schedules, spawn new agents from natural language.
 5. User-defined agents in UI — surface Chief's spawning action as a first-class UI flow.

 Each is its own brainstorm → implementation cycle. This file tracks sub-project 1.

 Critical files to know when implementing

 - scheduler.py — CRON entry point. Chief just becomes a new row in agent_schedules.
 - workers/user_agents/agent.py — base class Chief extends (persona from NocoDB).
 - workers/user_agents/generator_agent.py — structured-output variant, likely useful for Chief's internal review verdicts.
 - app/routers/agents.py — /run, /run/stream — Chief calls subordinates by POSTing here internally, OR we call the Agent class directly in-process (decide when implementing; in-process is simpler).
 - tools/dispatcher.py + tools/contract.py — tool registry. Chief's new tools (invoke_agent, review_output, etc.) register here.
 - tools/gate.py — permissions gate; Chief's write-tier tools plug in here.
 - infra/nocodb_client.py — agent, agent_schedules, agent_outputs tables. Chief reads/writes these.
 - infra/memory.py — ChromaDB for Chief's conversation memory / past-task recall.
 - workers/tool_queue.py — Huey queue. Long-running subordinate work should go through here with idle-gate backoff (per existing queue-design principle).

 Verification plan (for sub-project 1, when built)

 1. Create a Chief agent row in NocoDB with persona + system prompt.
 2. Send a chat task: "please research the latest news on X and summarize."
 3. Observe: Chief invokes research subordinate → reviews → editorial pass → returns.
 4. Send a deliberately bad task: watch that Chief re-prompts up to 2×, then escalates.
 5. Send a cron trigger (or manually fire one): confirm Chief's standing-instruction flow works and output lands in the chosen destination.
 6. Check ChromaDB: confirm conversation memory / past-task recall persists across runs.