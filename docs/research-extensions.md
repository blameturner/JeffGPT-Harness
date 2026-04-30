# Research Agent — Extension Catalog

Two ways to extend the research agent: cheap doc-type additions (config-only)
and new functions (own pipeline).

---

## A. Doc-type additions (config only)

Each entry below extends the `DOC_TYPES` dict in `tools/research/agent.py`.
Same pipeline (planner → sectioned synthesis → review → ops). The only
new code is the dict entry — opener, closer, tone, summary_role.

| key | when to use | how it's distinct |
|---|---|---|
| `rfp_response` | answering a structured RFP / RFI | requirement → response mapping; compliance matrix orientation |
| `audit_report` | controls / compliance findings | per-control, evidence-graded, neutral findings voice |
| `post_mortem` | incident review with timeline | timeline-first, blameless RCA register, action items |
| `project_plan` | scoping a piece of work | phases / tasks / owners / risks structure |
| `meeting_prep` | brief before a meeting | per-attendee context + agenda + suggested questions |
| `discovery_summary` | sales or product discovery call | needs / pains / criteria / next steps voice |
| `vendor_evaluation` | structured vendor assessment | RFI table, scorecard, references — not just comparison |
| `mentorship_plan` | coaching plan for one person | themes → exercises → progress markers |
| `lesson_plan` | single class / session | objective / activities / assessment / time blocks |
| `onboarding_brief` | new hire / new project ramp | week-by-week sequencing voice |
| `press_release` | external announcement | inverted pyramid, quote slots, embargo line |
| `pitch_brief` | one-pager before a deck | thesis / market / ask in <800 words |
| `executive_brief` | concise C-suite version | TL;DR + "so what" + decision needed |
| `whitepaper_executive` | short white paper for non-technical readers | position + evidence + call-to-action, lighter on jargon |
| `design_brief` | creative / UX direction | problem / audience / constraints / success criteria |
| `experiment_proposal` | A/B or research experiment | hypothesis / metric / design / power / next steps |
| `okr_proposal` | quarterly objectives + key results | objective → measurable KRs → initiatives |
| `roadmap_narrative` | quarter or year-ahead plan | phased themes, dependencies in prose form |
| `annual_review` | year-end personal/team retro | accomplishments / lessons / next year |
| `policy_proposal` | internal policy draft | rationale / scope / rules / exceptions / rollout |

Cost to add each: ~1 dict entry (4 lines). Total: ~15 minutes for all 20.

---

## B. New entire functions (own pipeline)

These need their own table, planner, and ops because the artifact shape
doesn't fit "single Markdown document with sections".

### 1. Code project generator
Multi-file output, not a single document. Planner produces a file tree;
section writer becomes "file writer", one bounded LLM call per file.
Review pass runs `pyright`/`tsc`/`eslint` and feeds errors back per file.

- Storage: `code_projects` row with `files_json` blob (path → content)
- Ops: `add_file`, `refactor_file`, `extract_module`, `generate_tests`,
  `port_to_<language>`, `generate_readme`
- Different from Research because:
  - The artifact is a directory, not prose
  - Quality signal is from compile/lint, not from a reviewer LLM
  - "Expand section" doesn't apply; "refactor file" does

### 2. Data report builder
Same plan → synthesise → review shape, but the corpus is **query results**,
not web pages. Planner generates SQL/NocoDB queries from a question.
Executor runs them. Section writer synthesises findings per query group.
Review pass checks numerical sanity (do percentages sum, are samples large
enough, are the time windows consistent).

- Ops: `add_chart_spec`, `regenerate_for_period`, `extract_kpis`,
  `compare_to_prior_period`
- Different because:
  - The corpus is structured data, not text — needs different prompt patterns
  - Charts are first-class artifacts (Vega-Lite specs)
  - Queries should be re-runnable on a schedule for "freshness" without LLM cost

### 3. Pitch deck builder
Slides are not sections. Each slide has a layout, 5–8 word title, 3–6
bullets, speaker notes. Total document is an ordered array of slide
objects, not a Markdown blob.

- Storage: `slides_json: [{layout, title, bullets, notes, citations}]`
- Ops: `regenerate_slide`, `add_slide`, `reorder`, `expand_speaker_notes`,
  `reframe_for_audience` (investor / customer / board / internal)
- Per-slide token budget is small and tight; constraints matter
- Export targets `.pptx` via existing pptx tooling

### 4. Email sequence / campaign generator
N emails to send over a window, voice-consistent across the series.
Planner produces the sequence outline (subject + intent per email). Writer
drafts each email. Review checks tone consistency, deliverability triggers,
CTA pacing.

- Ops: `regenerate_email`, `add_email`, `tighten_subject_lines`,
  `regenerate_for_persona`, `generate_followups`
- Different because:
  - The artifact is a structured list of small items, not one large document
  - Voice consistency *across* items is the hard problem
  - Each item has metadata (subject, send hint, target segment)

### 5. Workflow / SOP designer
Procedural artifact with decision branches, roles, and RACI. Planner
produces the step graph; writer fills each step with action / inputs /
outputs / owner / failure mode. Review pass checks coverage of edge cases
and dead-ends.

- Storage: nodes + edges (graph)
- Ops: `add_step`, `split_step`, `add_branch`, `assign_owners`,
  `extract_runbook`
- Visual rendering matters (Mermaid / flowchart export)

### 6. Survey / questionnaire designer
Forms with branching logic, question types, validation, scoring. Planner
produces the question outline; writer produces each question's wording,
type, options, branching rules. Review pass checks bias, leading questions,
dead-ends.

- Ops: `add_question`, `simplify_wording`, `add_branch`,
  `regenerate_options`, `generate_pilot_summary_template`
- Different because:
  - Each item is a structured form node, not prose
  - Branching logic is data, not text
  - The artifact has executable behaviour

### 7. Decision matrix / RFD
Options × criteria × weighted scores → recommendation with sensitivity
analysis. Planner extracts options and criteria from the topic. Section
writer fills the cells with evidence + score + rationale. Review pass
stress-tests the weighting (what changes if you halve criterion X?).

- Ops: `add_option`, `add_criterion`, `reweight`, `sensitivity_sweep`,
  `extract_recommendation_paragraph`
- Different because:
  - The artifact is a numerical model with prose annotation, not text
    with a single comparison table
  - Sensitivity analysis is a first-class operation (numerical)
  - Recommendation is computed, then explained

### 8. Contract / agreement drafter
Clause-by-clause composition from a parameterised template + a clause
library. Planner picks which clauses apply and what variables to fill.
Writer drafts each clause and any deviations. Review pass flags clauses
that diverge from a baseline.

- Ops: `add_clause`, `tighten_liability`, `mark_redlines`,
  `regenerate_for_jurisdiction`, `extract_summary_for_signer`
- Different because:
  - Clauses are reusable units with version history
  - Risk-flagging is the dominant review concern
  - Variables / parameters are first-class

---

## C. Gitea-driven automated coding (separate function)

Why this works where "agentic research" struggled: the deliverable is a
**PR**, the verification is **CI**, and the gate is a **human reviewer**.
Every loop has a clear bound — there is no open-ended "is this paper good"
question, only "does the test pass and is the diff small enough to read".

### Concept

A Gitea issue is the spec. A bot opens a draft PR with a code change. The
human reviews in Gitea and either merges, closes, or comments — the bot
revises only on explicit comment. No autonomous merging. No editing main
branch. No work without a corresponding issue.

### Trigger paths

- **Webhook**: Gitea posts to the harness when an issue is labelled
  `auto-code` (or one of `auto-fix`, `auto-test`, `auto-docs`,
  `auto-deps`). The label determines which task type runs.
- **Manual API**: `POST /gitea/{repo}/issues/{n}/run` queues the same
  job. Useful when you want to run a one-off without labelling.
- **Comment trigger** (revision): a comment containing `/revise <text>`
  on the open PR re-queues the bot with the comment as instructions.

### Pipeline (per task)

1. **Fetch** — clone the repo to a per-job working tree (shallow), check
   out a fresh branch `auto/<issue#>-<slug>`.
2. **Spec** — read the issue body + comments; produce a structured spec:
   `{intent, target_files, constraints, success_criteria}`.
3. **Plan** — list the file edits needed, ordered. Hard cap: max 5 files
   touched per task; max 400 LOC change. Larger plans → request the
   issue be split.
4. **Edit** — for each file in the plan, run a section-writer-equivalent:
   read the file, produce a unified diff, apply, run the file's tests
   (if any) before moving on.
5. **Verify** — run the repo's CI command set locally (`pytest`, `tsc`,
   `eslint`, whatever the repo declares in `.harness-ci`). If anything
   fails, run a single retry pass with the failure log fed back in.
6. **Open PR** — push the branch, open a draft PR with: linked issue,
   summary of changes, test results, list of files touched, and any
   constraints the bot couldn't meet.
7. **Review loop**:
   - Human approves → merge happens via Gitea's normal flow (NOT the bot).
   - Human comments `/revise <text>` → re-queue from step 3 with the
     comment appended to constraints.
   - Human closes → bot deletes the branch.

### Task types (label → behaviour)

| Label | What runs | Bound |
|---|---|---|
| `auto-fix` | Read stack trace from issue, find offending code, propose minimal fix | Touched files ≤ 3 |
| `auto-test` | For files in `target_files`, generate test cases that match style of existing tests | New test files only; no edits to source |
| `auto-docs` | Update README / docs to reflect recent code changes | docs/ + README only |
| `auto-deps` | Bump declared deps, run tests, write changelog | requirements.txt / package.json + changelog |
| `auto-feature` | Small feature from spec ≤ 400 LOC | Reject if planner estimates > 400 LOC |
| `auto-refactor` | Refactor a named module per a stated goal | Single module only |

### Storage

A new `gitea_jobs` table mirrors the research/op pattern:

| field | type | notes |
|---|---|---|
| `Id` | int | NocoDB primary |
| `repo` | text | `owner/name` |
| `issue_number` | int | Gitea issue ref |
| `task_type` | text | one of the labels above |
| `branch_name` | text | `auto/<n>-<slug>` |
| `pr_number` | int \| null | filled once PR opened |
| `spec_json` | longtext | structured spec from step 2 |
| `plan_json` | longtext | planned edits from step 3 |
| `files_touched` | text | comma-separated list |
| `loc_changed` | int | ins+del |
| `ci_log` | longtext | last verification output (truncated) |
| `status` | text | `queued / planning / editing / verifying / pr_open / awaiting_review / revising / merged / closed / failed` |
| `error_message` | text | when failed |
| `iterations` | int | revision count |
| `created_at` / `updated_at` | datetime | |

### Endpoints

```
POST   /gitea/webhook                    — Gitea posts here on issue/comment events
POST   /gitea/{repo}/issues/{n}/run      — manual trigger for a labelled issue
POST   /gitea/jobs/{id}/cancel           — close PR + delete branch
GET    /gitea/jobs                       — list jobs with filtering
GET    /gitea/jobs/{id}                  — full state of one job
```

### Tool-queue handlers

Mirror the research pattern:

- `gitea_plan` — runs steps 1–3 (clone, spec, plan); status → `editing`
- `gitea_edit` — runs steps 4–5 (edit + verify); status → `pr_open`
- `gitea_revise` — re-run from step 4 with new instructions

Each is a separate handler so they can be retried independently. The plan
phase is cheap (LLM only); the edit phase is where most time/tokens go.

### Hard rails (non-negotiable)

- **Never push to default branch.** Every change is a PR.
- **Never merge.** Only humans merge, via Gitea's UI.
- **Per-task LOC cap** enforced before edit phase begins.
- **File allowlist** per task type (e.g. `auto-docs` cannot touch source).
- **Token cap** per task ($X equivalent); abort if exceeded.
- **CI gate**: PR is opened as draft if CI fails; explicitly noted in PR
  body so reviewer is not surprised.
- **Repo allowlist** in config — bot only runs on repos you've onboarded.
- **No secrets in prompts** — clone with a token but redact `.env`,
  `secrets/*`, etc. before any file is sent to the LLM.

### Why this isn't a re-skin of the research agent

- Verification is **executable** (CI), not LLM-judged. Different review
  primitive — fact-checking and rewriting don't apply; "the test passed
  or it didn't" does.
- The artifact is a **diff in version control**, not a row in NocoDB.
  The bot's job is to produce a reviewable change, not a finished doc.
- Iteration is **comment-driven** by the human, not auto-loop. No
  "agentic" pass-after-pass behaviour.
- Failure modes are **mechanical** (CI fails, lock conflicts, branch
  exists) and have well-known recoveries — none of the "is this paper
  good enough" judgement that bogged research down.

### Recommended first task type to ship

**`auto-test`** — lowest blast radius (only adds new test files), highest
signal (tests either pass or fail), and immediate value (every codebase
has under-tested files). Once that loop is rock-solid, `auto-docs` is
the natural next step (also additive). `auto-fix` and `auto-feature` only
after the boring ones are reliable.

---

## Recommended next picks

- **Cheapest doc-type to add**: `rfp_response` — high real-world value, no
  architectural cost.
- **Highest-leverage new function**: Code project generator (#1). Stress-tests
  the section-writer pattern in a new direction (per-file with executable
  verification), clarifying whether the architecture lifts cleanly to a
  generic structured-artifact pipeline.
- **Cleanest pivot for a generic artifact pipeline**: Pitch deck builder
  (#3) — the first artifact shape that isn't one big Markdown blob, forcing
  a clean separation between "artifact storage" and "section synthesis".
