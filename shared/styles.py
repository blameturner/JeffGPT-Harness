from __future__ import annotations

from workers.chat.config import CHAT_DEFAULT_STYLE, CHAT_STYLES


CODE_STYLES: dict[str, str] = {
    "explain": (
        "Explain the provided code clearly. Walk through what it does section "
        "by section. Cover why it is written the way it is, not just what it "
        "does. Highlight any non-obvious decisions or patterns. Assume the "
        "user wants to understand the code deeply, not just use it."
    ),
    "review": (
        "Review the provided code as a senior engineer. Identify issues with "
        "correctness, clarity, performance, and best practices. Be specific — "
        "name the line or pattern, explain why it is a problem, and suggest "
        "the fix. Do not pad with praise. Focus on what needs to change."
    ),
    "refactor": (
        "Refactor the provided code. Improve readability, structure, and "
        "adherence to best practices without changing the behaviour. After "
        "showing the refactored code, briefly explain what changed and why. "
        "Do not add new features or fix bugs unless they are directly caused "
        "by structural problems."
    ),
    "debug": (
        "Debug the provided code. Identify the root cause of the problem, "
        "not just the symptom. Explain why the bug occurs. Provide the fix "
        "and explain what changed. If there are multiple issues, address "
        "them in order of severity."
    ),
    "build": (
        "Build the requested code from scratch. Follow the project's "
        "existing patterns and conventions where evident. Write "
        "production-quality code — not a prototype. After the code, briefly "
        "explain any key decisions made during implementation."
    ),
    "test": (
        "Write tests for the provided code. Cover happy paths, edge cases, "
        "and failure modes. Use the testing framework already in use in the "
        "project. After the tests, briefly explain what each test covers "
        "and why it matters."
    ),
    "optimise": (
        "Optimise the provided code for performance and efficiency. Identify "
        "what is slow or wasteful and explain why. Rewrite to address those "
        "specific bottlenecks. Quantify the improvement where possible. Do "
        "not sacrifice readability unnecessarily — explain any tradeoffs made."
    ),
    "security": (
        "Audit the provided code for security vulnerabilities. Look for "
        "injection risks, authentication and authorisation issues, data "
        "exposure, insecure dependencies, and any other security concerns. "
        "For each issue, state the vulnerability, its severity, and the "
        "specific remediation. Be thorough — assume this code is going to "
        "production."
    ),
}

CODE_DEFAULT_STYLE = "review"


def resolve_style(
    requested: str | None,
    catalog: dict[str, str],
    default: str,
) -> tuple[str, str]:
    key = (requested or "").strip().lower()
    if key in catalog:
        return key, catalog[key]
    return default, catalog[default]


def chat_style_prompt(requested: str | None) -> tuple[str, str]:
    return resolve_style(requested, CHAT_STYLES, CHAT_DEFAULT_STYLE)


def code_style_prompt(requested: str | None) -> tuple[str, str]:
    return resolve_style(requested, CODE_STYLES, CODE_DEFAULT_STYLE)


def list_chat_styles() -> list[dict]:
    return [{"key": k, "prompt": v} for k, v in CHAT_STYLES.items()]


def list_code_styles() -> list[dict]:
    return [{"key": k, "prompt": v} for k, v in CODE_STYLES.items()]


_NO_INLINE_CITATIONS = (
    "Do NOT include numbered citations like [1] or [2] in your response. "
    "Do NOT list sources at the end of your reply. The UI renders source "
    "attribution separately from your prose."
)

SEARCH_CONTEXT_TEMPLATES: dict[str, str] = {
    "conversational_weave": (
        "BACKGROUND FACTS (from web search): the user mentioned real-world "
        "things in their message. The facts below are available for "
        "grounding. Weave the useful ones into your reply naturally, the "
        "way a friend who happens to know would mention them in passing. "
        "Do NOT list them, do NOT announce that you searched. If a fact "
        "contradicts what the user said, mention it gently (\"actually I "
        "think they won that one?\"), not formally. Skip any fact that "
        "isn't relevant to what the user is actually doing or feeling. "
        + _NO_INLINE_CITATIONS
    ),

    "direct_answer": (
        "SEARCH RESULTS: the user asked for a specific fact. Lead with the "
        "answer in one sentence. One follow-up sentence of context if "
        "useful. Mention the source in passing only when it materially "
        "affects trust (e.g. \"per the official standings\"). No headers, "
        "no bullets, no preamble. " + _NO_INLINE_CITATIONS
    ),

    "explanatory": (
        "SEARCH RESULTS: the user asked to understand a concept. Build "
        "progressively — core idea first, then mechanism, then a concrete "
        "example. Use prose paragraphs, not structure. Draw on the sources "
        "below for accuracy and specifics. Mention sources in passing only "
        "where it matters for trust. " + _NO_INLINE_CITATIONS
    ),

    "recommendation": (
        "SEARCH RESULTS: the user is looking for options. Suggest 2-4 "
        "choices, one casual sentence per option explaining why it might "
        "suit. Match the user's asking register — if they asked casually, "
        "answer casually. Write it as prose, not as a numbered list with "
        "headers. Mention specific names, prices, and standout details "
        "where the sources provide them. " + _NO_INLINE_CITATIONS
    ),

    "comparison": (
        "SEARCH RESULTS: the user is comparing options. Cover each "
        "option's strengths and tradeoffs in plain prose. Use a small "
        "table only if there are 3+ dimensions to compare and a table "
        "actually clarifies things. Lead with which option suits which "
        "use case if a clear winner exists per scenario. "
        + _NO_INLINE_CITATIONS
    ),

    "research_synthesis": (
        "SEARCH RESULTS: the user is researching a non-trivial topic. "
        "Synthesise across the sources below into an authority-weighted "
        "answer. Lead with the strongest finding. Qualify with conflicting "
        "views or counter-evidence. Distinguish primary sources (research "
        "papers, official docs) from secondary (blog posts, forum "
        "discussions). Longer form is OK here — every paragraph should "
        "carry information. " + _NO_INLINE_CITATIONS
    ),

    "troubleshooting": (
        "SEARCH RESULTS: the user has a problem. Lead with the most "
        "likely cause based on the sources. Give the specific fix or "
        "workaround second. Mention secondary causes only if relevant. "
        "Preserve any error messages or commands verbatim. Don't pad "
        "with general advice. " + _NO_INLINE_CITATIONS
    ),

    "code_lookup": (
        "SEARCH RESULTS: the user is looking up an API or syntax. Lead "
        "with the actual signature or example code in a fenced code "
        "block. One sentence of context after. Mention the version it "
        "applies to if the sources specify. Source mentioned in passing "
        "only. " + _NO_INLINE_CITATIONS
    ),

    "code_debug": (
        "SEARCH RESULTS: the user has a code error. Lead with the "
        "diagnosed cause from the sources. Show the fix in a fenced code "
        "block. Explain why the fix works in one or two sentences. Don't "
        "lecture on best practices. " + _NO_INLINE_CITATIONS
    ),

    "code_build": (
        "SEARCH RESULTS: the user is building something with a named "
        "library. Use the example code from the sources as a reference "
        "for the API specifics. Produce production-quality code in your "
        "reply. " + _NO_INLINE_CITATIONS
    ),

    "task_confirm": (
        "TASK CONTEXT: the user gave you a command. Confirm what you "
        "did in one sentence. No preamble."
    ),

    # empty bodies for no-search intents — keeps search_context_for() from raising KeyError
    "chitchat_casual": "",
    "code_explain": "",
    "code_review": "",
    "code_refactor": "",
    "code_test": "",
    "code_optimise": "",
    "code_security": "",
}


def search_context_for(template_key: str) -> str:
    return SEARCH_CONTEXT_TEMPLATES.get(
        template_key,
        SEARCH_CONTEXT_TEMPLATES["direct_answer"],
    )


def list_search_templates() -> list[dict]:
    return [{"key": k, "prompt": v} for k, v in SEARCH_CONTEXT_TEMPLATES.items()]
