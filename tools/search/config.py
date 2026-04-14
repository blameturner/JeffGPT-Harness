from __future__ import annotations

_NO_INLINE_CITATIONS = (
    "Do NOT include numbered citations like [1] or [2] in your response. "
    "Do NOT list sources at the end of your reply. The UI renders source "
    "attribution separately from your prose."
)

SEARCH_STYLES: dict[str, str] = {
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

    "chitchat_casual": "",
    "code_explain": "",
    "code_review": "",
    "code_refactor": "",
    "code_test": "",
    "code_optimise": "",
    "code_security": "",
}


def search_context_for(template_key: str) -> str:
    return SEARCH_STYLES.get(template_key, SEARCH_STYLES["direct_answer"])


def resolve_style(requested: str | None, catalog: dict[str, str], default: str) -> tuple[str, str]:
    key = (requested or "").strip().lower()
    if key in catalog:
        return key, catalog[key]
    return default, catalog[default]


def search_style_prompt(requested: str | None) -> tuple[str, str]:
    return resolve_style(requested, SEARCH_STYLES, "direct_answer")


def list_search_styles() -> list[dict]:
    return [{"key": k, "prompt": v} for k, v in SEARCH_STYLES.items() if v]


# for future use
# SEARCH_MAX_TOKENS: dict[str, int] = {
#     "conversational_weave": 1500,
#     "direct_answer": 1000,
#     "explanatory": 2000,
#     "recommendation": 2000,
#     "comparison": 2500,
#     "research_synthesis": 3000,
#     "troubleshooting": 1500,
#     "code_lookup": 1500,
#     "code_debug": 1500,
#     "code_build": 2000,
#     "task_confirm": 200,
#     "default": 1500,
# }
#
# SEARCH_TEMPERATURES: dict[str, float] = {
#     "conversational_weave": 0.7,
#     "direct_answer": 0.7,
#     "explanatory": 0.7,
#     "recommendation": 0.7,
#     "comparison": 0.7,
#     "research_synthesis": 0.7,
#     "troubleshooting": 0.7,
#     "code_lookup": 0.7,
#     "code_debug": 0.7,
#     "code_build": 0.7,
#     "task_confirm": 0.7,
#     "default": 0.7,
# }
#
# def search_max_tokens(response_style: str | None) -> int:
#     key = (response_style or "").strip().lower() or "default"
#     return SEARCH_MAX_TOKENS.get(key, SEARCH_MAX_TOKENS.get("default", 1500))
#
#
# def search_temperature(response_style: str | None) -> float:
#     key = (response_style or "").strip().lower() or "default"
#     return SEARCH_TEMPERATURES.get(key, SEARCH_TEMPERATURES.get("default", 0.7))