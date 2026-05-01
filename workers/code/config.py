CODE_MODES: dict[str, str] = {
    "chat": "Open-ended code conversation. Do not write files.",
    "plan": "Create a concrete implementation plan. Do not write files.",
    "apply": "Implement changes and emit fenced file blocks for persistence.",
    "review": "Review code or diffs. Do not write files.",
    "explain": "Explain how code works. Do not write files.",
    "decide": "Propose a single ADR file in fenced-output format.",
    "scaffold": "Generate initial project/file scaffolding in fenced-output format.",
    "refine": "Rewrite a selected file region using a constrained patch.",
    # Back-compat alias retained for older clients.
    "execute": "Alias of apply.",
}

CODE_DEFAULT_MODE = "plan"

CODE_MODE_META: dict[str, dict[str, str]] = {
    "chat": {"label": "Chat", "description": "Open-ended conversation without file writes."},
    "plan": {"label": "Plan", "description": "Produce a concrete implementation checklist."},
    "apply": {"label": "Apply", "description": "Implement and persist file changes."},
    "review": {"label": "Review", "description": "Review code quality and risks."},
    "explain": {"label": "Explain", "description": "Explain code behavior and flow."},
    "decide": {"label": "Decide", "description": "Write one ADR proposal."},
    "scaffold": {"label": "Scaffold", "description": "Generate starter files from a prompt."},
    "refine": {"label": "Refine", "description": "Patch only a selected file region."},
    "execute": {"label": "Execute", "description": "Legacy alias for Apply."},
}

CODE_STYLES: dict[str, str] = {
    "none": "No extra lens; follow mode instructions directly.",
    "bug_fix": "Focus on correctness and root-cause bug fixes.",
    "tests": "Prioritize test quality and coverage across happy/edge/failure cases.",
    "security": "Prioritize security posture and risk reduction.",
    "optimize": "Prioritize performance and efficiency without needless complexity.",
    "refactor": "Improve structure and maintainability while preserving behavior.",
    "document": "Prioritize docs, comments, and code readability for maintainers.",
    "new_feature": "Prioritize clean feature implementation aligned with project conventions.",
    "accessibility": "Treat accessibility requirements as first-class constraints.",
    "migration": "Prioritize safe migration with compatibility notes and deprecation handling.",
}

CODE_DEFAULT_STYLE = "none"

CODE_STYLE_META: dict[str, dict[str, str]] = {
    "none": {"label": "None", "description": "Pure mode-driven behavior."},
    "bug_fix": {"label": "Bug Fix", "description": "Debug and resolve defects."},
    "tests": {"label": "Tests", "description": "Author and improve tests."},
    "security": {"label": "Security", "description": "Harden security."},
    "optimize": {"label": "Optimize", "description": "Improve performance."},
    "refactor": {"label": "Refactor", "description": "Improve structure without behavior drift."},
    "document": {"label": "Document", "description": "Improve docs and comments."},
    "new_feature": {"label": "New Feature", "description": "Build new capabilities."},
    "accessibility": {"label": "Accessibility", "description": "Improve accessibility quality."},
    "migration": {"label": "Migration", "description": "Handle upgrades and migrations."},
}

CODE_MAX_TOKENS: dict[str, int] = {
    "none": 3000,
    "bug_fix": 2500,
    "tests": 2800,
    "security": 2800,
    "optimize": 2800,
    "refactor": 3000,
    "document": 2200,
    "new_feature": 4096,
    "accessibility": 2800,
    "migration": 3200,
}

CODE_TEMPERATURES: dict[str, float] = {
    "none": 0.3,
    "bug_fix": 0.2,
    "tests": 0.2,
    "security": 0.2,
    "optimize": 0.2,
    "refactor": 0.3,
    "document": 0.3,
    "new_feature": 0.4,
    "accessibility": 0.2,
    "migration": 0.2,
}


def _resolve_key(requested: str | None, catalog: dict[str, str], default: str) -> str:
    key = (requested or "").strip().lower()
    if not key:
        return default
    if key == "test":
        key = "tests"
    if key == "general":
        key = "none"
    return key if key in catalog else default


def resolve_code_mode(requested: str | None) -> str:
    key = _resolve_key(requested, CODE_MODES, CODE_DEFAULT_MODE)
    return "apply" if key == "execute" else key


def code_mode_prompt(mode: str) -> tuple[str, str]:
    key = resolve_code_mode(mode)
    return key, CODE_MODES.get(key, CODE_MODES[CODE_DEFAULT_MODE])


def list_code_modes() -> list[dict]:
    out: list[dict] = []
    for k, v in CODE_MODES.items():
        meta = CODE_MODE_META.get(k, {})
        out.append(
            {
                "key": k,
                "label": meta.get("label") or k.replace("_", " ").title(),
                "description": meta.get("description", ""),
                "prompt": v,
            }
        )
    return out


def code_style_prompt(requested: str | None) -> tuple[str, str]:
    key = _resolve_key(requested, CODE_STYLES, CODE_DEFAULT_STYLE)
    return key, CODE_STYLES[key]


def list_code_styles() -> list[dict]:
    out: list[dict] = []
    for k, v in CODE_STYLES.items():
        meta = CODE_STYLE_META.get(k, {})
        out.append(
            {
                "key": k,
                "label": meta.get("label") or k.replace("_", " ").title(),
                "description": meta.get("description", ""),
                "prompt": v,
            }
        )
    return out


def code_max_tokens(response_style: str | None) -> int:
    key = _resolve_key(response_style, CODE_MAX_TOKENS, CODE_DEFAULT_STYLE)  # type: ignore[arg-type]
    return CODE_MAX_TOKENS.get(key, CODE_MAX_TOKENS[CODE_DEFAULT_STYLE])


def code_temperature(response_style: str | None) -> float:
    key = _resolve_key(response_style, CODE_TEMPERATURES, CODE_DEFAULT_STYLE)  # type: ignore[arg-type]
    return CODE_TEMPERATURES.get(key, CODE_TEMPERATURES[CODE_DEFAULT_STYLE])
