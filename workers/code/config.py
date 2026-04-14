CODE_STYLES: dict[str, str] = {
    "general": "General code help - answer questions, provide guidance",
    "bug_fix": "Debug and fix bugs or errors in existing code",
    "test": "Write or improve tests",
    "security": "Security audit and hardening",
    "optimize": "Performance optimization",
    "refactor": "Improve code structure without changing behavior",
    "document": "Add or improve documentation",
    "new_feature": "Build new features from scratch",
    "review": "Code review with specific feedback",
    "explain": "Explain how the code works",
}

CODE_DEFAULT_STYLE = "general"

CODE_MAX_TOKENS: dict[str, int] = {
    "general": 3000,
    "bug_fix": 2500,
    "test": 2500,
    "security": 2500,
    "optimize": 2500,
    "refactor": 3000,
    "document": 2000,
    "new_feature": 4096,
    "review": 3000,
    "explain": 2500,
}

CODE_TEMPERATURES: dict[str, float] = {
    "general": 0.7,
    "bug_fix": 0.7,
    "test": 0.7,
    "security": 0.7,
    "optimize": 0.7,
    "refactor": 0.7,
    "document": 0.7,
    "new_feature": 0.7,
    "review": 0.7,
    "explain": 0.7,
}


def resolve_style(requested: str | None, catalog: dict[str, str], default: str) -> tuple[str, str]:
    key = (requested or "").strip().lower()
    if not key:
        key = default
    if key in catalog:
        return key, catalog[key]
    return default, catalog[default]


def code_style_prompt(requested: str | None) -> tuple[str, str]:
    return resolve_style(requested, CODE_STYLES, CODE_DEFAULT_STYLE)


def list_code_styles() -> list[dict]:
    return [{"key": k, "prompt": v} for k, v in CODE_STYLES.items()]


def code_max_tokens(response_style: str | None) -> int:
    key = (response_style or "").strip().lower()
    if not key:
        key = "general"
    return CODE_MAX_TOKENS.get(key, CODE_MAX_TOKENS.get("general", 3000))


def code_temperature(response_style: str | None) -> float:
    key = (response_style or "").strip().lower()
    if not key:
        key = "general"
    return CODE_TEMPERATURES.get(key, CODE_TEMPERATURES.get("general", 0.7))