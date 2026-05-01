from __future__ import annotations

from workers.code.config import code_mode_prompt


def assemble_code_system_prompt(
    *,
    mode: str,
    style_prompt: str,
    project_name: str = "",
    project_slug: str = "",
    system_note: str = "",
    pinned_context: str = "",
    path_manifest: str = "",
    context_notice: str = "",
    interactive_fs: bool = False,
    glossary_terms: list[str] | None = None,
) -> str:
    mode_key, mode_prompt = code_mode_prompt(mode)
    parts: list[str] = []

    if project_name or project_slug:
        parts.append(f"Project: {project_name} ({project_slug})".strip())
    if system_note:
        parts.append("System note:\n" + system_note)
    if context_notice:
        parts.append("Context budget notes:\n" + context_notice)
    if pinned_context:
        parts.append("Pinned files:\n" + pinned_context)
    if path_manifest:
        parts.append("Workspace manifest:\n" + path_manifest)
    if glossary_terms:
        parts.append("Project vocabulary (use these terms verbatim): " + ", ".join(glossary_terms))

    parts.append(f"Mode ({mode_key}): {mode_prompt}")
    parts.append("Style lens: " + style_prompt)

    if mode_key in {"apply", "scaffold", "decide", "refine"} and not interactive_fs:
        parts.append(
            "When changing files, emit fenced blocks in this exact format: "
            "```file path=/abs/path mode=replace|patch|append|delete summary=\"...\"```"
        )

    if interactive_fs:
        parts.append(
            "interactive_fs is enabled. Use explicit tool directives when you need filesystem actions."
        )
        parts.append(
            "Tool directive format:\n"
            "```tool name=fs_list```\n"
            "or\n"
            "```tool name=fs_read path=/abs/path```\n"
            "or\n"
            "```tool name=fs_write path=/abs/path summary=\"...\"\n<full file content>\n```\n"
            "or\n"
            "```tool name=fs_delete path=/abs/path```"
        )

    return "\n\n".join(p for p in parts if p.strip())



