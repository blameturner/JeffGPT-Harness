from __future__ import annotations

import re


def parse_period_days(period: str) -> int:
    p = (period or "30d").strip().lower()
    m = re.match(r"^(\d+)([dw])$", p)
    if not m:
        return 30
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "w":
        return max(1, n * 7)
    return max(1, n)


def count_todo_markers(text: str) -> int:
    if not text:
        return 0
    total = 0
    for marker in ("TODO", "FIXME", "XXX", "HACK", "NOTE"):
        total += len(re.findall(rf"\b{marker}\b", text, flags=re.IGNORECASE))
    return total

