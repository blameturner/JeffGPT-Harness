from __future__ import annotations

import json
import logging
import re

from infra.config import get_function_config
from infra.graph import write_relationship
from shared.models import model_call

_log = logging.getLogger("workers.relationships")


_RELATIONSHIP_EXTRACTION_PROMPT = """You are building a knowledge graph that will power retrieval and reasoning later. Every edge you emit should earn its place in the graph by encoding a NON-OBVIOUS fact that a reader couldn't guess from the entity names alone.

# YOUR TASK

Read the CONTENT below and extract the most valuable entity relationships you can find, as a strict JSON array. Quality >> quantity. 3-8 strong, specific relationships beats 15 shallow ones. If the content is thin, emit fewer. Never pad.

# THE SCHEMA (strict — any deviation makes the output unusable)

Return ONLY a JSON array. Each element is an object with EXACTLY these five keys:

```
{{
  "from_type":    "<PascalCase entity type, e.g. Library, Protocol, Vulnerability, Regulation, Company, Person, Algorithm, Tool, Concept>",
  "from_name":    "<the specific named entity, verbatim from the content where possible>",
  "relationship": "<UPPER_SNAKE_CASE verb phrase — see preferred verbs below>",
  "to_type":      "<PascalCase entity type>",
  "to_name":      "<specific named entity>"
}}
```

No extra keys. No prose outside the array. No markdown fences. No comments. The first character of your response must be `[` and the last must be `]`.

# PREFERRED RELATIONSHIP VERBS (causal, structural, mechanism-revealing)

Use these verbs when they fit. They carry far more signal than flat "IS_A" / "HAS":

- CAUSES — A reliably produces B (e.g. "rate limiting CAUSES search degradation")
- ENABLES — A makes B possible (e.g. "TLS 1.3 ENABLES 0-RTT resumption")
- REQUIRES — A cannot work without B (e.g. "Kubernetes REQUIRES etcd")
- DEPENDS_ON — A uses B at runtime (e.g. "FastAPI DEPENDS_ON Starlette")
- BYPASSES — A circumvents B's restriction (e.g. "Docker BYPASSES UFW")
- REPLACES — A supersedes B (e.g. "HTTP/3 REPLACES HTTP/2 transport")
- IMPLEMENTS — A provides B's interface (e.g. "Pydantic IMPLEMENTS JSONSchema")
- EXPLOITS — A takes advantage of B's weakness (e.g. "CVE-2024-3094 EXPLOITS xz backdoor")
- MITIGATES — A reduces risk from B
- CONFLICTS_WITH — A and B are incompatible
- PRECEDES — A must run before B in a workflow
- CONSTRAINS — A limits what B can do
- AUTHORED_BY — attributes creation
- REGULATED_BY — jurisdiction / standards body
- BUILT_ON — A inherits from B's architecture

Use IS_A / HAS / CONTAINS / PART_OF only when the taxonomic fact is non-obvious (e.g. "PostgreSQL IS_A MVCC_Database" is OK, but "Python IS_A ProgrammingLanguage" is forbidden — see DO NOT EMIT below).

# QUALITY BAR — BEFORE EMITTING, CHECK EACH TRIPLE

For every triple you're about to include, ask: "Would a reader who knows nothing about this content learn something specific from this edge?" If the answer is "no, I could have guessed that from the entity names", DELETE the triple.

Other checks:
- Is the from_name / to_name a SPECIFIC named thing from the content, not a generic noun? (e.g. "PostgreSQL 17" not "the database", "CVE-2024-3094" not "a vulnerability")
- Is the relationship verb more informative than "HAS" or "IS_A"?
- Could the same fact be expressed more precisely with a different verb from the list?

# DO NOT EMIT (shallow or vacuous — these add bulk, no signal)

- Generic taxonomy: `Python IS_A ProgrammingLanguage`, `Linux IS_A OperatingSystem`, `Apple IS_A Company`
- Self-referential: `FastAPI IS_A Framework` (the entity name already says so)
- Dictionary definitions: `Docker HAS Containers`, `Database HAS Tables`
- Duplicates with trivial variation: if you've emitted `A CAUSES B`, don't also emit `B CAUSED_BY A`
- Relationships where either side is unnamed ("the system", "this tool", "the user") — only named entities
- Speculative or hedged claims the content merely mentions ("X might cause Y") — only assertive facts

# READ THE WHOLE CONTENT

Do not stop extracting after the first few paragraphs. Scan the entire CONTENT block below. The most valuable relationships are often in the middle or near the end where the technical detail lives. If you find yourself only citing the first paragraph, re-read the rest.

# POSITIVE EXAMPLES (what "good" looks like)

Content excerpt: "The xz-utils backdoor (CVE-2024-3094) was introduced by a maintainer who had gained commit access over two years. The malicious code hooks into OpenSSH's sshd via a liblzma dependency, allowing pre-authentication remote code execution on systems running sshd linked against liblzma."

Good triples from that excerpt:
```
[
  {{"from_type":"Vulnerability","from_name":"CVE-2024-3094","relationship":"EXPLOITS","to_type":"Library","to_name":"liblzma"}},
  {{"from_type":"Vulnerability","from_name":"CVE-2024-3094","relationship":"ENABLES","to_type":"AttackClass","to_name":"PreAuth RCE"}},
  {{"from_type":"Service","from_name":"sshd","relationship":"DEPENDS_ON","to_type":"Library","to_name":"liblzma"}},
  {{"from_type":"Vulnerability","from_name":"CVE-2024-3094","relationship":"INTRODUCED_BY","to_type":"ActorPattern","to_name":"Long-term maintainer access compromise"}}
]
```

# NEGATIVE EXAMPLES (what NOT to do)

Bad triples from the same excerpt (explained):
```
{{"from_type":"Library","from_name":"xz-utils","relationship":"IS_A","to_type":"Software","to_name":"Library"}}
  ↑ Vacuous taxonomy. Entity name already says "Library".

{{"from_type":"Vulnerability","from_name":"CVE-2024-3094","relationship":"HAS","to_type":"Code","to_name":"Malicious code"}}
  ↑ "HAS malicious code" is what a vulnerability IS. No signal added.

{{"from_type":"Service","from_name":"sshd","relationship":"IS_A","to_type":"Software","to_name":"Server"}}
  ↑ Generic. Doesn't help the graph.
```

# CONTENT

{content}

# OUTPUT

Emit the JSON array now. First character `[`, last character `]`. No prose."""


def _salvage_json_array(text: str) -> list | None:
    # handle JSON truncated at token limit — close the array after the last complete object
    last_close = text.rfind("}")
    if last_close == -1:
        return None
    truncated = text[:last_close + 1].rstrip(", \n\t") + "]"
    if not truncated.startswith("["):
        truncated = "[" + truncated
    try:
        result = json.loads(truncated)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return None


def _extract_relationships(
    text: str,
    org_id: int,
    conversation_id: int | None = None,
    source_chunk_id: str | None = None,
) -> tuple[int, int]:
    cfg = get_function_config("relationships")
    max_input = cfg.get("max_input_chars", 8000)
    _log.debug(
        "extracting relationships  org=%d text_len=%d max_input=%d conv=%s chunk=%s",
        org_id, len(text), max_input, conversation_id, source_chunk_id,
    )
    prompt = _RELATIONSHIP_EXTRACTION_PROMPT.format(
        content=text[:max_input],
    )
    raw, tokens = model_call("relationships", prompt)
    if not raw:
        _log.warning("relationship extraction returned empty  tokens=%d", tokens)
        return 0, tokens
    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
        triples = json.loads(cleaned)
    except json.JSONDecodeError:
        triples = _salvage_json_array(cleaned)
        if triples is None:
            _log.warning("relationship extraction unparseable: %s", raw[:300])
            return 0, tokens
        _log.info("relationship extraction salvaged %d items from truncated JSON", len(triples))
    if not isinstance(triples, list):
        _log.warning("relationship extraction returned non-list: %s", type(triples).__name__)
        return 0, tokens

    _log.debug("extracted %d relationship candidates", len(triples))
    written = 0
    for t in triples[:15]:
        try:
            write_relationship(
                org_id=org_id,
                from_type=str(t["from_type"]),
                from_name=str(t["from_name"]),
                relationship=str(t["relationship"]),
                to_type=str(t["to_type"]),
                to_name=str(t["to_name"]),
                conversation_id=conversation_id,
                source_chunk_id=source_chunk_id,
            )
            written += 1
        except Exception:
            _log.error("relationship write failed  triple=%s", t, exc_info=True)
    _log.info("relationships written  %d/%d  org=%d", written, len(triples[:15]), org_id)
    return written, tokens
