from __future__ import annotations

import argparse
import re

from infra.nocodb_client import NocodbClient


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug[:80] or "project"


def run(org_id: int | None, dry_run: bool) -> None:
    db = NocodbClient()
    if "knowledge_sources" not in db.tables:
        raise RuntimeError("knowledge_sources table not found")

    where = "(type,eq,codebase)"
    if org_id is not None:
        where = f"{where}~and(org_id,eq,{org_id})"

    rows = db._get_paginated("knowledge_sources", params={"where": where, "limit": 5000})
    created = 0
    skipped = 0

    for row in rows:
        row_org_id = int(row.get("org_id") or 0)
        name = (row.get("name") or "").strip() or f"codebase-{row.get('Id')}"
        slug = slugify(name)
        collection = (row.get("collection_name") or "").strip()
        existing = db._get(
            "projects",
            params={
                "where": f"(org_id,eq,{row_org_id})~and(slug,eq,{slug})~and(archived_at,is,null)",
                "limit": 1,
            },
        ).get("list", []) if "projects" in db.tables else []
        if existing:
            skipped += 1
            continue
        if dry_run:
            print(f"[dry-run] create project org={row_org_id} name={name!r} slug={slug!r} collection={collection!r}")
            created += 1
            continue
        db.create_project(
            org_id=row_org_id,
            name=name,
            slug=slug,
            description=row.get("description") or "",
            system_note="",
            default_model="code",
            retrieval_scope=[collection] if collection else [],
            chroma_collection=collection,
        )
        created += 1

    print(f"done: scanned={len(rows)} created={created} skipped={skipped} dry_run={dry_run}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate codebase knowledge sources into projects")
    parser.add_argument("--org-id", type=int, default=None, help="Optional org filter")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    args = parser.parse_args()
    run(org_id=args.org_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

