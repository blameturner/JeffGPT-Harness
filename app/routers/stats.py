import logging

from fastapi import APIRouter, HTTPException

from infra.nocodb_client import NocodbClient

_log = logging.getLogger("main.stats")

router = APIRouter()


@router.get("/stats/usage")
def stats_usage(org_id: int, period: str = "30d"):
    from datetime import datetime, timedelta, timezone
    try:
        db = NocodbClient()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    now = datetime.now(timezone.utc)
    if period == "7d":
        start = now - timedelta(days=7)
    elif period == "30d":
        start = now - timedelta(days=30)
    else:
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)

    period_start = start.strftime("%Y-%m-%d")
    period_end = now.strftime("%Y-%m-%d")

    try:
        msg_where = f"(org_id,eq,{org_id})~and(CreatedAt,gte,{start.isoformat()})"
        messages = db._get("messages", params={"where": msg_where, "limit": 5000}).get("list", [])
    except Exception:
        messages = []

    try:
        run_where = f"(org_id,eq,{org_id})~and(CreatedAt,gte,{start.isoformat()})"
        runs = db._get("agent_runs", params={"where": run_where, "limit": 5000}).get("list", [])
    except Exception:
        runs = []

    total_tokens_in = sum(int(m.get("tokens_input") or 0) for m in messages)
    total_tokens_in += sum(int(r.get("tokens_input") or 0) for r in runs)
    total_tokens_out = sum(int(m.get("tokens_output") or 0) for m in messages)
    total_tokens_out += sum(int(r.get("tokens_output") or 0) for r in runs)
    total_requests = len(messages) + len(runs)
    total_conversations = len({m.get("conversation_id") for m in messages if m.get("conversation_id")})
    failed_runs = [r for r in runs if r.get("status") == "failed"]
    total_errors = len(failed_runs)

    by_model: dict = {}
    for m in messages:
        model = (m.get("model") or "unknown").strip() or "unknown"
        entry = by_model.setdefault(model, {
            "model_name": model, "requests": 0, "tokens_input": 0,
            "tokens_output": 0, "durations": [], "error_count": 0,
        })
        entry["requests"] += 1
        entry["tokens_input"] += int(m.get("tokens_input") or 0)
        entry["tokens_output"] += int(m.get("tokens_output") or 0)
    for r in runs:
        model = (r.get("model_name") or "unknown").strip() or "unknown"
        entry = by_model.setdefault(model, {
            "model_name": model, "requests": 0, "tokens_input": 0,
            "tokens_output": 0, "durations": [], "error_count": 0,
        })
        entry["requests"] += 1
        entry["tokens_input"] += int(r.get("tokens_input") or 0)
        entry["tokens_output"] += int(r.get("tokens_output") or 0)
        dur = float(r.get("duration_seconds") or 0)
        if dur > 0:
            entry["durations"].append(dur)
        if r.get("status") == "failed":
            entry["error_count"] += 1

    def _percentile(sorted_vals: list[float], pct: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = int(len(sorted_vals) * pct)
        idx = min(idx, len(sorted_vals) - 1)
        return round(sorted_vals[idx], 2)

    by_model_list = []
    for entry in sorted(by_model.values(), key=lambda x: x["requests"], reverse=True):
        avg_tokens = (entry["tokens_input"] + entry["tokens_output"]) // max(entry["requests"], 1)
        durations = sorted(entry["durations"])
        avg_dur = round(sum(durations) / len(durations), 2) if durations else 0.0
        by_model_list.append({
            "model_name": entry["model_name"],
            "requests": entry["requests"],
            "tokens_input": entry["tokens_input"],
            "tokens_output": entry["tokens_output"],
            "avg_tokens_per_request": avg_tokens,
            "avg_duration_seconds": avg_dur,
            "p50_duration_seconds": _percentile(durations, 0.50),
            "p95_duration_seconds": _percentile(durations, 0.95),
            "p99_duration_seconds": _percentile(durations, 0.99),
            "time_to_first_token_ms": 0,  # not tracked yet
            "error_count": entry["error_count"],
            "error_rate": round(entry["error_count"] / max(entry["requests"], 1), 4),
        })

    by_day: dict = {}
    for m in messages:
        day = (m.get("CreatedAt") or "")[:10]
        if not day:
            continue
        d = by_day.setdefault(day, {"date": day, "requests": 0, "tokens_input": 0, "tokens_output": 0, "errors": 0})
        d["requests"] += 1
        d["tokens_input"] += int(m.get("tokens_input") or 0)
        d["tokens_output"] += int(m.get("tokens_output") or 0)
    for r in runs:
        day = (r.get("CreatedAt") or "")[:10]
        if not day:
            continue
        d = by_day.setdefault(day, {"date": day, "requests": 0, "tokens_input": 0, "tokens_output": 0, "errors": 0})
        d["requests"] += 1
        d["tokens_input"] += int(r.get("tokens_input") or 0)
        d["tokens_output"] += int(r.get("tokens_output") or 0)
        if r.get("status") == "failed":
            d["errors"] += 1
    by_day_list = sorted(by_day.values(), key=lambda x: x["date"])

    by_hour: dict = {}
    for m in messages:
        ts = m.get("CreatedAt") or ""
        if len(ts) < 13:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            key = (dt.hour, dt.isoweekday() % 7)  # 0=Sun
            bucket = by_hour.setdefault(key, {"hour": dt.hour, "day_of_week": dt.isoweekday() % 7, "requests": 0})
            bucket["requests"] += 1
        except (ValueError, AttributeError):
            continue
    by_hour_list = sorted(by_hour.values(), key=lambda x: (x["day_of_week"], x["hour"]))

    by_style: dict = {}
    for m in messages:
        style = (m.get("response_style") or "").strip() or "default"
        by_style[style] = by_style.get(style, 0) + 1
    by_style_list = [{"style": k, "requests": v} for k, v in sorted(by_style.items(), key=lambda x: -x[1])]

    convos: dict = {}
    for m in messages:
        cid = m.get("conversation_id")
        if not cid:
            continue
        c = convos.setdefault(cid, {"conversation_id": cid, "title": "", "message_count": 0, "total_tokens": 0, "last_active": ""})
        c["message_count"] += 1
        c["total_tokens"] += int(m.get("tokens_input") or 0) + int(m.get("tokens_output") or 0)
        ts = m.get("CreatedAt") or ""
        if ts > c["last_active"]:
            c["last_active"] = ts
    top_conversations = sorted(convos.values(), key=lambda x: x["message_count"], reverse=True)[:10]
    if top_conversations:
        try:
            conv_rows = db._get("conversations", params={
                "where": "~or".join(f"(Id,eq,{c['conversation_id']})" for c in top_conversations),
                "limit": 10,
            }).get("list", [])
            title_map = {r["Id"]: r.get("title") or "" for r in conv_rows}
            for c in top_conversations:
                c["title"] = title_map.get(c["conversation_id"], "")
        except Exception:
            pass

    successful_runs = [r for r in runs if r.get("status") == "complete"]
    by_agent: dict = {}
    for r in runs:
        name = r.get("agent_name") or "unknown"
        a = by_agent.setdefault(name, {"agent_name": name, "runs": 0, "successful": 0, "total_steps": 0})
        a["runs"] += 1
        if r.get("status") == "complete":
            a["successful"] += 1
        a["total_steps"] += int(r.get("steps") or 0)
    agent_runs_section = {
        "total_runs": len(runs),
        "successful": len(successful_runs),
        "failed": len(failed_runs),
        "avg_steps": round(sum(int(r.get("steps") or 0) for r in runs) / max(len(runs), 1), 1),
        "by_agent": [
            {
                "agent_name": a["agent_name"],
                "runs": a["runs"],
                "success_rate": round(a["successful"] / max(a["runs"], 1), 3),
                "avg_steps": round(a["total_steps"] / max(a["runs"], 1), 1),
            }
            for a in sorted(by_agent.values(), key=lambda x: x["runs"], reverse=True)
        ],
    }

    return {
        "total_requests": total_requests,
        "total_tokens_input": total_tokens_in,
        "total_tokens_output": total_tokens_out,
        "total_conversations": total_conversations,
        "total_errors": total_errors,
        "period_start": period_start,
        "period_end": period_end,
        "by_model": by_model_list,
        "by_day": by_day_list,
        "by_hour": by_hour_list,
        "by_style": by_style_list,
        "top_conversations": top_conversations,
        "agent_runs": agent_runs_section,
    }


@router.get("/graph/snapshot")
def graph_snapshot(org_id: int, limit: int = 20):
    from infra.graph import get_graph
    try:
        g = get_graph(org_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        node_result = g.query(
            "MATCH (n) OPTIONAL MATCH (n)-[r]-() "
            "WITH n, labels(n)[0] AS lbl, count(r) AS deg "
            "RETURN n.name, lbl, deg ORDER BY deg DESC LIMIT $lim",
            {"lim": limit},
        )
    except Exception:
        node_result = type("R", (), {"result_set": []})()

    nodes = []
    for row in node_result.result_set:
        if not row or not row[0]:
            continue
        nodes.append({
            "id": row[0],
            "label": row[0],
            "type": (row[1] or "unknown").lower(),
            "degree": int(row[2] or 0),
        })

    node_ids = {n["id"] for n in nodes}
    edges = []
    if node_ids:
        try:
            edge_result = g.query(
                "MATCH (a)-[r]->(b) WHERE a.name IN $names OR b.name IN $names "
                "RETURN a.name, type(r), b.name LIMIT $lim",
                {"names": list(node_ids), "lim": limit * 5},
            )
            for row in edge_result.result_set:
                if row and len(row) == 3:
                    edges.append({"source": row[0], "target": row[2], "relation": row[1]})
        except Exception:
            _log.debug("edge query failed", exc_info=True)

    summary = {"total_nodes": 0, "total_edges": 0, "node_types": {}}
    try:
        count_result = g.query("MATCH (n) RETURN labels(n)[0] AS lbl, count(n) AS cnt")
        for row in count_result.result_set:
            lbl = (row[0] or "unknown").lower()
            cnt = int(row[1] or 0)
            summary["node_types"][lbl] = cnt
            summary["total_nodes"] += cnt
    except Exception:
        pass
    try:
        edge_count = g.query("MATCH ()-[r]->() RETURN count(r)")
        summary["total_edges"] = int(edge_count.result_set[0][0] or 0)
    except Exception:
        pass

    return {"nodes": nodes, "edges": edges, "summary": summary}


@router.get("/chroma/snapshot")
def chroma_snapshot(org_id: int):
    from infra.memory import client
    try:
        cols = client.list_collections()
        prefix = f"org_{org_id}_"
        collections = []
        total_docs = 0
        for c in cols:
            if not c.name.startswith(prefix):
                continue
            count = c.count()
            collections.append({"name": c.name, "count": count})
            total_docs += count
        collections.sort(key=lambda x: x["name"])
        return {
            "collections": collections,
            "total_documents": total_docs,
            "total_collections": len(collections),
        }
    except Exception as e:
        _log.error("chroma snapshot failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
