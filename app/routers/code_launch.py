from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from shared.jobs import STORE, run_in_background, stream_events
from workers.code.agent import CodeAgent


def start_code_job(
    *,
    org_id: int,
    model: str,
    message: str,
    mode: str,
    approved_plan: str | None,
    files: list[dict] | None,
    conversation_id: int | None,
    title: str | None,
    codebase_collection: str | None,
    response_style: str | None,
    knowledge_enabled: bool | None,
    search_enabled: bool,
    temperature: float,
    max_tokens: int,
    project_id: int | None,
    interactive_fs: bool,
) -> dict:
    try:
        agent = CodeAgent(
            model=model,
            org_id=org_id,
            mode=mode,  # type: ignore[arg-type]
            approved_plan=approved_plan,
            files=files,
            search_enabled=search_enabled,
            project_id=project_id,
            interactive_fs=interactive_fs,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = STORE.create()
    run_in_background(
        job,
        lambda j: agent.run_job(
            j,
            user_message=message,
            conversation_id=conversation_id,
            temperature=temperature,
            max_tokens=max_tokens,
            title=title,
            codebase_collection=codebase_collection,
            response_style=response_style,
            knowledge_enabled=knowledge_enabled,
        ),
    )
    return {"job_id": job.id}


def stream_job_events_response(job_id: str, cursor: int = 0) -> StreamingResponse:
    return StreamingResponse(stream_events(job_id, cursor), media_type="text/event-stream")


