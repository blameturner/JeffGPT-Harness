from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tools.planned_search.agent import (
    approve_searches,
    get_pending_search,
    get_search_results,
    reject_searches,
)

_log = logging.getLogger("routers.planned_search")

router = APIRouter()


class ApproveRequest(BaseModel):
    org_id: int


@router.post("/{message_id}/approve")
async def approve(message_id: int, request: ApproveRequest):
    result = await approve_searches(message_id, request.org_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@router.post("/{message_id}/reject")
async def reject(message_id: int):
    result = reject_searches(message_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@router.get("/{message_id}")
async def get_pending(message_id: int):
    result = get_pending_search(message_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


@router.get("/{message_id}/results")
async def get_results(message_id: int, org_id: int):
    result = get_search_results(message_id, org_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result
