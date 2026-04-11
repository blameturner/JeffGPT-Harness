from __future__ import annotations

import concurrent.futures
import logging

from rag import retrieve

_log = logging.getLogger("chat.rag_phase")


def submit_rag_future(
    user_message: str,
    org_id: int,
    collection_name: str,
    enabled: bool,
) -> tuple[concurrent.futures.ThreadPoolExecutor | None, concurrent.futures.Future | None]:
    if not enabled:
        return None, None
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="chat-rag"
    )
    future = executor.submit(
        retrieve,
        query=user_message,
        org_id=org_id,
        collection_name=collection_name,
        n_results=10,
        top_k=3,
    )
    return executor, future


def collect_rag(
    rag_future: concurrent.futures.Future | None,
    rag_executor: concurrent.futures.ThreadPoolExecutor | None,
) -> str:
    rag_context = ""
    if rag_future is not None:
        try:
            rag_context = rag_future.result(timeout=45)
        except Exception:
            _log.error("RAG retrieval failed", exc_info=True)
            rag_context = ""
        finally:
            if rag_executor is not None:
                rag_executor.shutdown(wait=False)
    return rag_context


def cancel_rag(
    rag_future: concurrent.futures.Future | None,
    rag_executor: concurrent.futures.ThreadPoolExecutor | None,
) -> None:
    if rag_future is not None:
        rag_future.cancel()
    if rag_executor is not None:
        rag_executor.shutdown(wait=False)
