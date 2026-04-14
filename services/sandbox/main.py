from __future__ import annotations

import logging
import os
import subprocess
import tempfile

from fastapi import FastAPI
from pydantic import BaseModel, Field

_log = logging.getLogger("sandbox")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

MAX_TIMEOUT = 120
MAX_STDOUT = 50_000
MAX_STDERR = 10_000

app = FastAPI(title="mst-ag-sandbox")


class ExecRequest(BaseModel):
    language: str = Field(default="python", pattern="^(python|bash)$")
    code: str
    timeout: int = Field(default=30, ge=1, le=MAX_TIMEOUT)
    stdin: str = ""


class ExecResponse(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    error: str | None = None


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "service": "sandbox"}


@app.post("/exec", response_model=ExecResponse)
async def execute(req: ExecRequest) -> ExecResponse:
    suffix = ".py" if req.language == "python" else ".sh"
    script_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, dir="/tmp", delete=False
        ) as f:
            f.write(req.code)
            script_path = f.name

        cmd = ["python", script_path] if req.language == "python" else ["bash", script_path]

        try:
            result = subprocess.run(
                cmd,
                input=req.stdin if req.stdin else None,
                capture_output=True,
                text=True,
                timeout=req.timeout,
                cwd="/tmp",
                env={
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "HOME": "/home/sandbox",
                    "PYTHONPATH": "",
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                },
            )
            return ExecResponse(
                stdout=result.stdout[:MAX_STDOUT],
                stderr=result.stderr[:MAX_STDERR],
                exit_code=result.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return ExecResponse(
                stdout="",
                stderr=f"Execution timed out after {req.timeout}s",
                exit_code=124,
                timed_out=True,
            )
    except Exception as e:
        _log.error("exec failed: %s", e, exc_info=True)
        return ExecResponse(
            stdout="", stderr="", exit_code=1, timed_out=False, error=str(e),
        )
    finally:
        if script_path:
            try:
                os.unlink(script_path)
            except Exception:
                pass
