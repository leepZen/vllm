# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from vllm import envs

if TYPE_CHECKING:
    from vllm.v1.engine.async_llm import AsyncLLM

router = APIRouter(prefix="/v1/prefill_latency")


def _engine(request: Request) -> "AsyncLLM":
    engine = getattr(request.app.state, "engine_client", None)
    if not callable(getattr(engine, "prefill_latency", None)):
        raise HTTPException(status_code=503, detail="The async engine is unavailable")
    return engine


@router.get("")
async def get_prefill_latency(request: Request):
    return {
        "request_clock": "host_monotonic_ns",
        "npu_task_profiling_enabled": envs.VLLM_PREFILL_LATENCY_NPU_PROFILE,
        "engines": await _engine(request).prefill_latency(),
    }


@router.post("/reset")
async def reset_prefill_latency(request: Request):
    await _engine(request).prefill_latency(reset=True)
    return {"reset": True}
