# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Literal

from fastapi import FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send

from vllm.v1.metrics import decode_latency as trace


class DecodeIngressMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        start = time.monotonic_ns()
        token = trace.ingress.set((trace.clock_id(), start))
        try:
            await self.app(scope, receive, send)
        finally:
            trace.ingress.reset(token)


def attach_router(app: FastAPI, directory: str) -> None:
    app.add_middleware(DecodeIngressMiddleware)

    @app.get("/v1/decode-latency")
    def decode_latency(view: Literal["summary"] = "summary"):
        return trace.summarize(directory)

    @app.post("/v1/decode-latency/reset")
    def reset_decode_latency():
        trace.reset_summary(directory)
        return {"reset": True}
