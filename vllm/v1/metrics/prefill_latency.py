# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in prefill spans retained until the next measurement reset."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from socket import gethostname
from time import monotonic_ns
from typing import Any


def interval(start: int, end: int) -> dict[str, int]:
    return {"start_ns": start, "end_ns": end, "duration_ns": end - start}


class PrefillLatency:
    def __init__(self) -> None:
        self.requests: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self.totals: dict[str, int] = {}

    def record(
        self, request_id: str, segment: str, start: int, end: int, **details: Any
    ) -> None:
        spans = self.requests.setdefault(request_id, {})
        spans.setdefault(segment, []).append({**interval(start, end), **details})

    def add_total(self, segment: str, duration_ns: int) -> None:
        self.totals[segment] = self.totals.get(segment, 0) + duration_ns

    def snapshot(self) -> dict[str, Any]:
        return {
            "host": gethostname(),
            "requests": {
                req_id: {name: spans.copy() for name, spans in segments.items()}
                for req_id, segments in self.requests.items()
            },
            "shared_totals_ns": self.totals.copy(),
        }


class SchedulerPrefillLatency(PrefillLatency):
    def __init__(self) -> None:
        super().__init__()
        self.queued: dict[str, int] = {}
        self.checked: dict[str, int] = {}
        self.lookup_started: dict[str, int] = {}
        self.load_started: dict[str, int] = {}
        self.first_scheduled: dict[str, int] = {}

    def enqueue(self, request_id: str) -> None:
        self.queued[request_id] = monotonic_ns()
        self.requests.setdefault(request_id, {})

    def check(self, request_id: str) -> None:
        if request_id in self.queued and request_id not in self.checked:
            now = monotonic_ns()
            self.checked[request_id] = now
            self.record(request_id, "waiting", self.queued[request_id], now)

    def schedule(self, request_id: str) -> None:
        if request_id in self.checked and request_id not in self.first_scheduled:
            now = monotonic_ns()
            self.first_scheduled[request_id] = now
            self.record(request_id, "scheduling", self.checked[request_id], now)

    def finish(self, request_id: str, result_ready_ns: int | None = None) -> None:
        queued = self.queued.pop(request_id, None)
        self.checked.pop(request_id, None)
        self.lookup_started.pop(request_id, None)
        self.load_started.pop(request_id, None)
        self.first_scheduled.pop(request_id, None)
        if queued is not None and result_ready_ns is not None:
            now = monotonic_ns()
            self.record(
                request_id,
                "first_token",
                result_ready_ns,
                now,
                boundary="engine_result_to_first_token",
            )
            self.record(request_id, "prefill_total", queued, now)

    def reset(self) -> None:
        self.requests.clear()
        self.totals.clear()
        self.queued.clear()
        self.checked.clear()
        self.lookup_started.clear()
        self.load_started.clear()
        self.first_scheduled.clear()


_worker_trace: ContextVar[tuple[PrefillLatency, tuple[str, ...]] | None] = ContextVar(
    "worker_prefill_trace", default=None
)


def start_request_kv_load(request_id: str) -> int | None:
    trace = _worker_trace.get()
    return monotonic_ns() if trace is not None and request_id in trace[1] else None


def finish_request_kv_load(request_id: str, start_ns: int, boundary: str) -> None:
    trace = _worker_trace.get()
    if trace is not None and request_id in trace[1]:
        trace[0].record(
            request_id, "load_cache", start_ns, monotonic_ns(), boundary=boundary
        )


@contextmanager
def worker_trace(
    recorder: PrefillLatency, request_ids: tuple[str, ...]
) -> Iterator[None]:
    token = _worker_trace.set((recorder, request_ids))
    try:
        yield
    finally:
        _worker_trace.reset(token)


@contextmanager
def trace_kv_load(boundary: str, request_id: str | None = None) -> Iterator[None]:
    trace = _worker_trace.get()
    if trace is None or (request_id is not None and request_id not in trace[1]):
        yield
        return

    recorder, _ = trace
    start = monotonic_ns()
    yield
    end = monotonic_ns()
    if request_id is None:
        recorder.add_total(boundary, end - start)
    else:
        recorder.record(request_id, "load_cache", start, end, boundary=boundary)
