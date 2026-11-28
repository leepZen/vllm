# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in prefill spans retained until the next measurement reset."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from socket import gethostname
from time import monotonic_ns
from typing import Any, Literal


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
                req_id: {
                    name: [span.copy() for span in spans]
                    for name, spans in segments.items()
                }
                for req_id, segments in self.requests.items()
            },
            "shared_totals_ns": self.totals.copy(),
        }


class SchedulerPrefillLatency(PrefillLatency):
    def __init__(self) -> None:
        super().__init__()
        # Scheduler enqueue and prefill start, respectively.
        self.queued: dict[str, tuple[int, int]] = {}
        self.checked: dict[str, int] = {}
        self.lookup_started: dict[str, int] = {}
        self.load_started: dict[str, int] = {}
        self.first_scheduled: dict[str, int] = {}

    def enqueue(self, request_id: str) -> None:
        now = monotonic_ns()
        self.queued[request_id] = (now, now)
        self.requests.setdefault(request_id, {})

    def record_input_queue(self, request_id: str, start_ns: int, end_ns: int) -> None:
        self.record(request_id, "input_queue", start_ns, end_ns)
        if (queued := self.queued.get(request_id)) is not None:
            scheduler_enqueue_ns, prefill_start_ns = queued
            # Later ADD messages for a streaming session do not start its prefill.
            if end_ns <= scheduler_enqueue_ns:
                self.queued[request_id] = (
                    scheduler_enqueue_ns,
                    min(prefill_start_ns, start_ns),
                )

    def check(self, request_id: str) -> None:
        queued = self.queued.get(request_id)
        if queued is not None and request_id not in self.checked:
            scheduler_enqueue_ns, _ = queued
            now = monotonic_ns()
            self.checked[request_id] = now
            self.record(request_id, "waiting", scheduler_enqueue_ns, now)

    def schedule(self, request_id: str) -> None:
        if request_id in self.checked and request_id not in self.first_scheduled:
            now = monotonic_ns()
            self.first_scheduled[request_id] = now
            self.record(request_id, "scheduling", self.checked[request_id], now)

    def record_cache_lookup(
        self,
        request_id: str,
        start: int,
        end: int,
        *,
        source: Literal["local", "external"],
        matched_tokens: int | None = None,
    ) -> None:
        """Aggregate calls by source while preserving the first-to-last span."""
        lookups = self.requests[request_id].setdefault("cache_lookup", [])
        for lookup in lookups:
            if lookup["source"] == source:
                break
        else:
            lookup = {
                "source": source,
                "start_ns": start,
                "calls": 0,
                "total_call_ns": 0,
            }
            lookups.append(lookup)

        lookup["end_ns"] = end
        lookup["duration_ns"] = end - lookup["start_ns"]
        lookup["calls"] += 1
        lookup["total_call_ns"] += end - start

        if source == "external":
            lookup["matched_tokens"] = matched_tokens
            lookup["pending_count"] = lookup.get("pending_count", 0)
            if matched_tokens is None:
                lookup["pending_count"] += 1
                self.lookup_started.setdefault(request_id, start)
            else:
                result_start = self.lookup_started.pop(request_id, start)
                if "first_result_call" not in lookup:
                    lookup["first_result_call"] = lookup["calls"]
                    lookup["later_result_count"] = 0
                    lookup["later_result_total_ns"] = 0
                    self.record(
                        request_id, "external_lookup_elapsed", result_start, end
                    )
                else:
                    lookup["later_result_count"] += 1
                    lookup["later_result_total_ns"] += end - result_start

    def finish(self, request_id: str, result_ready_ns: int | None = None) -> None:
        queued = self.queued.pop(request_id, None)
        self.checked.pop(request_id, None)
        self.lookup_started.pop(request_id, None)
        self.load_started.pop(request_id, None)
        self.first_scheduled.pop(request_id, None)
        if queued is not None and result_ready_ns is not None:
            _, prefill_start_ns = queued
            now = monotonic_ns()
            self.record(
                request_id,
                "first_token",
                result_ready_ns,
                now,
                boundary="engine_result_to_first_token",
            )
            self.record(request_id, "prefill_total", prefill_start_ns, now)

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
