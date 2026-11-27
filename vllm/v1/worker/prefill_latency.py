# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from socket import gethostname
from time import monotonic_ns
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.metrics.prefill_latency import PrefillLatency, worker_trace

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_forward_timer: ContextVar[Callable[[], AbstractContextManager[None]] | None] = (
    ContextVar("prefill_forward_timer", default=None)
)


def trace_model_forward() -> AbstractContextManager[None]:
    timer = _forward_timer.get()
    return nullcontext() if timer is None else timer()


class WorkerPrefillLatency(PrefillLatency):
    def __init__(self) -> None:
        super().__init__()
        self.host = gethostname()
        self.forward_events: list[tuple[Any, Any]] = []
        self.sampling_pending = False
        self.npu_task_profiler = envs.VLLM_PREFILL_LATENCY_NPU_PROFILE

    @contextmanager
    def execute(
        self, worker: "WorkerBase", output: "SchedulerOutput"
    ) -> Iterator[None]:
        request_ids = output.prefill_latency_request_ids
        self.sampling_pending = bool(request_ids and output.total_num_scheduled_tokens)
        if not request_ids:
            yield
            return

        received = monotonic_ns()
        dispatch = output.prefill_latency_dispatch
        if dispatch is not None and dispatch[0] == self.host:
            self.add_total("worker_dispatch", received - dispatch[1])

        runner = getattr(worker, "model_runner", None)
        forward = getattr(runner, "_model_forward", None)
        device_type = getattr(getattr(worker, "device", None), "type", "")
        npu = getattr(torch, "npu", None) if device_type == "npu" else None
        event_cls = getattr(npu, "Event", None)

        first_forward = True

        @contextmanager
        def time_forward() -> Iterator[None]:
            nonlocal first_forward
            start = monotonic_ns()
            if first_forward:
                self.add_total("worker_before_forward", start - execute_start)
                first_forward = False
            events = None
            if event_cls is not None:
                try:
                    first = event_cls(enable_timing=True)
                    last = event_cls(enable_timing=True)
                    first.record()
                    events = first, last
                except Exception:
                    logger.warning_once("Prefill latency: cannot start device event")
            yield
            self.add_total("forward_host", monotonic_ns() - start)
            if events is not None:
                try:
                    events[1].record()
                    self.forward_events.append(events)
                except Exception:
                    logger.warning_once("Prefill latency: cannot finish device event")

        def timed_forward(*args: Any, **kwargs: Any) -> Any:
            with time_forward():
                return forward(*args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(worker_trace(self, request_ids))
            if self.npu_task_profiler and device_type == "npu":
                from vllm.v1.metrics.prefill_device import trace_npu_tasks

                stack.enter_context(trace_npu_tasks(self))
            if callable(forward):
                stack.enter_context(
                    patch.object(runner, "_model_forward", timed_forward)
                )
            else:
                stack.callback(_forward_timer.reset, _forward_timer.set(time_forward))
            execute_start = monotonic_ns()
            yield
            self.add_total("worker_execute_host", monotonic_ns() - execute_start)
        self.collect_forward_events(wait=False)

    @contextmanager
    def sample(self) -> Iterator[None]:
        if not self.sampling_pending:
            yield
            return
        self.sampling_pending = False
        start = monotonic_ns()
        yield
        self.add_total("sampling_host", monotonic_ns() - start)

    def collect_forward_events(self, wait: bool) -> None:
        pending = []
        for first, last in self.forward_events:
            try:
                if wait:
                    last.synchronize()
                elif not last.query():
                    pending.append((first, last))
                    continue
                self.add_total("npu_forward", round(first.elapsed_time(last) * 1e6))
            except Exception:
                logger.warning_once("Prefill latency: cannot read device event")
        self.forward_events = pending

    def snapshot(self) -> dict[str, Any]:
        self.collect_forward_events(wait=True)
        return super().snapshot()
