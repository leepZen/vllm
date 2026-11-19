# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe runner boundaries on the installed worker, including Ascend plugins."""

import logging
import time
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any

from vllm.v1.metrics import decode_latency as trace

logger = logging.getLogger(__name__)
_loads: ContextVar[bool] = ContextVar("decode_kv_loads", default=False)
_dispatch_ns: ContextVar[int | None] = ContextVar("decode_dispatch_ns", default=None)
_device: ContextVar["DeviceTimings | None"] = ContextVar("decode_device", default=None)


class DeviceTimings:
    """Read current-stream event pairs off the inference thread."""

    def __init__(self, device_api: Any):
        self.device_api = device_api
        self.pending: Queue = Queue(maxsize=1024)
        self.previous: dict[Any, tuple[int, Any]] = {}
        self.enabled = True
        self.warned_full = False
        self.closing = Event()
        self.device_index = device_api.current_device()
        self.reader = Thread(target=self._read, daemon=True, name="decode-device-time")
        self.reader.start()

    def _read(self) -> None:
        try:
            self.device_api.set_device(self.device_index)
        except Exception:
            self.disable()
            return
        while True:
            try:
                kind, start, end, previous = self.pending.get(timeout=0.25)
            except Empty:
                if self.closing.is_set():
                    return
                continue
            try:
                end.synchronize()
                values = {f"npu_{kind}_ms": start.elapsed_time(end)}
                if previous:
                    values["npu_stream_gap_ms"] = previous.elapsed_time(start)
                trace.observe(**values)
            except Exception:
                self.disable()

    def close(self) -> None:
        self.closing.set()
        self.reader.join(timeout=2)

    def disable(self) -> None:
        if self.enabled:
            self.enabled = False
            logger.exception("Device events unavailable for decode latency statistics")

    @contextmanager
    def measure(self, kind: str, dispatch_ns: int):
        try:
            capturing = self.device_api.is_current_stream_capturing()
        except Exception:
            self.disable()
            capturing = True
        if capturing:
            yield
            return
        try:
            stream = self.device_api.current_stream()
            start = self.device_api.Event(enable_timing=True)
            end = self.device_api.Event(enable_timing=True)
            start.record(stream)
        except Exception:
            self.disable()
            yield
            return
        try:
            yield
        finally:
            try:
                end.record(stream)
                previous = self.previous.get(stream)
                gap = (
                    previous[1]
                    if kind == "forward" and previous and previous[0] != dispatch_ns
                    else None
                )
                if kind in ("forward", "sample_tokens"):
                    self.previous[stream] = (dispatch_ns, end)
                self.pending.put_nowait((kind, start, end, gap))
            except Full:
                if not self.warned_full:
                    logger.warning("Decode latency device timing queue is full")
                    self.warned_full = True
            except Exception:
                self.disable()


@contextmanager
def _phase(kind: str, device: bool):
    dispatch_ns = _dispatch_ns.get()
    if dispatch_ns is None:
        yield
        return
    start = time.monotonic_ns()
    timings = _device.get()
    try:
        if device and timings and timings.enabled:
            with timings.measure(kind, dispatch_ns):
                yield
        else:
            yield
    finally:
        trace.observe(**{f"{kind}_host_ms": (time.monotonic_ns() - start) / 1e6})


def forward_scope():
    """V2 dispatches eager, microbatch and full-graph forwards separately."""
    return _phase("forward", True) if trace.DIRECTORY else nullcontext()


def install_runner_trace(runner: Any, device_api: Any = None) -> DeviceTimings | None:
    """Wrap host-side runner entry points once, after the runner is built."""
    if getattr(runner, "_decode_latency_installed", False):
        return None
    pending_samples: deque[int] = deque()
    timings = None
    if device_api is not None:
        try:
            timings = DeviceTimings(device_api)
        except Exception:
            logger.exception(
                "Device timing initialization failed, keeping host tracing"
            )

    def wrap_phase(name: str, kind: str, device: bool = False) -> None:
        method = getattr(runner, name, None)
        if not callable(method):
            return

        @wraps(method)
        def wrapped(*args, **kwargs):
            with _phase(kind, device):
                return method(*args, **kwargs)

        setattr(runner, name, wrapped)

    execute = runner.execute_model

    @wraps(execute)
    def execute_model(scheduler_output, *args, **kwargs):
        dispatch = scheduler_output.latency_dispatch
        if dispatch is None:
            return execute(scheduler_output, *args, **kwargs)
        dispatch_clock, dispatch_ns = dispatch
        start = time.monotonic_ns()
        token = _dispatch_ns.set(dispatch_ns)
        loads = _loads.set(scheduler_output.latency_has_kv_load)
        device_token = _device.set(timings)
        try:
            result = execute(scheduler_output, *args, **kwargs)
            if result is None and scheduler_output.total_num_scheduled_tokens:
                pending_samples.append(dispatch_ns)
            return result
        finally:
            trace.observe(
                worker_execute_ms=(time.monotonic_ns() - start) / 1e6,
                dispatch_to_worker_ms=(start - dispatch_ns) / 1e6
                if dispatch_clock == trace.clock_id()
                else None,
            )
            _device.reset(device_token)
            _loads.reset(loads)
            _dispatch_ns.reset(token)

    runner.execute_model = execute_model
    prepare = (
        "_prepare_inputs" if hasattr(runner, "_prepare_inputs") else "prepare_inputs"
    )
    wrap_phase(prepare, "prepare_inputs")
    wrap_phase("_model_forward", "forward", device=True)

    sample: Callable | None = getattr(runner, "sample_tokens", None)
    if sample is not None:
        wrap_phase("sample_tokens", "sample_tokens", device=True)
        timed_sample = runner.sample_tokens

        @wraps(sample)
        def sample_tokens(*args, **kwargs):
            if _dispatch_ns.get() is not None or not pending_samples:
                return timed_sample(*args, **kwargs)
            token = _dispatch_ns.set(pending_samples.popleft())
            device_token = _device.set(timings)
            try:
                return timed_sample(*args, **kwargs)
            finally:
                _device.reset(device_token)
                _dispatch_ns.reset(token)

        runner.sample_tokens = sample_tokens
    runner._decode_latency_installed = True
    return timings


def install_connector_trace(connector: Any) -> None:
    """Time worker API calls for steps which actually requested an external load."""

    def wrap_load(name: str, kind: str, device_timing: bool = False) -> None:
        method = getattr(connector, name)

        @wraps(method)
        def measured(*args, **kwargs):
            if not _loads.get():
                return method(*args, **kwargs)
            with _phase(kind, device_timing):
                return method(*args, **kwargs)

        setattr(connector, name, measured)

    wrap_load("start_load_kv", "kv_load")
    wrap_load("wait_for_layer_load", "kv_wait_for_layer", device_timing=True)

    adapter = getattr(connector, "_lmcache_engine", None)
    backend = getattr(adapter, "lmcache_engine", None)
    retrieve = getattr(backend, "retrieve", None)
    if not callable(retrieve) or getattr(retrieve, "_decode_latency_installed", False):
        return

    @wraps(retrieve)
    def measured_retrieve(*args, **kwargs):
        if not _loads.get():
            return retrieve(*args, **kwargs)
        with _phase("kv_retrieve", device=False):
            return retrieve(*args, **kwargs)

    measured_retrieve._decode_latency_installed = True  # type: ignore[attr-defined]
    backend.retrieve = measured_retrieve
