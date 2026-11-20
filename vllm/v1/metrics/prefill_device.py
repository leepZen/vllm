# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device measurements for the opt-in prefill latency collector."""

import csv
import json
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm.logger import init_logger
from vllm.v1.metrics.prefill_latency import PrefillLatency

logger = init_logger(__name__)


def _active_ns(intervals: list[tuple[Decimal, Decimal]]) -> int:
    end = Decimal(0)
    elapsed = Decimal(0)
    for start, stop in sorted(intervals):
        uncovered_start = max(start, end)
        if stop > uncovered_start:
            elapsed += stop - uncovered_start
            end = stop
    return int(elapsed * 1000)


def read_npu_profile(directory: Path) -> dict[str, int]:
    """Sum device activity in one Ascend worker's profile window."""
    h2d: list[tuple[Decimal, Decimal]] = []
    ai_core: list[tuple[Decimal, Decimal]] = []
    for path in directory.rglob("kernel_details.csv"):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                core = row.get("Accelerator Core", row.get("Task Type", ""))
                if core.upper() not in {"AI_CORE", "AICORE", "MIX_AIC", "MIX_AICORE"}:
                    continue
                try:
                    start = Decimal(
                        row.get("Start Time(us)") or row["Task Start Time(us)"]
                    )
                    duration = Decimal(
                        row.get("Duration(us)") or row["Task Duration(us)"]
                    )
                    ai_core.append((start, start + duration))
                except (InvalidOperation, KeyError, TypeError):
                    continue

    for path in directory.rglob("trace_view.json"):
        with path.open(encoding="utf-8") as handle:
            events = json.load(handle, parse_float=Decimal)
        if isinstance(events, dict):
            events = events["traceEvents"]
        for event in events:
            args = event.get("args") or {}
            operation = "".join(
                c for c in str(args.get("operation", "")).lower() if c.isalnum()
            )
            if operation not in {"hosttodevice", "h2d", "htod"}:
                continue
            # CANN reports the actual device copy interval in microseconds.
            try:
                if "Start Time" in args and "End Time" in args:
                    h2d.append((Decimal(args["Start Time"]), Decimal(args["End Time"])))
                elif event.get("ph") == "X" and "dur" in event:
                    start = Decimal(event["ts"])
                    h2d.append((start, start + Decimal(event["dur"])))
            except (InvalidOperation, KeyError, TypeError):
                continue
    return {
        name: _active_ns(intervals)
        for name, intervals in (("ai_core", ai_core), ("h2d", h2d))
        if intervals
    }


@contextmanager
def trace_npu_tasks(recorder: PrefillLatency) -> Iterator[None]:
    """Record expensive CANN task profiling only when explicitly requested."""
    try:
        from torch_npu import profiler
    except ImportError:
        logger.warning_once("Prefill latency: torch_npu profiler is unavailable")
        yield
        return

    with TemporaryDirectory(prefix="vllm-prefill-") as directory:
        try:
            profile = profiler.profile(
                activities=[
                    profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.NPU,
                ],
                schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                experimental_config=profiler._ExperimentalConfig(
                    profiler_level=profiler.ProfilerLevel.Level1,
                    aic_metrics=profiler.AiCMetrics.AiCoreNone,
                ),
                on_trace_ready=profiler.tensorboard_trace_handler(directory),
            )
            profile.start()
        except Exception:
            logger.exception("Could not start NPU profiling for prefill latency")
            yield
            return
        try:
            yield
        finally:
            try:
                profile.stop()
                for segment, duration in read_npu_profile(Path(directory)).items():
                    recorder.add_total(segment, duration)
            except Exception:
                logger.exception("Could not read NPU profiling for prefill latency")
