# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-size latency histograms shared by local vLLM processes."""

import logging
import os
import socket
from bisect import bisect_left
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from math import inf, isfinite
from mmap import mmap
from pathlib import Path
from struct import Struct
from tempfile import mkstemp
from threading import Lock
from typing import BinaryIO

from filelock import FileLock

from vllm import envs

DIRECTORY = envs.VLLM_DECODE_LATENCY_DIR
ingress: ContextVar[tuple[str, int] | None] = ContextVar("decode_ingress", default=None)
logger = logging.getLogger(__name__)

_REQUEST_LATENCIES = (
    "ingress_ms",
    "queue_ms",
    "admission_ms",
    "first_compute_to_token_ms",
    "decode_ms",
    "ttft_ms",
    "total_ms",
    "tpot_ms",
    "lookup_call_ms",
    "lookup_to_result_ms",
    "async_kv_wait_ms",
)
_COMPONENT_LATENCIES = (
    "schedule_ms",
    "schedule_thread_cpu_ms",
    "result_processing_ms",
    "dispatch_to_worker_ms",
    "worker_execute_ms",
    "prepare_inputs_host_ms",
    "forward_host_ms",
    "sample_tokens_host_ms",
    "kv_load_host_ms",
    "kv_wait_for_layer_host_ms",
    "kv_retrieve_host_ms",
    "npu_forward_ms",
    "npu_sample_tokens_ms",
    "npu_kv_wait_for_layer_ms",
    "npu_stream_gap_ms",
)

# Changing fields or buckets requires a new shard filename version.
_SHARD_PREFIX = "decode-hist-v1-"
_BOUNDS_MS = (0.0, *(0.001 * 1.02**i for i in range(1050)), inf)
_COUNT = Struct("<Q")
_SUM = Struct("<d")
_HISTOGRAM = Struct(f"<d{len(_BOUNDS_MS)}Q")
_OFFSETS = {
    field: _COUNT.size + index * _HISTOGRAM.size
    for index, field in enumerate((*_REQUEST_LATENCIES, *_COMPONENT_LATENCIES))
}
_SHARD_SIZE = _COUNT.size + len(_OFFSETS) * _HISTOGRAM.size


@lru_cache(maxsize=1)
def clock_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return socket.gethostname()


def _directory_lock(folder: Path) -> FileLock:
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    return FileLock(folder / "decode-hist-v1.lock", mode=0o600)


@contextmanager
def _lock_shard(file: BinaryIO, *, write: bool):
    import fcntl

    fcntl.flock(file, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
    try:
        yield
    finally:
        fcntl.flock(file, fcntl.LOCK_UN)


class _HistogramShard:
    def __init__(self, directory: str):
        folder = Path(directory)
        self.lock = Lock()
        self.failed = False
        with _directory_lock(folder):
            fd, path = mkstemp(prefix=_SHARD_PREFIX, suffix=".bin", dir=folder)
            self.file = os.fdopen(fd, "r+b", buffering=0)
            try:
                self.file.truncate(_SHARD_SIZE)
                self.data = mmap(fd, _SHARD_SIZE)
            except OSError:
                self.file.close()
                Path(path).unlink()
                raise

    def observe(self, completed_request: bool, values: dict[str, float | None]) -> None:
        with self.lock, _lock_shard(self.file, write=True):
            if completed_request:
                self._increment(0)
            for field, value in values.items():
                if value is None or value < 0 or not isfinite(value):
                    continue
                offset = _OFFSETS[field]
                (total,) = _SUM.unpack_from(self.data, offset)
                _SUM.pack_into(self.data, offset, total + value)
                bucket = bisect_left(_BOUNDS_MS, value)
                self._increment(offset + _SUM.size + bucket * _COUNT.size)

    def _increment(self, offset: int) -> None:
        (count,) = _COUNT.unpack_from(self.data, offset)
        _COUNT.pack_into(self.data, offset, count + 1)


@lru_cache(maxsize=1)
def _shard(pid: int, directory: str) -> _HistogramShard | None:
    # A forked child owns its own shard and lock.
    try:
        return _HistogramShard(directory)
    except OSError:
        logger.exception("Unable to initialize decode latency histograms")
        return None


def observe(*, completed_request: bool = False, **values: float | None) -> None:
    if not DIRECTORY or (shard := _shard(os.getpid(), DIRECTORY)) is None:
        return
    if not shard.failed:
        try:
            shard.observe(completed_request, values)
        except OSError:
            shard.failed = True
            logger.exception("Disabling decode latency updates in this process")


def _shards(folder: Path):
    return (
        path
        for path in folder.glob(f"{_SHARD_PREFIX}*.bin")
        if not path.is_symlink() and path.is_file()
    )


def _percentile(buckets: list[int], rank: int) -> float | None:
    cumulative = 0
    for bound, count in zip(_BOUNDS_MS, buckets):
        cumulative += count
        if cumulative >= rank:
            return bound if isfinite(bound) else None
    return None


def summarize(directory: str) -> dict:
    """Merge cumulative histograms without opening per-request or event logs."""
    folder = Path(directory)
    request_count = 0
    sums = dict.fromkeys(_OFFSETS, 0.0)
    histograms = {field: [0] * len(_BOUNDS_MS) for field in _OFFSETS}
    with _directory_lock(folder):
        for path in _shards(folder):
            with path.open("rb") as file, _lock_shard(file, write=False):
                data = file.read()
            if len(data) != _SHARD_SIZE:
                continue
            request_count += _COUNT.unpack_from(data)[0]
            for field, offset in _OFFSETS.items():
                total, *buckets = _HISTOGRAM.unpack_from(data, offset)
                sums[field] += total
                histograms[field] = [
                    left + right for left, right in zip(histograms[field], buckets)
                ]

    stats = {}
    for field, buckets in histograms.items():
        count = sum(buckets)
        stats[field] = {
            "count": count,
            "avg_ms": sums[field] / count if count else None,
            "p90_ms": _percentile(buckets, (90 * count + 99) // 100) if count else None,
            "p99_ms": _percentile(buckets, (99 * count + 99) // 100) if count else None,
        }
    return {
        "request_count": request_count,
        "latencies": {field: stats[field] for field in _REQUEST_LATENCIES},
        "components": {field: stats[field] for field in _COMPONENT_LATENCIES},
    }


def reset_summary(directory: str) -> None:
    folder = Path(directory)
    zeros = bytes(_SHARD_SIZE)
    with _directory_lock(folder):
        for path in _shards(folder):
            with path.open("r+b", buffering=0) as file, _lock_shard(file, write=True):
                if file.write(zeros) != _SHARD_SIZE:
                    raise OSError("Incomplete decode latency reset")


def _ms(start: int | None, end: int | None) -> float | None:
    return None if start is None or end is None else (end - start) / 1e6


@dataclass
class RequestTiming:
    ingress_ns: int | None = None
    queued_ns: int | None = None
    checked_ns: int | None = None
    admitted_ns: int | None = None
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    output_tokens: int = 0
    lookup_call_ns: int = 0
    lookup_start_ns: int | None = None
    lookup_result_ns: int | None = None
    kv_wait_start_ns: int | None = None
    kv_wait_ns: int | None = None

    def lookup(self, start: int, end: int, result: int | None) -> None:
        self.lookup_call_ns += end - start
        if self.lookup_start_ns is None:
            self.lookup_start_ns = start
        if result is not None and self.lookup_result_ns is None:
            self.lookup_result_ns = end

    def finish_kv_wait(self, now: int) -> None:
        if self.kv_wait_start_ns is not None:
            self.kv_wait_ns = (self.kv_wait_ns or 0) + now - self.kv_wait_start_ns
            self.kv_wait_start_ns = None

    def output(self, count: int, now: int) -> None:
        if self.first_token_ns is None:
            self.first_token_ns = now
        self.last_token_ns = now
        self.output_tokens += count

    def finish(self) -> None:
        decode = _ms(self.first_token_ns, self.last_token_ns)
        observe(
            completed_request=True,
            ingress_ms=_ms(self.ingress_ns, self.queued_ns),
            queue_ms=_ms(self.queued_ns, self.checked_ns),
            admission_ms=_ms(self.checked_ns, self.admitted_ns),
            first_compute_to_token_ms=_ms(self.admitted_ns, self.first_token_ns),
            decode_ms=decode,
            ttft_ms=_ms(self.ingress_ns, self.first_token_ns),
            total_ms=_ms(self.ingress_ns, self.last_token_ns),
            tpot_ms=(
                decode / (self.output_tokens - 1)
                if decode is not None and self.output_tokens > 1
                else None
            ),
            lookup_call_ms=_ms(0, self.lookup_call_ns)
            if self.lookup_start_ns is not None
            else None,
            lookup_to_result_ms=_ms(self.lookup_start_ns, self.lookup_result_ns),
            async_kv_wait_ms=_ms(0, self.kv_wait_ns),
        )
