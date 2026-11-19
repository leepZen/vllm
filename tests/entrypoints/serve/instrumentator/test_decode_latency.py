# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
from threading import current_thread
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.instrumentator.decode_latency import attach_router
from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware
from vllm.v1.metrics import decode_latency as trace
from vllm.v1.metrics.decode_worker import (
    DeviceTimings,
    install_connector_trace,
    install_runner_trace,
)


def test_ingress_and_summary_use_the_serving_app_and_its_api_key(tmp_path):
    app = FastAPI()
    attach_router(app, str(tmp_path))
    app.add_middleware(AuthenticationMiddleware, tokens=["bench-key"])

    @app.post("/v1/completions")
    async def generation():
        return trace.ingress.get()

    with TestClient(app) as client:
        assert client.get("/v1/decode-latency").status_code == 401
        assert client.post("/v1/decode-latency/reset").status_code == 401
        client.headers.update({"Authorization": "Bearer bench-key"})
        domain, start = client.post("/v1/completions").json()
        assert domain == trace.clock_id() and start > 0
        assert client.get("/v1/decode-latency?view=events").status_code == 422
        assert client.get("/v1/decode-latency?view=requests").status_code == 422
        assert (
            client.get("/v1/decode-latency?view=summary").json()["request_count"] == 0
        )


def test_summary_merges_live_process_writers_and_resets_without_growing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(trace, "DIRECTORY", str(tmp_path))
    app = FastAPI()
    attach_router(app, str(tmp_path))
    code = """
import sys
from vllm.v1.metrics.decode_latency import observe
for line in sys.stdin:
    observe(completed_request=True, ttft_ms=float(line), schedule_ms=1.0)
    print("ok", flush=True)
"""
    with (
        subprocess.Popen(
            [sys.executable, "-c", code],
            env={**os.environ, "VLLM_DECODE_LATENCY_DIR": str(tmp_path)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        ) as child,
        TestClient(app) as client,
    ):
        assert child.stdin is not None and child.stdout is not None
        try:
            child.stdin.write("10\n")
            child.stdin.flush()
            assert child.stdout.readline().strip() == "ok"
            trace.observe(completed_request=True, ttft_ms=20, schedule_ms=2)
            summary = client.get("/v1/decode-latency").json()
            assert summary["request_count"] == 2
            assert summary["latencies"]["ttft_ms"]["avg_ms"] == 15
            assert summary["components"]["schedule_ms"]["avg_ms"] == 1.5
            shards = {p.name: p.stat().st_size for p in tmp_path.glob("*.bin")}
            assert len(shards) == 2

            assert client.post("/v1/decode-latency/reset").json() == {"reset": True}
            summary = client.get("/v1/decode-latency").json()
            assert summary["request_count"] == 0
            assert summary["latencies"]["ttft_ms"]["count"] == 0
            assert summary["components"]["schedule_ms"]["count"] == 0

            child.stdin.write("30\n")
            child.stdin.flush()
            assert child.stdout.readline().strip() == "ok"
            summary = client.get("/v1/decode-latency").json()
            assert summary["request_count"] == 1
            assert summary["latencies"]["ttft_ms"]["avg_ms"] == 30
            assert summary["components"]["schedule_ms"]["count"] == 1
            assert shards == {p.name: p.stat().st_size for p in tmp_path.glob("*.bin")}
            assert not list(tmp_path.glob("*.jsonl"))
        finally:
            child.stdin.close()


def test_histogram_percentiles_use_bucket_limits_and_ignore_missing_timings(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(trace, "DIRECTORY", str(tmp_path))
    for value in range(1, 21):
        trace.observe(completed_request=True, ttft_ms=value, decode_ms=0, tpot_ms=None)
    trace.observe(completed_request=True)
    summary = trace.summarize(str(tmp_path))
    assert summary["request_count"] == 21
    latencies = summary["latencies"]
    assert latencies["ttft_ms"]["count"] == 20
    assert latencies["ttft_ms"]["avg_ms"] == 10.5
    assert 18 <= latencies["ttft_ms"]["p90_ms"] < 18 * 1.02
    assert 20 <= latencies["ttft_ms"]["p99_ms"] < 20 * 1.02
    assert latencies["decode_ms"] == {
        "count": 20,
        "avg_ms": 0,
        "p90_ms": 0,
        "p99_ms": 0,
    }
    assert latencies["tpot_ms"] == {
        "count": 0,
        "avg_ms": None,
        "p90_ms": None,
        "p99_ms": None,
    }


def test_queued_sampling_ignores_load_only_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "DIRECTORY", str(tmp_path))
    runner = SimpleNamespace(execute_model=lambda _: None, sample_tokens=lambda x: x)
    install_runner_trace(runner)
    for dispatch, tokens in ((1, 1), (2, 0), (3, 1)):
        runner.execute_model(
            SimpleNamespace(
                latency_dispatch=(trace.clock_id(), dispatch),
                latency_has_kv_load=not tokens,
                total_num_scheduled_tokens=tokens,
            )
        )
    for value in range(3):
        assert runner.sample_tokens(value) == value
    assert (
        trace.summarize(str(tmp_path))["components"]["sample_tokens_host_ms"]["count"]
        == 2
    )


def test_device_timings_aggregate_completed_work_off_the_runner_thread(
    tmp_path, monkeypatch
):
    inference_thread = current_thread()

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self, stream):
            assert current_thread() is inference_thread

        def synchronize(self):
            assert current_thread() is not inference_thread

        def elapsed_time(self, end):
            return 2.5

    api = SimpleNamespace(
        Event=Event,
        current_device=lambda: 0,
        set_device=lambda _: None,
        current_stream=lambda: 1,
        is_current_stream_capturing=lambda: False,
    )
    monkeypatch.setattr(trace, "DIRECTORY", str(tmp_path))
    reader = DeviceTimings(api)
    for dispatch in (1, 2):
        with reader.measure("forward", dispatch):
            pass
    reader.close()
    components = trace.summarize(str(tmp_path))["components"]
    assert components["npu_forward_ms"]["count"] == 2
    assert components["npu_forward_ms"]["avg_ms"] == 2.5
    assert components["npu_stream_gap_ms"]["count"] == 1


def test_external_retrieve_updates_histogram_and_preserves_result(
    tmp_path, monkeypatch
):
    hits = object()
    backend = SimpleNamespace(retrieve=lambda **kwargs: hits)
    connector = SimpleNamespace(
        _lmcache_engine=SimpleNamespace(lmcache_engine=backend),
        start_load_kv=lambda: backend.retrieve(req_id="r"),
        wait_for_layer_load=lambda layer: None,
    )
    runner = SimpleNamespace(execute_model=lambda batch: connector.start_load_kv())
    monkeypatch.setattr(trace, "DIRECTORY", str(tmp_path))
    install_connector_trace(connector)
    install_runner_trace(runner)
    batch = SimpleNamespace(
        latency_dispatch=(trace.clock_id(), 1),
        latency_has_kv_load=True,
    )
    assert runner.execute_model(batch) is hits
    components = trace.summarize(str(tmp_path))["components"]
    assert components["kv_load_host_ms"]["count"] == 1
    assert components["kv_retrieve_host_ms"]["count"] == 1
