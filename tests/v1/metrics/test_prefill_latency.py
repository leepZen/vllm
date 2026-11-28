# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.prefill_latency import router
from vllm.v1.metrics.prefill_device import read_npu_profile
from vllm.v1.metrics.prefill_latency import SchedulerPrefillLatency


def test_prefill_total_starts_at_engine_input_queue():
    recorder = SchedulerPrefillLatency()
    with patch(
        "vllm.v1.metrics.prefill_latency.monotonic_ns", side_effect=[200, 300, 500]
    ):
        recorder.enqueue("request")
        recorder.record_input_queue("request", 50, 100)
        recorder.check("request")
        recorder.finish("request", 400)

    spans = recorder.snapshot()["requests"]["request"]
    assert spans["waiting"][0]["start_ns"] == 200
    assert spans["prefill_total"] == [
        {"start_ns": 50, "end_ns": 500, "duration_ns": 450}
    ]


def test_reset_discards_requests_that_were_already_in_flight():
    recorder = SchedulerPrefillLatency()
    recorder.enqueue("old")
    recorder.check("old")
    recorder.schedule("old")
    initial = recorder.snapshot()

    recorder.reset()
    recorder.finish("old", 1)
    recorder.enqueue("new")
    recorder.check("new")

    assert set(initial["requests"]) == {"old"}
    assert set(recorder.snapshot()["requests"]) == {"new"}


def test_cache_lookup_aggregates_attempts_by_source():
    recorder = SchedulerPrefillLatency()
    recorder.enqueue("request")
    recorder.record_cache_lookup("request", 10, 13, source="local")
    recorder.record_cache_lookup("request", 15, 20, source="external")
    initial = recorder.snapshot()["requests"]["request"]
    recorder.record_cache_lookup("request", 120, 122, source="local")
    recorder.record_cache_lookup(
        "request", 124, 126, source="external", matched_tokens=0
    )
    recorder.record_cache_lookup(
        "request", 130, 134, source="external", matched_tokens=8
    )
    recorder.record_cache_lookup("request", 140, 142, source="external")
    recorder.record_cache_lookup(
        "request", 160, 165, source="external", matched_tokens=16
    )
    recorder.record_cache_lookup("request", 180, 181, source="external")

    spans = recorder.snapshot()["requests"]["request"]
    assert len(spans["cache_lookup"]) == 2
    assert [entry["calls"] for entry in initial["cache_lookup"]] == [1, 1]
    assert spans["cache_lookup"] == [
        {
            "source": "local",
            "start_ns": 10,
            "end_ns": 122,
            "duration_ns": 112,
            "calls": 2,
            "total_call_ns": 5,
        },
        {
            "source": "external",
            "start_ns": 15,
            "end_ns": 181,
            "duration_ns": 166,
            "calls": 6,
            "total_call_ns": 19,
            "matched_tokens": None,
            "pending_count": 3,
            "first_result_call": 2,
            "later_result_count": 2,
            "later_result_total_ns": 29,
        },
    ]
    assert spans["external_lookup_elapsed"] == [
        {"start_ns": 15, "end_ns": 126, "duration_ns": 111}
    ]


def test_npu_profile_counts_overlapping_tasks_once(tmp_path: Path):
    (tmp_path / "kernel_details.csv").write_text(
        "Accelerator Core,Start Time(us),Duration(us)\n"
        "AI_CORE,10,3\nAI_CORE,12,3\nAI_CPU,20,8\n",
        encoding="utf-8",
    )
    h2d = {"operation": "HostToDevice", "Start Time": 13, "End Time": 15}
    d2h = {"operation": "DeviceToHost", "Start Time": 15, "End Time": 20}
    (tmp_path / "trace_view.json").write_text(
        json.dumps([{"args": h2d}, {"args": d2h}]), encoding="utf-8"
    )

    assert read_npu_profile(tmp_path) == {"ai_core": 5000, "h2d": 2000}


def test_prefill_latency_api_reads_and_resets_request_spans():
    recorder = SchedulerPrefillLatency()
    recorder.enqueue("request")
    recorder.check("request")

    async def prefill_latency(reset: bool = False):
        if reset:
            recorder.reset()
            return [None]
        return [recorder.snapshot()]

    app = FastAPI()
    app.state.engine_client = SimpleNamespace(prefill_latency=prefill_latency)
    app.include_router(router)
    with TestClient(app) as client:
        url = "/v1/prefill_latency"
        payload = client.get(url).json()
        span = payload["engines"][0]["requests"]["request"]["waiting"][0]
        assert span["duration_ns"] == span["end_ns"] - span["start_ns"]
        assert client.post(f"{url}/reset").json() == {"reset": True}
        assert client.get(url).json()["engines"][0]["requests"] == {}
