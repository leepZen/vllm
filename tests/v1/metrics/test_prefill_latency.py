# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.prefill_latency import router
from vllm.v1.metrics.prefill_device import read_npu_profile
from vllm.v1.metrics.prefill_latency import SchedulerPrefillLatency


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
