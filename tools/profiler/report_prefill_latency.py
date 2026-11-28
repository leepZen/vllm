# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Print a Markdown report for a /v1/prefill_latency JSON snapshot."""

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Any

REQUEST_COLUMNS = (
    "input_queue",
    "input_queue → scheduler",
    "waiting",
    "scheduling",
    "scheduled → engine result",
    "first_token",
    "prefill_total",
)
WORKER_METRICS = (
    "worker_dispatch",
    "worker_before_forward",
    "forward_host",
    "worker_execute_host",
    "sampling_host",
    "npu_forward",
    "h2d",
    "ai_core",
)
MISSING = "—"


def ms(ns: int | float | None) -> str:
    return MISSING if ns is None else f"{ns / 1_000_000:,.3f}"


def table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    def print_row(row: Iterable[object]) -> None:
        cells = (str(cell).replace("|", r"\|").replace("\n", " ") for cell in row)
        print("| " + " | ".join(cells) + " |")

    headers = tuple(headers)
    print_row(headers)
    print_row(["---"] * len(headers))
    for row in rows:
        print_row(row)
    print()


def first_span(spans: dict[str, Any], name: str) -> dict[str, Any]:
    return next(iter(spans.get(name, [])), {})


def elapsed(start: int | None, end: int | None) -> int | None:
    if start is not None and end is not None and start <= end:
        return end - start
    return None


def request_latencies(spans: dict[str, Any]) -> tuple[int | None, ...]:
    input_queue = first_span(spans, "input_queue")
    waiting = first_span(spans, "waiting")
    scheduling = first_span(spans, "scheduling")
    first_token = first_span(spans, "first_token")
    prefill = first_span(spans, "prefill_total")
    return (
        input_queue.get("duration_ns"),
        elapsed(input_queue.get("end_ns"), waiting.get("start_ns")),
        waiting.get("duration_ns"),
        scheduling.get("duration_ns"),
        elapsed(scheduling.get("end_ns"), first_token.get("start_ns")),
        first_token.get("duration_ns"),
        prefill.get("duration_ns"),
    )


def lookup_summary(spans: dict[str, Any]) -> tuple[object, ...]:
    local = []
    external = []
    for lookup in spans.get("cache_lookup", []):
        if lookup.get("source") == "local":
            local.append(lookup)
        elif lookup.get("source") == "external":
            external.append(lookup)
    result_spans = spans.get("external_lookup_elapsed", [])
    first_result_span = next(iter(result_spans), {})
    result_ns = first_result_span.get("end_ns")
    first_result_call = None
    external_calls = 0
    for lookup in external:
        if first_result_call is None:
            result_call = lookup.get("first_result_call")
            if "calls" not in lookup and lookup.get("matched_tokens") is not None:
                result_call = 1
                result_ns = lookup["end_ns"]
            if result_call is not None:
                first_result_call = external_calls + result_call
                if result_ns is None and result_call == lookup.get("calls", 1):
                    result_ns = lookup["end_ns"]
        external_calls += lookup.get("calls", 1)
    external_summary = external[-1] if external else {}
    last_match = external_summary.get("matched_tokens", MISSING)
    local_call_ns = sum(
        lookup.get("total_call_ns", lookup["duration_ns"]) for lookup in local
    )
    external_call_ns = sum(
        lookup.get("total_call_ns", lookup["duration_ns"]) for lookup in external
    )
    rechecks = (
        external_calls - first_result_call if first_result_call is not None else MISSING
    )
    first_call_ns = external[0]["start_ns"] if external else None
    last_call_end_ns = external[-1]["end_ns"] if external else None
    result_wait_ns = first_result_span.get(
        "duration_ns", elapsed(first_call_ns, result_ns)
    )
    if "later_result_count" in external_summary:
        later_result_count = external_summary["later_result_count"]
        later_result_total_ns = external_summary["later_result_total_ns"]
    elif result_spans:
        later_results = result_spans[1:]
        later_result_count = len(later_results)
        later_result_total_ns = sum(span["duration_ns"] for span in later_results)
    else:
        later_result_count = MISSING
        later_result_total_ns = None
    return (
        sum(lookup.get("calls", 1) for lookup in local),
        ms(local_call_ns if local else None),
        external_calls,
        ms(external_call_ns if external else None),
        ms(elapsed(first_call_ns, last_call_end_ns)),
        sum(
            lookup.get("pending_count", lookup.get("matched_tokens") is None)
            for lookup in external
        ),
        rechecks,
        "null" if last_match is None else last_match,
        ms(result_wait_ns),
        ms(elapsed(result_ns, first_span(spans, "scheduling").get("end_ns"))),
        later_result_count,
        ms(later_result_total_ns),
    )


def print_cache_loads(requests: dict[str, Any]) -> None:
    print("### KV loads (ms)\n")
    loads = defaultdict(list)
    for request_id, spans in requests.items():
        for span in spans.get("load_cache", []):
            key = (
                request_id,
                span.get("boundary", MISSING),
                span.get("worker_rank", MISSING),
                span.get("host", MISSING),
            )
            loads[key].append(span)
    if not loads:
        print("No `load_cache` records in this snapshot.\n")
        return
    table(
        ("Request", "Boundary", "Worker rank", "Host", "Count", "Failed", "Sum"),
        (
            (
                *key,
                len(spans),
                sum(span.get("failed") is True for span in spans),
                ms(sum(span["duration_ns"] for span in spans)),
            )
            for key, spans in loads.items()
        ),
    )


def print_worker_totals(workers: list[dict[str, Any]]) -> None:
    print("### Worker cumulative times (ms)\n")
    totals = [worker.get("shared_totals_ns", {}) for worker in workers]
    extra_metrics = sorted(
        {key for total in totals for key in total} - set(WORKER_METRICS)
    )
    metrics = (*WORKER_METRICS, *extra_metrics)
    table(
        ("Metric", "Ranks sampled", "Rank mean", "Rank min", "Rank max"),
        (
            (name, len(values), ms(mean(values)), ms(min(values)), ms(max(values)))
            if (values := [total[name] for total in totals if name in total])
            else (name, 0, MISSING, MISSING, MISSING)
            for name in metrics
        ),
    )
    if workers:
        table(
            ("Rank", "Host", *metrics),
            (
                (
                    worker.get("rank", MISSING),
                    worker.get("host", MISSING),
                    *(ms(total.get(name)) for name in metrics),
                )
                for worker, total in zip(workers, totals)
            ),
        )


def print_details(requests: dict[str, Any]) -> None:
    print("### Raw request spans\n")
    for request_id, segments in requests.items():
        print(f"#### {request_id}\n")
        table(
            ("Stage", "Start ns", "End ns", "Duration ns", "Duration ms", "Metadata"),
            (
                (
                    name,
                    span["start_ns"],
                    span["end_ns"],
                    span["duration_ns"],
                    ms(span["duration_ns"]),
                    json.dumps(
                        {
                            key: value
                            for key, value in span.items()
                            if key not in {"start_ns", "end_ns", "duration_ns"}
                        }
                    ),
                )
                for name, spans in segments.items()
                for span in spans
            ),
        )


def print_report(snapshot: dict[str, Any], details: bool, partial: bool) -> None:
    print("# Prefill latency report\n")
    if partial:
        print("> Partial snapshot. Worker statistics cover complete ranks only.\n")
    print(f"- Request clock: `{snapshot.get('request_clock', MISSING)}`")
    print(
        f"- NPU task profiling: `{snapshot.get('npu_task_profiling_enabled', MISSING)}`"
    )
    print(f"- {MISSING} denotes missing data. Worker rows are rank totals.\n")
    for engine in snapshot["engines"]:
        requests = engine.get("requests", {})
        workers = engine.get("workers", [])
        rank = engine.get("engine_rank", MISSING)
        host = engine.get("host", MISSING)
        print(f"## Engine {rank} · {host}\n")
        print(f"Requests: {len(requests)}. Worker ranks: {len(workers)}.\n")
        print("### Request latencies (ms, first recorded span per stage)\n")
        latencies = [request_latencies(spans) for spans in requests.values()]
        rows = [
            (request_id, *map(ms, values))
            for request_id, values in zip(requests, latencies)
        ]
        if latencies:
            samples = (
                [ns for ns in column if ns is not None] for column in zip(*latencies)
            )
            rows.append(("Mean", *(ms(mean(ns)) if ns else MISSING for ns in samples)))
        table(("Request", *REQUEST_COLUMNS), rows)
        print("### Cache lookup (ms)\n")
        print("Call sums cover lookup calls. The span also includes gaps.\n")
        table(
            (
                "Request",
                "Local calls",
                "Local call sum",
                "External calls",
                "External call sum",
                "External first → last span",
                "Pending",
                "Calls after first result",
                "Last matched tokens",
                "First call → first result",
                "First result → scheduled",
                "Later result spans",
                "Later result span sum",
            ),
            (
                (request_id, *lookup_summary(spans))
                for request_id, spans in requests.items()
            ),
        )
        print_cache_loads(requests)
        print_worker_totals(workers)
        engine_totals = engine.get("shared_totals_ns", {})
        if engine_totals:
            print("### Engine cumulative times (ms)\n")
            table(
                ("Metric", "Duration"), ((k, ms(v)) for k, v in engine_totals.items())
            )
        if details:
            print_details(requests)


def recover_worker_tail(raw: str) -> dict[str, Any]:
    """Read complete records from a snapshot clipped in the final workers array."""
    worker_arrays = list(re.finditer(r'"workers"\s*:\s*\[', raw))
    if not worker_arrays:
        raise ValueError("no workers array found in the partial snapshot")

    start = worker_arrays[-1].end()
    snapshot = json.loads(raw[:start] + "]}]}")
    workers = snapshot["engines"][-1]["workers"]
    decoder = json.JSONDecoder()
    remaining = raw[start:].lstrip()
    while remaining.startswith("{"):
        try:
            worker, end = decoder.raw_decode(remaining)
        except json.JSONDecodeError:
            break
        workers.append(worker)
        remaining = remaining[end:].lstrip()
        if not remaining.startswith(","):
            break
        remaining = remaining[1:].lstrip()
    if remaining.startswith("]") and remaining.rstrip("]} \t\r\n"):
        raise ValueError("the final workers array is followed by other data")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="JSON snapshot file, or - for stdin")
    parser.add_argument(
        "--details", action="store_true", help="show every request span"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="read complete records when the final workers array is cut off",
    )
    args = parser.parse_args()
    try:
        if args.trace == Path("-"):
            raw = sys.stdin.read()
        else:
            raw = args.trace.read_text(encoding="utf-8-sig")
    except (OSError, ValueError) as exc:
        parser.error(f"could not read trace: {exc}")

    partial = False
    try:
        snapshot = json.loads(raw)
    except json.JSONDecodeError as exc:
        if not args.allow_partial:
            parser.error(f"could not read trace: {exc}")
        try:
            snapshot = recover_worker_tail(raw)
        except (ValueError, KeyError, IndexError, TypeError) as recovery_error:
            parser.error(f"could not recover partial trace: {recovery_error}")
        partial = True

    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("engines"), list):
        parser.error("expected a /v1/prefill_latency snapshot with an engines array")
    try:
        print_report(snapshot, args.details, partial)
    except (AttributeError, KeyError, TypeError) as exc:
        parser.error(f"invalid trace data: {exc}")


if __name__ == "__main__":
    main()
