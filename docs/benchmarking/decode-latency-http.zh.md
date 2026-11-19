# Decode 时延累计统计

Python 版 `vllm serve` 在已有端口提供 `GET /v1/decode-latency` 和 `POST /v1/decode-latency/reset`。设置 `VLLM_DECODE_LATENCY_DIR` 后，同一台 Linux 主机上的 EngineCore、worker 和 API 进程使用共享目录。写入进程使用固定大小的内存映射直方图，请求数增长时文件大小保持固定。请求结束时，各段计时加入直方图，请求级临时计时状态随即释放。组件计时在各自完成时直接累计。

## 启动 D 端

原始 v0.30.0 使用 `vllm-v0.30.0-decode-latency-aggregate-full.patch`。已装过上一版累计接口的容器使用 `vllm-v0.30.0-decode-latency-aggregate-upgrade.patch`。将补丁放入容器的 `/tmp`，在服务实际加载的源码根目录执行以下命令。源码安装使用 checkout 根目录，wheel 安装使用包含 `vllm/envs.py` 的 site-packages 目录。

```sh
PATCH=/tmp/vllm-v0.30.0-decode-latency-aggregate-full.patch
test -f vllm/envs.py && git apply --check "$PATCH" && git apply "$PATCH"
```

启动或重启 D 端时设置 `VLLM_DECODE_LATENCY_DIR=/dev/shm/vllm-latency/run-01`。Docker 示例使用 `-p 127.0.0.1:18000:8000` 发布已有推理端口，容器内的 `vllm serve` 监听 `--host 0.0.0.0 --port 8000`。已有主机端口映射时，可直接使用该端口。每次新建服务运行可以指定一个新目录，同一次服务运行里的多轮实验可用清零接口。采集器仅使用 `decode-hist-v1-*.bin` 文件，升级前的 NDJSON 文件可以单独归档。

## 压测与查询

```sh
BASE_URL=http://127.0.0.1:18000

curl -fsS -X POST "$BASE_URL/v1/decode-latency/reset"
# 发送本轮压测请求，待请求结束后查询。
curl -fsS "$BASE_URL/v1/decode-latency" -o summary.json
jq '{request_count, ttft: .latencies.ttft_ms, npu: .components.npu_forward_ms}' summary.json
```

配置了 `--api-key` 时，两条 curl 都添加 `-H "Authorization: Bearer $VLLM_API_KEY"`。GET 返回一个 JSON 对象。旧版的两个明细视图 `view=requests` 和 `view=events` 已移除，`view=summary` 仍可使用。

顶层 `request_count` 为结束的引擎请求数，包含取消和失败。各指标的 `count` 只统计取得有效时延的样本。平均值由累计总耗时除以数量。P90、P99 取最邻近秩落入的直方图桶上界。非零桶从 1 微秒开始，相邻桶上界相差 2%。零值单独计数。超出最高有限桶的百分位显示为 `null`，平均值和数量仍计入该样本。空指标的 `count` 为 0，其他值为 `null`。

清零成功返回 `{"reset":true}`，同目录的请求和组件统计都会归零。测试不同 CPU 频率时，先等预热请求结束和设备后台采样完成，再清零。清零时仍在途的请求会在结束后纳入新一轮，后台设备计时在采样完成后纳入统计。

## 请求时延 `latencies`

| 字段 | 起止边界 |
| --- | --- |
| `ingress_ms` | D 端 HTTP 入口 → scheduler 入队 |
| `queue_ms` | 入队 → 首次检查该请求 |
| `admission_ms` | 首次检查 → 首次获配大于零的计算 token |
| `first_compute_to_token_ms` | 首次获配计算 token → EngineCore 首次出词 |
| `decode_ms` | EngineCore 首次出词 → 最后一次出词 |
| `ttft_ms`、`total_ms` | D 端 HTTP 入口分别到首次、最后一次出词 |
| `tpot_ms` | `decode_ms / (output_tokens - 1)`。少于两个输出 token 时没有样本。 |
| `lookup_call_ms`、`lookup_to_result_ms` | 单请求内外部查询的累计调用时间，以及首次调用到取得确定结果的跨度 |
| `async_kv_wait_ms` | 单请求等待异步 KV 完成通知的累计时间 |

API 与 EngineCore 需要共享单调时钟域才能计算入口相关时延。等待 KV 的 step 如未分配计算 token，仍计入 `admission_ms`。

## 组件时延 `components`

| 字段 | 计时与计数单位 |
| --- | --- |
| `schedule_ms`、`schedule_thread_cpu_ms`、`result_processing_ms` | 每次 EngineCore 调度的墙钟时间和线程 CPU 时间，以及每个 batch 的结果处理时间 |
| `dispatch_to_worker_ms`、`worker_execute_ms` | 每个 worker 的派发到 runner 入口耗时及 runner 执行耗时。派发计时需要共享单调时钟域。 |
| `prepare_inputs_host_ms`、`forward_host_ms`、`sample_tokens_host_ms` | 每次对应 host 函数调用 |
| `kv_load_host_ms`、`kv_retrieve_host_ms`、`kv_wait_for_layer_host_ms` | 每次 start load、retrieve 或逐层等待的 host 调用，取决于运行时 connector 提供的接口。异步传输完成等待另见请求级 `async_kv_wait_ms`。 |
| `npu_forward_ms`、`npu_sample_tokens_ms`、`npu_kv_wait_for_layer_ms`、`npu_stream_gap_ms` | 每个 worker 上对应的设备计时。批间空档按同一设备流测量，设备事件在后台完成后入统计。 |

组件统计合并同目录各进程的直方图。共享 forward 每个执行 worker 各计一次。NPU 样本量取决于设备计时支持和后台采样进度。
