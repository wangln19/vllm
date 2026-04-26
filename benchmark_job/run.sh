#!/bin/bash
set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

MODEL_PATH="${MODEL_PATH:-/mnt/moonfs/public-models-m4/moonshotai/Kimi-K2.5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k25}"
MODEL_ALIAS_ROOT="${MODEL_ALIAS_ROOT:-/tmp/vllm_model_alias}"
PD_ROUTE="${PD_ROUTE:-nixl}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
PROXY_PORT="${PROXY_PORT:-8192}"

PREFILL_KV_PORT="${PREFILL_KV_PORT:-14579}"
DECODE_KV_PORT="${DECODE_KV_PORT:-14580}"
KV_BUFFER_SIZE="${KV_BUFFER_SIZE:-5000000000}"
KV_MEM_POOL_SIZE_GB="${KV_MEM_POOL_SIZE_GB:-4}"
ENABLE_P2P_PROXY="${ENABLE_P2P_PROXY:-1}"
P2P_PROXY_PORT="${P2P_PROXY_PORT:-30001}"

PREFILL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_CHANNEL_PORT:-5600}"
DECODE_SIDE_CHANNEL_PORT="${DECODE_SIDE_CHANNEL_PORT:-5600}"
NIXL_KV_ROLE="${NIXL_KV_ROLE:-kv_both}"
NIXL_KV_LOAD_FAILURE_POLICY="${NIXL_KV_LOAD_FAILURE_POLICY:-fail}"
UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
UCX_TLS="${UCX_TLS:-all}"

BENCH_DATASET_NAME="${BENCH_DATASET_NAME:-sharegpt}"
BENCH_DATASET_PATH="${BENCH_DATASET_PATH:-/mnt/moonfs/integration-m4/texts/ShareGPT_V3_unfiltered_cleaned_split.json}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-16}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-16}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-1}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-2}"
BENCH_ENDPOINT="${BENCH_ENDPOINT:-/v1/chat/completions}"
BENCH_BACKEND="${BENCH_BACKEND:-vllm}"
BENCH_NUM_WARMUPS="${BENCH_NUM_WARMUPS:-0}"
BENCH_SAVE_DETAILED="${BENCH_SAVE_DETAILED:-1}"

LOAD_FORMAT="${LOAD_FORMAT:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
DEBUG_SLEEP_ON_FAILURE="${DEBUG_SLEEP_ON_FAILURE:-0}"

LOG_DIR="/tmp/vllm_logs"
RESULT_DIR="/tmp/pd_bench_results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

HOST_LIST_RAW="${MSH_ALL_HOSTS:-}"
HOST_LIST_NORMALIZED="${HOST_LIST_RAW//,/ }"
ALL_HOSTS=($HOST_LIST_NORMALIZED)
if [ "${#ALL_HOSTS[@]}" -lt 2 ]; then
    echo "ERROR: PD benchmark requires at least 2 hosts, got: ${MSH_ALL_HOSTS:-<empty>}"
    exit 1
fi
PREFILL_HOST="${ALL_HOSTS[0]}"
DECODE_HOST="${ALL_HOSTS[1]}"

if [ "$PD_ROUTE" = "nixl" ]; then
    export VLLM_USE_V1=1
fi

prepare_model_path() {
    MODEL_PATH_EFFECTIVE="${MODEL_PATH}"
    local model_basename safe_basename safe_model_path

    model_basename="$(basename "${MODEL_PATH}")"
    if [[ "${model_basename}" == *.* ]]; then
        safe_basename="${model_basename//./_}"
        safe_model_path="${MODEL_ALIAS_ROOT}/${safe_basename}"
        mkdir -p "${MODEL_ALIAS_ROOT}"
        rm -f "${safe_model_path}"
        ln -s "${MODEL_PATH}" "${safe_model_path}"
        MODEL_PATH_EFFECTIVE="${safe_model_path}"
    fi
}

print_debug_context() {
    echo "NODE_RANK=$NODE_RANK"
    echo "PD_ROUTE=$PD_ROUTE"
    echo "MODEL_PATH=$MODEL_PATH"
    echo "MODEL_PATH_EFFECTIVE=$MODEL_PATH_EFFECTIVE"
    echo "MSH_ALL_HOSTS=$MSH_ALL_HOSTS"
    echo "HOST_LIST_NORMALIZED=$HOST_LIST_NORMALIZED"
    echo "ALL_HOSTS=${ALL_HOSTS[*]}"
    echo "PREFILL_HOST=$PREFILL_HOST"
    echo "DECODE_HOST=$DECODE_HOST"
    echo "PWD=$(pwd)"
    echo "DATE=$(date)"
    ls -la || true
}

upload_artifacts() {
    echo "Uploading logs and results..."
    if ! command -v lpc >/dev/null 2>&1; then
        gyro_install_action="auto-install" sh -c "$(curl -fsSL https://gyro.app.msh.team/install/install.sh)" || true
        export PATH=/root/.gyro/bin:$PATH
        gyro install lpc || true
    fi
    if command -v lpc >/dev/null 2>&1; then
        find "$LOG_DIR" "$RESULT_DIR" -maxdepth 1 -type f -print0 2>/dev/null | while IFS= read -r -d '' f; do
            lpc artifact upload --filepath "$f" 2>/dev/null || true
        done
    fi
}

sync_results_to_persistent() {
    local label="${1:-sync}"
    mkdir -p /mnt/moonfs/kimiv-m4/wanglinian/pd_bench_results
    if [ -d "$LOG_DIR" ]; then
        cp -r "$LOG_DIR"/* /mnt/moonfs/kimiv-m4/wanglinian/pd_bench_results/ 2>/dev/null || true
    fi
    if [ -d "$RESULT_DIR" ]; then
        cp -r "$RESULT_DIR"/* /mnt/moonfs/kimiv-m4/wanglinian/pd_bench_results/ 2>/dev/null || true
    fi
    echo "[$label] Synced results to persistent storage"
}

cleanup_local_processes() {
    local maybe_pid
    for maybe_pid in "${PREFILL_PID:-}" "${DECODE_PID:-}" "${PROXY_PID:-}"; do
        if [ -n "${maybe_pid}" ] && kill -0 "$maybe_pid" 2>/dev/null; then
            kill "$maybe_pid" 2>/dev/null || true
            wait "$maybe_pid" 2>/dev/null || true
        fi
    done
}

debug_pause_on_failure() {
    local log_file="$1"
    cat "$log_file" 2>/dev/null || true
    if [ "${DEBUG_SLEEP_ON_FAILURE}" -gt 0 ]; then
        echo "Debug sleep on failure: ${DEBUG_SLEEP_ON_FAILURE}s"
        sleep "${DEBUG_SLEEP_ON_FAILURE}"
    fi
}

PERSISTENT_RESULT_DIR="/mnt/moonfs/kimiv-m4/wanglinian/pd_bench_results"

on_exit() {
    upload_artifacts
    # Also copy results to persistent shared storage
    if [ -d "$RESULT_DIR" ]; then
        mkdir -p "$PERSISTENT_RESULT_DIR"
        cp -r "$RESULT_DIR"/* "$PERSISTENT_RESULT_DIR"/ 2>/dev/null || true
        echo "Results copied to $PERSISTENT_RESULT_DIR"
    fi
    if [ -d "$LOG_DIR" ]; then
        mkdir -p "$PERSISTENT_RESULT_DIR"
        cp -r "$LOG_DIR"/* "$PERSISTENT_RESULT_DIR"/ 2>/dev/null || true
        echo "Logs copied to $PERSISTENT_RESULT_DIR"
    fi
    cleanup_local_processes
    if [ "${DEBUG_SLEEP_ON_FAILURE}" -gt 0 ]; then
        echo "Exit trap debug sleep: ${DEBUG_SLEEP_ON_FAILURE}s"
        sleep "${DEBUG_SLEEP_ON_FAILURE}"
    fi
}

trap on_exit EXIT

wait_for_http() {
    local name="$1"
    local url="$2"
    local timeout="${3:-1800}"
    local log_file="$4"

    echo "Waiting for ${name}: ${url} (timeout=${timeout}s)"
    for i in $(seq 1 "$timeout"); do
        if curl -sf "$url" >/dev/null 2>&1; then
            echo "${name} ready after ${i}s"
            return 0
        fi
        if [ $((i % 60)) -eq 0 ]; then
            echo "${name} still not ready after ${i}s"
            tail -n 50 "$log_file" 2>/dev/null || true
            sync_results_to_persistent "wait_${name}"
        fi
        sleep 1
    done
    echo "ERROR: ${name} timed out"
    tail -n 200 "$log_file" 2>/dev/null || true
    sync_results_to_persistent "wait_${name}_timeout"
    if [ "${DEBUG_SLEEP_ON_FAILURE}" -gt 0 ]; then
        echo "Debug sleep on failure: ${DEBUG_SLEEP_ON_FAILURE}s"
        sleep "${DEBUG_SLEEP_ON_FAILURE}"
    fi
    return 1
}

require_python_module() {
    local module_name="$1"
    python - "$module_name" <<'PY'
import importlib
import sys

module_name = sys.argv[1]
try:
    importlib.import_module(module_name)
except Exception as exc:
    raise SystemExit(
        f"Python module check failed for {module_name}: {exc}"
    ) from exc
print(f"Python module ready: {module_name}")
PY
}

build_server_extra_args() {
    SERVER_EXTRA_ARGS=()

    if [ -n "${EXTRA_VLLM_ARGS}" ]; then
        # shellcheck disable=SC2206
        EXTRA_ARGS_ARRAY=(${EXTRA_VLLM_ARGS})
        SERVER_EXTRA_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
    fi

    if [ -n "${LOAD_FORMAT}" ]; then
        SERVER_EXTRA_ARGS+=(--load-format "${LOAD_FORMAT}")
    fi

    if [ "${ENFORCE_EAGER}" = "1" ]; then
        SERVER_EXTRA_ARGS+=(--enforce-eager)
    fi
}

find_benchmark_script() {
    local candidate
    echo "[DEBUG] find_benchmark_script: pwd=$(pwd), looking for benchmark_disagg_prefill_decode.py" >&2
    for candidate in \
        "./benchmark_disagg_prefill_decode.py" \
        "$(pwd)/benchmark_disagg_prefill_decode.py" \
        "/app/benchmark_disagg_prefill_decode.py" \
        "/app/benchmark_job/benchmark_disagg_prefill_decode.py" \
        "/app/Megatron-DeepSpeed/benchmark_disagg_prefill_decode.py" \
        "/app/Megatron-DeepSpeed/benchmark_job/benchmark_disagg_prefill_decode.py"
    do
        echo "[DEBUG] Checking: $candidate" >&2
        if [ -f "$candidate" ]; then
            echo "[DEBUG] Found: $candidate" >&2
            echo "$candidate"
            return 0
        fi
    done

    local found
    found=$(find /app -maxdepth 5 -name benchmark_disagg_prefill_decode.py 2>/dev/null | head -n 1)
    if [ -n "$found" ]; then
        echo "[DEBUG] Found via find: $found" >&2
        echo "$found"
        return 0
    fi
    echo "[DEBUG] Not found anywhere" >&2
    return 1
}

run_benchmark_script() {
    local mode="$1"
    local result_file="$2"
    local benchmark_script="$3"

    python "$benchmark_script" \
        --mode "$mode" \
        --prefill-url "http://${PREFILL_HOST}:${PREFILL_PORT}" \
        --decode-url "http://${DECODE_HOST}:${DECODE_PORT}" \
        --model "$SERVED_MODEL_NAME" \
        --dataset-name "$BENCH_DATASET_NAME" \
        --dataset-path "$BENCH_DATASET_PATH" \
        --num-prompts "$BENCH_NUM_PROMPTS" \
        --sharegpt-output-len "$BENCH_OUTPUT_LEN" \
        --request-rate "$BENCH_REQUEST_RATE" \
        --max-concurrency "$BENCH_MAX_CONCURRENCY" \
        --prefill-kv-port "$PREFILL_KV_PORT" \
        --decode-kv-port "$DECODE_KV_PORT" \
        --disable-tqdm \
        --save-result \
        --result-dir "$RESULT_DIR" \
        --result-filename "$result_file"
}

build_p2p_kv_config() {
    local role="$1"
    local kv_rank="$2"
    local kv_port="$3"
    local http_port="$4"
    local bind_host="$5"

    if [ "${ENABLE_P2P_PROXY}" = "1" ]; then
        printf '%s' \
            '{"kv_connector":"P2pNcclConnector","kv_role":"'"${role}"'","kv_rank":'"${kv_rank}"',"kv_parallel_size":2,"kv_buffer_size":'"${KV_BUFFER_SIZE}"',"kv_port":'"${kv_port}"',"kv_connector_extra_config":{"proxy_ip":"'"${PREFILL_HOST}"'","proxy_port":"'"${P2P_PROXY_PORT}"'","http_port":"'"${http_port}"'","send_type":"PUT_ASYNC","mem_pool_size_gb":'"${KV_MEM_POOL_SIZE_GB}"'}}'
    else
        printf '%s' \
            '{"kv_connector":"P2pNcclConnector","kv_role":"'"${role}"'","kv_rank":'"${kv_rank}"',"kv_parallel_size":2,"kv_buffer_size":'"${KV_BUFFER_SIZE}"',"kv_ip":"'"${bind_host}"'","kv_port":'"${kv_port}"',"kv_connector_extra_config":{"mem_pool_size_gb":'"${KV_MEM_POOL_SIZE_GB}"'}}'
    fi
}

build_nixl_kv_config() {
    printf '%s' \
        '{"kv_connector":"NixlConnector","kv_role":"'"${NIXL_KV_ROLE}"'","kv_load_failure_policy":"'"${NIXL_KV_LOAD_FAILURE_POLICY}"'"}'
}

start_vllm_server() {
    local node_kind="$1"
    local bind_host="$2"
    local api_port="$3"
    local kv_config="$4"
    local log_file="$5"
    local side_channel_port="${6:-}"
    local extra_env=()

    if [ "$PD_ROUTE" = "nixl" ]; then
        extra_env+=(
            "UCX_TLS=${UCX_TLS}"
            "UCX_NET_DEVICES=${UCX_NET_DEVICES}"
            "VLLM_NIXL_SIDE_CHANNEL_HOST=${bind_host}"
            "VLLM_NIXL_SIDE_CHANNEL_PORT=${side_channel_port}"
        )
    else
        extra_env+=("VLLM_HOST_IP=${bind_host}")
    fi

    nohup env "${extra_env[@]}" python -u -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH_EFFECTIVE}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --trust-remote-code \
        "${SERVER_EXTRA_ARGS[@]}" \
        --port "${api_port}" \
        --kv-transfer-config "${kv_config}" \
        > "${log_file}" 2>&1 &

    local server_pid=$!
    echo "${node_kind} PID: ${server_pid}"
    sleep 10
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "ERROR: ${node_kind} process exited early"
        sync_results_to_persistent "${node_kind}_early_exit"
        debug_pause_on_failure "${log_file}"
        exit 1
    fi

    if [ "$node_kind" = "prefill" ]; then
        PREFILL_PID="${server_pid}"
    else
        DECODE_PID="${server_pid}"
    fi
}

run_proxy_benchmark() {
    local result_file="$1"
    local bench_log="$2"
    local cmd=(
        vllm bench serve
        --backend "${BENCH_BACKEND}"
        --model "${SERVED_MODEL_NAME}"
        --served-model-name "${SERVED_MODEL_NAME}"
        --tokenizer "${MODEL_PATH_EFFECTIVE}"
        --endpoint "${BENCH_ENDPOINT}"
        --host 127.0.0.1
        --port "${PROXY_PORT}"
        --dataset-name "${BENCH_DATASET_NAME}"
        --num-prompts "${BENCH_NUM_PROMPTS}"
        --request-rate "${BENCH_REQUEST_RATE}"
        --max-concurrency "${BENCH_MAX_CONCURRENCY}"
        --output-len "${BENCH_OUTPUT_LEN}"
        --num-warmups "${BENCH_NUM_WARMUPS}"
        --disable-tqdm
        --trust-remote-code
        --save-result
        --result-dir "${RESULT_DIR}"
        --result-filename "${result_file}"
        --percentile-metrics ttft,tpot,itl
        --metric-percentiles 50,90,99
        --metadata "pd_route=${PD_ROUTE}" "tp=${TP_SIZE}" "model=${SERVED_MODEL_NAME}"
    )

    if [ -n "${BENCH_DATASET_PATH}" ]; then
        cmd+=(--dataset-path "${BENCH_DATASET_PATH}")
    fi

    if [ "${BENCH_SAVE_DETAILED}" = "1" ]; then
        cmd+=(--save-detailed)
    fi

    set -o pipefail
    "${cmd[@]}" 2>&1 | tee "${bench_log}"
    local exit_code=$?
    set +o pipefail
    return $exit_code
}

start_nixl_toy_proxy() {
    nohup env \
        OPENAI_API_KEY="${OPENAI_API_KEY}" \
        PROXY_PORT="${PROXY_PORT}" \
        PREFILL_HOST="${PREFILL_HOST}" \
        PREFILL_PORT="${PREFILL_PORT}" \
        DECODE_HOST="${DECODE_HOST}" \
        DECODE_PORT="${DECODE_PORT}" \
        python -u - <<'PY' > "$LOG_DIR/nixl_proxy.log" 2>&1 &
import json
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

PREFILL_BASE_URL = (
    f"http://{os.environ['PREFILL_HOST']}:{os.environ['PREFILL_PORT']}/v1"
)
DECODE_BASE_URL = (
    f"http://{os.environ['DECODE_HOST']}:{os.environ['DECODE_PORT']}/v1"
)
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_client = httpx.AsyncClient(
        timeout=None,
        base_url=PREFILL_BASE_URL,
        limits=httpx.Limits(
            max_connections=None,
            max_keepalive_connections=None,
        ),
    )
    app.state.decode_client = httpx.AsyncClient(
        timeout=None,
        base_url=DECODE_BASE_URL,
        limits=httpx.Limits(
            max_connections=None,
            max_keepalive_connections=None,
        ),
    )
    yield
    await app.state.prefill_client.aclose()
    await app.state.decode_client.aclose()


app = FastAPI(lifespan=lifespan)


async def send_prefill_request(client: httpx.AsyncClient, endpoint: str,
                               req_data: dict, request_id: str):
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    req_data.pop("min_tokens", None)
    req_data.pop("min_completion_tokens", None)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-Request-Id": request_id,
    }

    response = await client.post(endpoint, json=req_data, headers=headers)
    if response.status_code >= 400:
        print(f"[PROXY] Prefill error {response.status_code}: {response.text}", flush=True)
        print(f"[PROXY] Request body: {json.dumps(req_data, default=str)[:2000]}", flush=True)
    response.raise_for_status()
    return response


async def stream_decode_response(client: httpx.AsyncClient, endpoint: str,
                                 req_data: dict, request_id: str):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-Request-Id": request_id,
    }
    async with client.stream(
            "POST", endpoint, json=req_data, headers=headers) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def handle_request(api: str, request: Request):
    req_data = await request.json()
    request_id = str(uuid.uuid4())

    response = await send_prefill_request(
        request.app.state.prefill_client, api, req_data, request_id)
    response_json = response.json()
    await response.aclose()

    kv_transfer_params = response_json.get("kv_transfer_params", {})
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params

    async def generate_stream():
        async for chunk in stream_decode_response(
                request.app.state.decode_client,
                api,
                req_data,
                request_id=request_id):
            yield chunk

    return StreamingResponse(generate_stream(), media_type="application/json")


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await handle_request("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await handle_request("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}


uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PROXY_PORT"]))
PY
    PROXY_PID=$!
}

wait_for_prefill_controller_exit() {
    local decode_log="$1"
    local prefill_seen=0
    local prefill_down_streak=0
    local prefill_wait_seconds=0

    while kill -0 "${DECODE_PID}" 2>/dev/null; do
        if curl -sf "http://${PREFILL_HOST}:${PREFILL_PORT}/health" >/dev/null 2>&1; then
            prefill_seen=1
            prefill_down_streak=0
            prefill_wait_seconds=0
        else
            if [ "${prefill_seen}" -eq 1 ]; then
                prefill_down_streak=$((prefill_down_streak + 1))
                echo "Prefill no longer healthy, streak=${prefill_down_streak}"
                if [ "${prefill_down_streak}" -ge 6 ]; then
                    echo "Prefill coordinator has exited, shutting down decode node"
                    break
                fi
            else
                prefill_wait_seconds=$((prefill_wait_seconds + 10))
                if [ "${prefill_wait_seconds}" -ge 1800 ]; then
                    echo "ERROR: Prefill never became healthy within 1800s"
                    debug_pause_on_failure "${decode_log}"
                    exit 1
                fi
            fi
        fi
        sleep 10
    done
}

run_p2p_route() {
    local benchmark_script

    if [ "$NODE_RANK" = "0" ]; then
        benchmark_script="$(find_benchmark_script)"
        if [ -z "$benchmark_script" ]; then
            echo "ERROR: benchmark_disagg_prefill_decode.py not found"
            exit 1
        fi
        echo "Using benchmark script: $benchmark_script"

        if [ "${ENABLE_P2P_PROXY}" = "1" ]; then
            echo "=== Starting P2P discovery proxy on ${PREFILL_HOST}:${P2P_PROXY_PORT} ==="
            nohup python -u - <<'PY' > "$LOG_DIR/p2p_proxy.log" 2>&1 &
import msgpack
import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.ROUTER)
sock.bind("tcp://0.0.0.0:30001")
print("P2P discovery proxy listening on 0.0.0.0:30001", flush=True)
while True:
    remote_address, message = sock.recv_multipart()
    data = msgpack.loads(message)
    print({"remote": remote_address.decode(), "data": data}, flush=True)
PY
            PROXY_PID=$!
        fi

        echo "=== Starting PREFILL node on ${PREFILL_HOST}:${PREFILL_PORT} ==="
        start_vllm_server \
            "prefill" \
            "${PREFILL_HOST}" \
            "${PREFILL_PORT}" \
            "$(build_p2p_kv_config kv_producer 0 "${PREFILL_KV_PORT}" "${PREFILL_PORT}" "${PREFILL_HOST}")" \
            "$LOG_DIR/vllm_prefill.log"

        wait_for_http "prefill" "http://localhost:${PREFILL_PORT}/health" 3600 "$LOG_DIR/vllm_prefill.log"
        wait_for_http "decode" "http://${DECODE_HOST}:${DECODE_PORT}/health" 3600 "$LOG_DIR/vllm_prefill.log"

        echo "=== Running prefill benchmark ==="
        run_benchmark_script prefill disagg_prefill_result.json "$benchmark_script" \
            2>&1 | tee "$LOG_DIR/bench_prefill.log"

        echo "=== Running decode benchmark ==="
        run_benchmark_script decode disagg_decode_result.json "$benchmark_script" \
            2>&1 | tee "$LOG_DIR/bench_decode.log"

        echo "=== Benchmark JSON Results ==="
        cat "$RESULT_DIR/disagg_prefill_result.json"
        cat "$RESULT_DIR/disagg_decode_result.json"

        kill "${PREFILL_PID}" 2>/dev/null || true
        wait "${PREFILL_PID}" 2>/dev/null || true
        exit 0
    fi

    if [ "$NODE_RANK" = "1" ]; then
        echo "=== Starting DECODE node on ${DECODE_HOST}:${DECODE_PORT} ==="
        start_vllm_server \
            "decode" \
            "${DECODE_HOST}" \
            "${DECODE_PORT}" \
            "$(build_p2p_kv_config kv_consumer 1 "${DECODE_KV_PORT}" "${DECODE_PORT}" "${DECODE_HOST}")" \
            "$LOG_DIR/vllm_decode.log"

        wait_for_http "decode" "http://localhost:${DECODE_PORT}/health" 3600 "$LOG_DIR/vllm_decode.log"
        wait_for_prefill_controller_exit "$LOG_DIR/vllm_decode.log"

        if kill -0 "${DECODE_PID}" 2>/dev/null; then
            kill "${DECODE_PID}" 2>/dev/null || true
            wait "${DECODE_PID}" 2>/dev/null || true
            exit 0
        fi

        echo "ERROR: Decode process exited unexpectedly"
        debug_pause_on_failure "$LOG_DIR/vllm_decode.log"
        exit 1
    fi
}

run_nixl_route() {
    if [ "$NODE_RANK" = "0" ]; then
        require_python_module nixl
        require_python_module fastapi
        require_python_module httpx
        require_python_module uvicorn

        echo "=== Starting NIXL PREFILL node on ${PREFILL_HOST}:${PREFILL_PORT} ==="
        start_vllm_server \
            "prefill" \
            "${PREFILL_HOST}" \
            "${PREFILL_PORT}" \
            "$(build_nixl_kv_config)" \
            "$LOG_DIR/vllm_prefill.log" \
            "${PREFILL_SIDE_CHANNEL_PORT}"

        wait_for_http "prefill" "http://localhost:${PREFILL_PORT}/health" 3600 "$LOG_DIR/vllm_prefill.log"
        wait_for_http "decode" "http://${DECODE_HOST}:${DECODE_PORT}/health" 3600 "$LOG_DIR/vllm_prefill.log"

        echo "=== Testing direct prefill request ==="
        curl -s -X POST "http://localhost:${PREFILL_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer EMPTY" \
            -d '{"model":"kimi-k25","messages":[{"role":"user","content":"Hello"}],"max_tokens":1}' \
            > "$LOG_DIR/prefill_test_direct.json" 2>&1 || true
        echo "Direct prefill response:"
        cat "$LOG_DIR/prefill_test_direct.json"

        echo "=== Testing prefill request with kv_transfer_params ==="
        curl -s -X POST "http://localhost:${PREFILL_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer EMPTY" \
            -d '{"model":"kimi-k25","messages":[{"role":"user","content":"Hello"}],"max_tokens":1,"kv_transfer_params":{"do_remote_decode":true,"do_remote_prefill":false,"remote_engine_id":null,"remote_block_ids":null,"remote_host":null,"remote_port":null}}' \
            > "$LOG_DIR/prefill_test_kv.json" 2>&1 || true
        echo "KV prefill response:"
        cat "$LOG_DIR/prefill_test_kv.json"

        echo "=== Starting NIXL toy proxy on localhost:${PROXY_PORT} ==="
        start_nixl_toy_proxy

        sleep 5
        if ! kill -0 "${PROXY_PID}" 2>/dev/null; then
            echo "ERROR: NIXL proxy process exited early"
            debug_pause_on_failure "$LOG_DIR/nixl_proxy.log"
            exit 1
        fi

        wait_for_http "nixl-proxy" "http://localhost:${PROXY_PORT}/healthcheck" 300 "$LOG_DIR/nixl_proxy.log"

        # Self-extract benchmark script to /tmp
        benchmark_script="/tmp/benchmark_disagg_prefill_decode.py"
        echo "[RUN.SH] Extracting benchmark script to $benchmark_script" > "$LOG_DIR/run_sh.log"
        python3 -c "import base64; open('$benchmark_script','wb').write(base64.b64decode('IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiIKQmVuY2htYXJrIHByZWZpbGwgYW5kIGRlY29kZSB0aHJvdWdocHV0IHNlcGFyYXRlbHkgZm9yIHZMTE0gUC9EIGRpc2FnZ3JlZ2F0aW9uLgoKU3VwcG9ydHMgU2hhcmVHUFQgZGF0YXNldCBhbmQgY2FuIHRhcmdldCBwcmVmaWxsIG5vZGUsIGRlY29kZSBub2RlLCBvciB0aGUKcHJveHkgZGlyZWN0bHkgZm9yIGVuZC10by1lbmQgbWVhc3VyZW1lbnQuCgpFeGFtcGxlIHVzYWdlOgoKICAjIDEuIFByZWZpbGwgYmVuY2htYXJrIChtYXhfdG9rZW5zPTEsIG1lYXN1cmVzIFRURlQpCiAgcHl0aG9uIGJlbmNobWFya19kaXNhZ2dfcHJlZmlsbF9kZWNvZGUucHkgXAogICAgLS1tb2RlIHByZWZpbGwgXAogICAgLS1wcmVmaWxsLXVybCBodHRwOi8vbG9jYWxob3N0OjgxMDAgXAogICAgLS1tb2RlbCA8bW9kZWxfbmFtZT4gXAogICAgLS1kYXRhc2V0LW5hbWUgc2hhcmVncHQgXAogICAgLS1kYXRhc2V0LXBhdGggL21udC9tb29uZnMvaW50ZWdyYXRpb24tbTQvdGV4dHMvU2hhcmVHUFRfVjNfdW5maWx0ZXJlZF9jbGVhbmVkX3NwbGl0Lmpzb24gXAogICAgLS1udW0tcHJvbXB0cyAxMDAgXAogICAgLS1zaGFyZWdwdC1vdXRwdXQtbGVuIDEyOAoKICAjIDIuIERlY29kZSBiZW5jaG1hcmsgKG5lZWRzIHByZWZpbGwgbm9kZSB0byB3YXJtIHVwIEtWIGNhY2hlIGZpcnN0KQogIHB5dGhvbiBiZW5jaG1hcmtfZGlzYWdnX3ByZWZpbGxfZGVjb2RlLnB5IFwKICAgIC0tbW9kZSBkZWNvZGUgXAogICAgLS1wcmVmaWxsLXVybCBodHRwOi8vbG9jYWxob3N0OjgxMDAgXAogICAgLS1kZWNvZGUtdXJsIGh0dHA6Ly9sb2NhbGhvc3Q6ODIwMCBcCiAgICAtLW1vZGVsIDxtb2RlbF9uYW1lPiBcCiAgICAtLWRhdGFzZXQtbmFtZSBzaGFyZWdwdCBcCiAgICAtLWRhdGFzZXQtcGF0aCAvbW50L21vb25mcy9pbnRlZ3JhdGlvbi1tNC90ZXh0cy9TaGFyZUdQVF9WM191bmZpbHRlcmVkX2NsZWFuZWRfc3BsaXQuanNvbiBcCiAgICAtLW51bS1wcm9tcHRzIDEwMCBcCiAgICAtLXNoYXJlZ3B0LW91dHB1dC1sZW4gMTI4CgogICMgMy4gRW5kLXRvLWVuZCB2aWEgcHJveHkKICBweXRob24gYmVuY2htYXJrX2Rpc2FnZ19wcmVmaWxsX2RlY29kZS5weSBcCiAgICAtLW1vZGUgZTJlIFwKICAgIC0tcHJveHktdXJsIGh0dHA6Ly9sb2NhbGhvc3Q6ODAwMCBcCiAgICAtLW1vZGVsIDxtb2RlbF9uYW1lPiBcCiAgICAtLWRhdGFzZXQtbmFtZSBzaGFyZWdwdCBcCiAgICAtLWRhdGFzZXQtcGF0aCAvbW50L21vb25mcy9pbnRlZ3JhdGlvbi1tNC90ZXh0cy9TaGFyZUdQVF9WM191bmZpbHRlcmVkX2NsZWFuZWRfc3BsaXQuanNvbiBcCiAgICAtLW51bS1wcm9tcHRzIDEwMCBcCiAgICAtLXNoYXJlZ3B0LW91dHB1dC1sZW4gMTI4CiIiIgoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBhc3luY2lvCmltcG9ydCBjb250ZXh0bGliCmltcG9ydCBqc29uCmltcG9ydCBvcwppbXBvcnQgdGltZQppbXBvcnQgdXVpZApmcm9tIGNvbGxlY3Rpb25zLmFiYyBpbXBvcnQgQXN5bmNHZW5lcmF0b3IKZnJvbSBkYXRhY2xhc3NlcyBpbXBvcnQgZGF0YWNsYXNzLCBmaWVsZApmcm9tIHR5cGluZyBpbXBvcnQgQW55CgppbXBvcnQgYWlvaHR0cAppbXBvcnQgbnVtcHkgYXMgbnAKZnJvbSB0cWRtLmFzeW5jaW8gaW1wb3J0IHRxZG0KCmZyb20gdmxsbS5iZW5jaG1hcmtzLmRhdGFzZXRzIGltcG9ydCBTYW1wbGVSZXF1ZXN0LCBhZGRfZGF0YXNldF9wYXJzZXIsIGdldF9zYW1wbGVzCmZyb20gdmxsbS5iZW5jaG1hcmtzLmxpYi5lbmRwb2ludF9yZXF1ZXN0X2Z1bmMgaW1wb3J0ICgKICAgIEFTWU5DX1JFUVVFU1RfRlVOQ1MsCiAgICBSZXF1ZXN0RnVuY0lucHV0LAogICAgUmVxdWVzdEZ1bmNPdXRwdXQsCikKZnJvbSB2bGxtLnRva2VuaXplcnMgaW1wb3J0IGdldF90b2tlbml6ZXIKZnJvbSB2bGxtLnV0aWxzLmFyZ3BhcnNlX3V0aWxzIGltcG9ydCBGbGV4aWJsZUFyZ3VtZW50UGFyc2VyCgpBSU9IVFRQX1RJTUVPVVQgPSBhaW9odHRwLkNsaWVudFRpbWVvdXQodG90YWw9NiAqIDYwICogNjApCgoKQGRhdGFjbGFzcwpjbGFzcyBEaXNhZ2dNZXRyaWNzOgogICAgY29tcGxldGVkOiBpbnQKICAgIGZhaWxlZDogaW50CiAgICB0b3RhbF9pbnB1dF90b2tlbnM6IGludAogICAgdG90YWxfb3V0cHV0X3Rva2VuczogaW50CiAgICBtZWFuX3R0ZnRfbXM6IGZsb2F0CiAgICBtZWRpYW5fdHRmdF9tczogZmxvYXQKICAgIHN0ZF90dGZ0X21zOiBmbG9hdAogICAgcDk5X3R0ZnRfbXM6IGZsb2F0CiAgICBtZWFuX3Rwb3RfbXM6IGZsb2F0CiAgICBtZWRpYW5fdHBvdF9tczogZmxvYXQKICAgIHN0ZF90cG90X21zOiBmbG9hdAogICAgcDk5X3Rwb3RfbXM6IGZsb2F0CiAgICByZXF1ZXN0X3Rocm91Z2hwdXQ6IGZsb2F0CiAgICBpbnB1dF90b2tlbl90aHJvdWdocHV0OiBmbG9hdAogICAgb3V0cHV0X3Rva2VuX3Rocm91Z2hwdXQ6IGZsb2F0CiAgICB0dGZ0czogbGlzdFtmbG9hdF0gPSBmaWVsZChkZWZhdWx0X2ZhY3Rvcnk9bGlzdCkKICAgIHRwb3RzOiBsaXN0W2Zsb2F0XSA9IGZpZWxkKGRlZmF1bHRfZmFjdG9yeT1saXN0KQogICAgbGF0ZW5jaWVzOiBsaXN0W2Zsb2F0XSA9IGZpZWxkKGRlZmF1bHRfZmFjdG9yeT1saXN0KQogICAgaXRsczogbGlzdFtsaXN0W2Zsb2F0XV0gPSBmaWVsZChkZWZhdWx0X2ZhY3Rvcnk9bGlzdCkKCgpkZWYgYnVpbGRfa3ZfaGVhZGVycygKICAgIHByZWZpbGxfdXJsOiBzdHIsCiAgICBkZWNvZGVfdXJsOiBzdHIsCiAgICBwcmVmaWxsX2t2X3BvcnQ6IGludCA9IDE0NTc5LAogICAgZGVjb2RlX2t2X3BvcnQ6IGludCA9IDE0NTgwLAopIC0+IHR1cGxlW3N0ciwgZGljdFtzdHIsIHN0cl1dOgogICAgIiIiQnVpbGQgcmVxdWVzdF9pZCBhbmQgaGVhZGVycyBmb3IgS1YgdHJhbnNmZXIgYmV0d2VlbiBwcmVmaWxsIGFuZCBkZWNvZGUuIiIiCiAgICBmcm9tIHVybGxpYi5wYXJzZSBpbXBvcnQgdXJscGFyc2UKCiAgICBwcmVmaWxsX3BhcnNlZCA9IHVybHBhcnNlKHByZWZpbGxfdXJsKQogICAgZGVjb2RlX3BhcnNlZCA9IHVybHBhcnNlKGRlY29kZV91cmwpCgogICAgcHJlZmlsbF9ob3N0ID0gcHJlZmlsbF9wYXJzZWQuaG9zdG5hbWUgb3IgImxvY2FsaG9zdCIKICAgIGRlY29kZV9ob3N0ID0gZGVjb2RlX3BhcnNlZC5ob3N0bmFtZSBvciAibG9jYWxob3N0IgoKICAgIHByZWZpbGxfa3ZfYWRkciA9IGYie3ByZWZpbGxfaG9zdH06e3ByZWZpbGxfa3ZfcG9ydH0iCiAgICBkZWNvZGVfa3ZfYWRkciA9IGYie2RlY29kZV9ob3N0fTp7ZGVjb2RlX2t2X3BvcnR9IgoKICAgIHJlcXVlc3RfaWQgPSAoCiAgICAgICAgZiJfX19wcmVmaWxsX2FkZHJfe3ByZWZpbGxfa3ZfYWRkcn1fX19kZWNvZGVfYWRkcl8iCiAgICAgICAgZiJ7ZGVjb2RlX2t2X2FkZHJ9X3t1dWlkLnV1aWQ0KCkuaGV4fSIKICAgICkKCiAgICBoZWFkZXJzID0gewogICAgICAgICJYLVJlcXVlc3QtSWQiOiByZXF1ZXN0X2lkLAogICAgICAgICJYLUtWLVRhcmdldCI6IGYie2RlY29kZV9ob3N0fTp7ZGVjb2RlX3BhcnNlZC5wb3J0IG9yIDgwfSIsCiAgICB9CiAgICBhcGlfa2V5ID0gb3MuZW52aXJvbi5nZXQoIk9QRU5BSV9BUElfS0VZIikKICAgIGlmIGFwaV9rZXk6CiAgICAgICAgaGVhZGVyc1siQXV0aG9yaXphdGlvbiJdID0gZiJCZWFyZXIge2FwaV9rZXl9IgoKICAgIHJldHVybiByZXF1ZXN0X2lkLCBoZWFkZXJzCgoKYXN5bmMgZGVmIGFzeW5jX3JlcXVlc3RfcHJlZmlsbF9vbmx5KAogICAgcmVxdWVzdF9mdW5jX2lucHV0OiBSZXF1ZXN0RnVuY0lucHV0LAogICAgc2Vzc2lvbjogYWlvaHR0cC5DbGllbnRTZXNzaW9uLAogICAgcGJhcjogdHFkbSB8IE5vbmUgPSBOb25lLAopIC0+IFJlcXVlc3RGdW5jT3V0cHV0OgogICAgIiIiU2VuZCBhIHJlcXVlc3Qgd2l0aCBtYXhfdG9rZW5zPTEgdG8gbWVhc3VyZSBwcmVmaWxsIHRpbWUgKFRURlQpLiIiIgogICAgcmV0dXJuIGF3YWl0IEFTWU5DX1JFUVVFU1RfRlVOQ1NbIm9wZW5haS1jaGF0Il0ocmVxdWVzdF9mdW5jX2lucHV0LCBzZXNzaW9uLCBwYmFyKQoKCmFzeW5jIGRlZiBhc3luY19yZXF1ZXN0X2RlY29kZV9hZnRlcl9wcmVmaWxsKAogICAgcmVxdWVzdF9mdW5jX2lucHV0OiBSZXF1ZXN0RnVuY0lucHV0LAogICAgc2Vzc2lvbjogYWlvaHR0cC5DbGllbnRTZXNzaW9uLAogICAgcHJlZmlsbF91cmw6IHN0ciwKICAgIGRlY29kZV91cmw6IHN0ciwKICAgIHByZWZpbGxfa3ZfcG9ydDogaW50LAogICAgZGVjb2RlX2t2X3BvcnQ6IGludCwKICAgIG5peGxfbW9kZTogYm9vbCA9IEZhbHNlLAogICAgcGJhcjogdHFkbSB8IE5vbmUgPSBOb25lLAopIC0+IFJlcXVlc3RGdW5jT3V0cHV0OgogICAgIiIiCiAgICBGaXJzdCBzZW5kIG1heF90b2tlbnM9MSB0byBwcmVmaWxsIG5vZGUgdG8gd2FybSB1cCBLViBjYWNoZSwKICAgIHRoZW4gc2VuZCB0aGUgZnVsbCByZXF1ZXN0IHRvIGRlY29kZSBub2RlIGFuZCBtZWFzdXJlIGRlY29kZSBwZXJmb3JtYW5jZS4KICAgICIiIgogICAgcmVxdWVzdF9pZCwgaGVhZGVycyA9IGJ1aWxkX2t2X2hlYWRlcnMoCiAgICAgICAgcHJlZmlsbF91cmwsIGRlY29kZV91cmwsIHByZWZpbGxfa3ZfcG9ydCwgZGVjb2RlX2t2X3BvcnQKICAgICkKCiAgICAjIDEuIFByZWZpbGwgc3RhZ2U6IG1heF90b2tlbnM9MQogICAgcHJlZmlsbF9pbnB1dCA9IFJlcXVlc3RGdW5jSW5wdXQoCiAgICAgICAgbW9kZWw9cmVxdWVzdF9mdW5jX2lucHV0Lm1vZGVsLAogICAgICAgIG1vZGVsX25hbWU9cmVxdWVzdF9mdW5jX2lucHV0Lm1vZGVsX25hbWUsCiAgICAgICAgcHJvbXB0PXJlcXVlc3RfZnVuY19pbnB1dC5wcm9tcHQsCiAgICAgICAgYXBpX3VybD1wcmVmaWxsX3VybC5yc3RyaXAoIi8iKSArICIvdjEvY2hhdC9jb21wbGV0aW9ucyIsCiAgICAgICAgcHJvbXB0X2xlbj1yZXF1ZXN0X2Z1bmNfaW5wdXQucHJvbXB0X2xlbiwKICAgICAgICBvdXRwdXRfbGVuPTEsCiAgICAgICAgbG9ncHJvYnM9cmVxdWVzdF9mdW5jX2lucHV0LmxvZ3Byb2JzLAogICAgICAgIG11bHRpX21vZGFsX2NvbnRlbnQ9cmVxdWVzdF9mdW5jX2lucHV0Lm11bHRpX21vZGFsX2NvbnRlbnQsCiAgICAgICAgaWdub3JlX2Vvcz1yZXF1ZXN0X2Z1bmNfaW5wdXQuaWdub3JlX2VvcywKICAgICAgICBleHRyYV9oZWFkZXJzPWhlYWRlcnMsCiAgICAgICAgZXh0cmFfYm9keT1yZXF1ZXN0X2Z1bmNfaW5wdXQuZXh0cmFfYm9keSwKICAgICAgICByZXF1ZXN0X2lkPXJlcXVlc3RfaWQsCiAgICApCgogICAgaWYgbml4bF9tb2RlOgogICAgICAgICMgSW4gTklYTCBtb2RlLCBzZW5kIGEgbm9uLXN0cmVhbWluZyBwcmVmaWxsIHJlcXVlc3Qgd2l0aAogICAgICAgICMga3ZfdHJhbnNmZXJfcGFyYW1zIHRvIGdldCB0aGUgcmVtb3RlIEtWIG1ldGFkYXRhIGJhY2suCiAgICAgICAgaW1wb3J0IGFpb2h0dHAKICAgICAgICBwYXlsb2FkID0gewogICAgICAgICAgICAibW9kZWwiOiByZXF1ZXN0X2Z1bmNfaW5wdXQubW9kZWxfbmFtZSBvciByZXF1ZXN0X2Z1bmNfaW5wdXQubW9kZWwsCiAgICAgICAgICAgICJtZXNzYWdlcyI6IFt7InJvbGUiOiAidXNlciIsICJjb250ZW50IjogcmVxdWVzdF9mdW5jX2lucHV0LnByb21wdH1dLAogICAgICAgICAgICAibWF4X2NvbXBsZXRpb25fdG9rZW5zIjogMSwKICAgICAgICAgICAgInN0cmVhbSI6IEZhbHNlLAogICAgICAgICAgICAia3ZfdHJhbnNmZXJfcGFyYW1zIjogewogICAgICAgICAgICAgICAgImRvX3JlbW90ZV9kZWNvZGUiOiBUcnVlLAogICAgICAgICAgICAgICAgImRvX3JlbW90ZV9wcmVmaWxsIjogRmFsc2UsCiAgICAgICAgICAgICAgICAicmVtb3RlX2VuZ2luZV9pZCI6IE5vbmUsCiAgICAgICAgICAgICAgICAicmVtb3RlX2Jsb2NrX2lkcyI6IE5vbmUsCiAgICAgICAgICAgICAgICAicmVtb3RlX2hvc3QiOiBOb25lLAogICAgICAgICAgICAgICAgInJlbW90ZV9wb3J0IjogTm9uZSwKICAgICAgICAgICAgfSwKICAgICAgICB9CiAgICAgICAgdHJ5OgogICAgICAgICAgICBhc3luYyB3aXRoIHNlc3Npb24ucG9zdCgKICAgICAgICAgICAgICAgIHByZWZpbGxfaW5wdXQuYXBpX3VybCwganNvbj1wYXlsb2FkLCBoZWFkZXJzPWhlYWRlcnMKICAgICAgICAgICAgKSBhcyByZXNwOgogICAgICAgICAgICAgICAgaWYgcmVzcC5zdGF0dXMgIT0gMjAwOgogICAgICAgICAgICAgICAgICAgIHRleHQgPSBhd2FpdCByZXNwLnRleHQoKQogICAgICAgICAgICAgICAgICAgIGlmIHBiYXI6CiAgICAgICAgICAgICAgICAgICAgICAgIHBiYXIudXBkYXRlKDEpCiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIFJlcXVlc3RGdW5jT3V0cHV0KAogICAgICAgICAgICAgICAgICAgICAgICBzdWNjZXNzPUZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICBlcnJvcj1mIk5JWEwgcHJlZmlsbCBmYWlsZWQ6IEhUVFAge3Jlc3Auc3RhdHVzfSAtIHt0ZXh0fSIsCiAgICAgICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgZGF0YSA9IGF3YWl0IHJlc3AuanNvbigpCiAgICAgICAgICAgICAgICBrdl90cmFuc2Zlcl9wYXJhbXMgPSBkYXRhLmdldCgia3ZfdHJhbnNmZXJfcGFyYW1zIikKICAgICAgICAgICAgICAgIGlmIG5vdCBrdl90cmFuc2Zlcl9wYXJhbXM6CiAgICAgICAgICAgICAgICAgICAgaWYgcGJhcjoKICAgICAgICAgICAgICAgICAgICAgICAgcGJhci51cGRhdGUoMSkKICAgICAgICAgICAgICAgICAgICByZXR1cm4gUmVxdWVzdEZ1bmNPdXRwdXQoCiAgICAgICAgICAgICAgICAgICAgICAgIHN1Y2Nlc3M9RmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgIGVycm9yPSJOSVhMIHByZWZpbGwgcmVzcG9uc2UgbWlzc2luZyBrdl90cmFuc2Zlcl9wYXJhbXMiLAogICAgICAgICAgICAgICAgICAgICkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgICAgIGlmIHBiYXI6CiAgICAgICAgICAgICAgICBwYmFyLnVwZGF0ZSgxKQogICAgICAgICAgICByZXR1cm4gUmVxdWVzdEZ1bmNPdXRwdXQoCiAgICAgICAgICAgICAgICBzdWNjZXNzPUZhbHNlLAogICAgICAgICAgICAgICAgZXJyb3I9ZiJOSVhMIHByZWZpbGwgcmVxdWVzdCBleGNlcHRpb246IHtlfSIsCiAgICAgICAgICAgICkKICAgIGVsc2U6CiAgICAgICAgcHJlZmlsbF9vdXRwdXQgPSBhd2FpdCBBU1lOQ19SRVFVRVNUX0ZVTkNTWyJvcGVuYWktY2hhdCJdKAogICAgICAgICAgICBwcmVmaWxsX2lucHV0LCBzZXNzaW9uLCBOb25lCiAgICAgICAgKQogICAgICAgIGlmIG5vdCBwcmVmaWxsX291dHB1dC5zdWNjZXNzOgogICAgICAgICAgICBpZiBwYmFyOgogICAgICAgICAgICAgICAgcGJhci51cGRhdGUoMSkKICAgICAgICAgICAgcmV0dXJuIFJlcXVlc3RGdW5jT3V0cHV0KAogICAgICAgICAgICAgICAgc3VjY2Vzcz1GYWxzZSwKICAgICAgICAgICAgICAgIGVycm9yPWYiUHJlZmlsbCBmYWlsZWQ6IHtwcmVmaWxsX291dHB1dC5lcnJvcn0iLAogICAgICAgICAgICApCiAgICAgICAga3ZfdHJhbnNmZXJfcGFyYW1zID0gTm9uZQoKICAgICMgMi4gRGVjb2RlIHN0YWdlOiBmdWxsIG91dHB1dF9sZW4KICAgIGRlY29kZV9leHRyYV9ib2R5ID0gZGljdChyZXF1ZXN0X2Z1bmNfaW5wdXQuZXh0cmFfYm9keSkgaWYgcmVxdWVzdF9mdW5jX2lucHV0LmV4dHJhX2JvZHkgZWxzZSB7fQogICAgaWYgbml4bF9tb2RlIGFuZCBrdl90cmFuc2Zlcl9wYXJhbXM6CiAgICAgICAgZGVjb2RlX2V4dHJhX2JvZHlbImt2X3RyYW5zZmVyX3BhcmFtcyJdID0ga3ZfdHJhbnNmZXJfcGFyYW1zCgogICAgZGVjb2RlX2lucHV0ID0gUmVxdWVzdEZ1bmNJbnB1dCgKICAgICAgICBtb2RlbD1yZXF1ZXN0X2Z1bmNfaW5wdXQubW9kZWwsCiAgICAgICAgbW9kZWxfbmFtZT1yZXF1ZXN0X2Z1bmNfaW5wdXQubW9kZWxfbmFtZSwKICAgICAgICBwcm9tcHQ9cmVxdWVzdF9mdW5jX2lucHV0LnByb21wdCwKICAgICAgICBhcGlfdXJsPWRlY29kZV91cmwucnN0cmlwKCIvIikgKyAiL3YxL2NoYXQvY29tcGxldGlvbnMiLAogICAgICAgIHByb21wdF9sZW49cmVxdWVzdF9mdW5jX2lucHV0LnByb21wdF9sZW4sCiAgICAgICAgb3V0cHV0X2xlbj1yZXF1ZXN0X2Z1bmNfaW5wdXQub3V0cHV0X2xlbiwKICAgICAgICBsb2dwcm9icz1yZXF1ZXN0X2Z1bmNfaW5wdXQubG9ncHJvYnMsCiAgICAgICAgbXVsdGlfbW9kYWxfY29udGVudD1yZXF1ZXN0X2Z1bmNfaW5wdXQubXVsdGlfbW9kYWxfY29udGVudCwKICAgICAgICBpZ25vcmVfZW9zPXJlcXVlc3RfZnVuY19pbnB1dC5pZ25vcmVfZW9zLAogICAgICAgIGV4dHJhX2hlYWRlcnM9aGVhZGVycywKICAgICAgICBleHRyYV9ib2R5PWRlY29kZV9leHRyYV9ib2R5LAogICAgICAgIHJlcXVlc3RfaWQ9cmVxdWVzdF9pZCwKICAgICkKCiAgICBkZWNvZGVfb3V0cHV0ID0gYXdhaXQgQVNZTkNfUkVRVUVTVF9GVU5DU1sib3BlbmFpLWNoYXQiXShkZWNvZGVfaW5wdXQsIHNlc3Npb24sIHBiYXIpCiAgICBpZiBub3QgZGVjb2RlX291dHB1dC5zdWNjZXNzOgogICAgICAgIHJldHVybiBkZWNvZGVfb3V0cHV0CgogICAgIyBGb3IgZGVjb2RlIG1vZGUsIHdlIHdhbnQgdGhlIGRlY29kZSBUVEZUIHRvIHJlcHJlc2VudCB0aGUgdGltZSB1bnRpbAogICAgIyB0aGUgZmlyc3QgZGVjb2RlIHRva2VuICh3aGljaCBpbmNsdWRlcyBLViB0cmFuc2ZlciArIGRlY29kZSBmaXJzdCB0b2tlbikuCiAgICAjIFRoZSB0b3RhbCBsYXRlbmN5IGlzIGZyb20gc2VuZGluZyBkZWNvZGUgcmVxdWVzdCB0byByZWNlaXZpbmcgYWxsIHRva2Vucy4KICAgICMgV2UgcHJlc2VydmUgdGhlIG9yaWdpbmFsIG1ldHJpY3MgYnV0IHJlbmFtZSB0aGVtIGZvciBjbGFyaXR5IGluIHJlcG9ydGluZy4KICAgIHJldHVybiBkZWNvZGVfb3V0cHV0CgoKYXN5bmMgZGVmIGdldF9yZXF1ZXN0KAogICAgaW5wdXRfcmVxdWVzdHM6IGxpc3RbU2FtcGxlUmVxdWVzdF0sCiAgICByZXF1ZXN0X3JhdGU6IGZsb2F0LAogICAgYnVyc3RpbmVzczogZmxvYXQgPSAxLjAsCikgLT4gQXN5bmNHZW5lcmF0b3JbU2FtcGxlUmVxdWVzdCwgTm9uZV06CiAgICAiIiJHZW5lcmF0ZSByZXF1ZXN0cyBhdCBzcGVjaWZpZWQgcmF0ZS4iIiIKICAgIGltcG9ydCBudW1weSBhcyBucAoKICAgIHRvdGFsX3JlcXVlc3RzID0gbGVuKGlucHV0X3JlcXVlc3RzKQogICAgZGVsYXlfdHMgPSBbXQogICAgZm9yIF8gaW4gcmFuZ2UodG90YWxfcmVxdWVzdHMpOgogICAgICAgIGlmIHJlcXVlc3RfcmF0ZSA9PSBmbG9hdCgiaW5mIik6CiAgICAgICAgICAgIGRlbGF5X3RzLmFwcGVuZCgwKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHRoZXRhID0gMS4wIC8gKHJlcXVlc3RfcmF0ZSAqIGJ1cnN0aW5lc3MpCiAgICAgICAgICAgIGRlbGF5X3RzLmFwcGVuZChucC5yYW5kb20uZ2FtbWEoc2hhcGU9YnVyc3RpbmVzcywgc2NhbGU9dGhldGEpKQoKICAgIGZvciBpIGluIHJhbmdlKDEsIGxlbihkZWxheV90cykpOgogICAgICAgIGRlbGF5X3RzW2ldICs9IGRlbGF5X3RzW2kgLSAxXQoKICAgIGlmIHJlcXVlc3RfcmF0ZSAhPSBmbG9hdCgiaW5mIikgYW5kIGRlbGF5X3RzOgogICAgICAgIHRhcmdldF90b3RhbCA9IHRvdGFsX3JlcXVlc3RzIC8gcmVxdWVzdF9yYXRlCiAgICAgICAgaWYgZGVsYXlfdHNbLTFdICE9IDA6CiAgICAgICAgICAgIG5vcm1hbGl6ZV9mYWN0b3IgPSB0YXJnZXRfdG90YWwgLyBkZWxheV90c1stMV0KICAgICAgICAgICAgZGVsYXlfdHMgPSBbZCAqIG5vcm1hbGl6ZV9mYWN0b3IgZm9yIGQgaW4gZGVsYXlfdHNdCgogICAgc3RhcnRfdHMgPSB0aW1lLnRpbWUoKQogICAgZm9yIGksIHJlcXVlc3QgaW4gZW51bWVyYXRlKGlucHV0X3JlcXVlc3RzKToKICAgICAgICBpZiBkZWxheV90c1tpXSA+IDA6CiAgICAgICAgICAgIHNsZWVwX2ludGVydmFsID0gc3RhcnRfdHMgKyBkZWxheV90c1tpXSAtIHRpbWUudGltZSgpCiAgICAgICAgICAgIGlmIHNsZWVwX2ludGVydmFsID4gMDoKICAgICAgICAgICAgICAgIGF3YWl0IGFzeW5jaW8uc2xlZXAoc2xlZXBfaW50ZXJ2YWwpCiAgICAgICAgeWllbGQgcmVxdWVzdAoKCmFzeW5jIGRlZiBiZW5jaG1hcmtfcHJlZmlsbCgKICAgIG1vZGVsX2lkOiBzdHIsCiAgICBtb2RlbF9uYW1lOiBzdHIgfCBOb25lLAogICAgdG9rZW5pemVyOiBBbnksCiAgICBpbnB1dF9yZXF1ZXN0czogbGlzdFtTYW1wbGVSZXF1ZXN0XSwKICAgIHByZWZpbGxfdXJsOiBzdHIsCiAgICByZXF1ZXN0X3JhdGU6IGZsb2F0LAogICAgYnVyc3RpbmVzczogZmxvYXQsCiAgICBtYXhfY29uY3VycmVuY3k6IGludCB8IE5vbmUsCiAgICBkaXNhYmxlX3RxZG06IGJvb2wsCikgLT4gRGlzYWdnTWV0cmljczoKICAgICIiIkJlbmNobWFyayBwcmVmaWxsIG5vZGUgYnkgc2VuZGluZyBtYXhfdG9rZW5zPTEgcmVxdWVzdHMuIiIiCiAgICBjb25uZWN0b3IgPSBhaW9odHRwLlRDUENvbm5lY3RvcigKICAgICAgICBsaW1pdD1tYXhfY29uY3VycmVuY3kgb3IgMCwKICAgICAgICBsaW1pdF9wZXJfaG9zdD1tYXhfY29uY3VycmVuY3kgb3IgMCwKICAgICkKICAgIHNlc3Npb24gPSBhaW9odHRwLkNsaWVudFNlc3Npb24oCiAgICAgICAgY29ubmVjdG9yPWNvbm5lY3RvciwKICAgICAgICB0aW1lb3V0PUFJT0hUVFBfVElNRU9VVCwKICAgICkKCiAgICBwYmFyID0gTm9uZSBpZiBkaXNhYmxlX3RxZG0gZWxzZSB0cWRtKHRvdGFsPWxlbihpbnB1dF9yZXF1ZXN0cykpCiAgICBzZW1hcGhvcmUgPSAoCiAgICAgICAgYXN5bmNpby5TZW1hcGhvcmUobWF4X2NvbmN1cnJlbmN5KQogICAgICAgIGlmIG1heF9jb25jdXJyZW5jeQogICAgICAgIGVsc2UgY29udGV4dGxpYi5udWxsY29udGV4dCgpCiAgICApCgogICAgYXN5bmMgZGVmIGxpbWl0ZWRfcmVxdWVzdChyZXFfaW5wdXQ6IFJlcXVlc3RGdW5jSW5wdXQpIC0+IFJlcXVlc3RGdW5jT3V0cHV0OgogICAgICAgIGFzeW5jIHdpdGggc2VtYXBob3JlOiAgIyB0eXBlOiBpZ25vcmVbYXR0ci1kZWZpbmVkXQogICAgICAgICAgICByZXR1cm4gYXdhaXQgYXN5bmNfcmVxdWVzdF9wcmVmaWxsX29ubHkocmVxX2lucHV0LCBzZXNzaW9uLCBwYmFyKQoKICAgIGFwaV91cmwgPSBwcmVmaWxsX3VybC5yc3RyaXAoIi8iKSArICIvdjEvY2hhdC9jb21wbGV0aW9ucyIKICAgIHRhc2tzOiBsaXN0W2FzeW5jaW8uVGFza10gPSBbXQogICAgc3RhcnRfdGltZSA9IHRpbWUucGVyZl9jb3VudGVyKCkKCiAgICBhc3luYyBmb3IgcmVxdWVzdCBpbiBnZXRfcmVxdWVzdChpbnB1dF9yZXF1ZXN0cywgcmVxdWVzdF9yYXRlLCBidXJzdGluZXNzKToKICAgICAgICByZXFfaW5wdXQgPSBSZXF1ZXN0RnVuY0lucHV0KAogICAgICAgICAgICBtb2RlbD1tb2RlbF9pZCwKICAgICAgICAgICAgbW9kZWxfbmFtZT1tb2RlbF9uYW1lLAogICAgICAgICAgICBwcm9tcHQ9cmVxdWVzdC5wcm9tcHQsCiAgICAgICAgICAgIGFwaV91cmw9YXBpX3VybCwKICAgICAgICAgICAgcHJvbXB0X2xlbj1yZXF1ZXN0LnByb21wdF9sZW4sCiAgICAgICAgICAgIG91dHB1dF9sZW49MSwgICMgcHJlZmlsbCBvbmx5CiAgICAgICAgICAgIGxvZ3Byb2JzPU5vbmUsCiAgICAgICAgICAgIG11bHRpX21vZGFsX2NvbnRlbnQ9cmVxdWVzdC5tdWx0aV9tb2RhbF9kYXRhLAogICAgICAgICAgICBpZ25vcmVfZW9zPUZhbHNlLAogICAgICAgICkKICAgICAgICB0YXNrcy5hcHBlbmQoYXN5bmNpby5jcmVhdGVfdGFzayhsaW1pdGVkX3JlcXVlc3QocmVxX2lucHV0KSkpCgogICAgb3V0cHV0czogbGlzdFtSZXF1ZXN0RnVuY091dHB1dF0gPSBhd2FpdCBhc3luY2lvLmdhdGhlcigqdGFza3MpCiAgICBkdXJhdGlvbiA9IHRpbWUucGVyZl9jb3VudGVyKCkgLSBzdGFydF90aW1lCgogICAgaWYgcGJhcjoKICAgICAgICBwYmFyLmNsb3NlKCkKICAgIGF3YWl0IHNlc3Npb24uY2xvc2UoKQoKICAgIHJldHVybiBfY29tcHV0ZV9tZXRyaWNzKG91dHB1dHMsIGR1cmF0aW9uLCB0b2tlbml6ZXIpCgoKYXN5bmMgZGVmIGJlbmNobWFya19kZWNvZGUoCiAgICBtb2RlbF9pZDogc3RyLAogICAgbW9kZWxfbmFtZTogc3RyIHwgTm9uZSwKICAgIHRva2VuaXplcjogQW55LAogICAgaW5wdXRfcmVxdWVzdHM6IGxpc3RbU2FtcGxlUmVxdWVzdF0sCiAgICBwcmVmaWxsX3VybDogc3RyLAogICAgZGVjb2RlX3VybDogc3RyLAogICAgcHJlZmlsbF9rdl9wb3J0OiBpbnQsCiAgICBkZWNvZGVfa3ZfcG9ydDogaW50LAogICAgcmVxdWVzdF9yYXRlOiBmbG9hdCwKICAgIGJ1cnN0aW5lc3M6IGZsb2F0LAogICAgbWF4X2NvbmN1cnJlbmN5OiBpbnQgfCBOb25lLAogICAgZGlzYWJsZV90cWRtOiBib29sLAogICAgbml4bF9tb2RlOiBib29sID0gRmFsc2UsCikgLT4gRGlzYWdnTWV0cmljczoKICAgICIiIkJlbmNobWFyayBkZWNvZGUgbm9kZSBhZnRlciB3YXJtaW5nIHVwIEtWIGNhY2hlIG9uIHByZWZpbGwgbm9kZS4iIiIKICAgIGNvbm5lY3RvciA9IGFpb2h0dHAuVENQQ29ubmVjdG9yKAogICAgICAgIGxpbWl0PW1heF9jb25jdXJyZW5jeSBvciAwLAogICAgICAgIGxpbWl0X3Blcl9ob3N0PW1heF9jb25jdXJyZW5jeSBvciAwLAogICAgKQogICAgc2Vzc2lvbiA9IGFpb2h0dHAuQ2xpZW50U2Vzc2lvbigKICAgICAgICBjb25uZWN0b3I9Y29ubmVjdG9yLAogICAgICAgIHRpbWVvdXQ9QUlPSFRUUF9USU1FT1VULAogICAgKQoKICAgIHBiYXIgPSBOb25lIGlmIGRpc2FibGVfdHFkbSBlbHNlIHRxZG0odG90YWw9bGVuKGlucHV0X3JlcXVlc3RzKSkKICAgIHNlbWFwaG9yZSA9ICgKICAgICAgICBhc3luY2lvLlNlbWFwaG9yZShtYXhfY29uY3VycmVuY3kpCiAgICAgICAgaWYgbWF4X2NvbmN1cnJlbmN5CiAgICAgICAgZWxzZSBjb250ZXh0bGliLm51bGxjb250ZXh0KCkKICAgICkKCiAgICBhc3luYyBkZWYgbGltaXRlZF9yZXF1ZXN0KHJlcV9pbnB1dDogUmVxdWVzdEZ1bmNJbnB1dCkgLT4gUmVxdWVzdEZ1bmNPdXRwdXQ6CiAgICAgICAgYXN5bmMgd2l0aCBzZW1hcGhvcmU6ICAjIHR5cGU6IGlnbm9yZVthdHRyLWRlZmluZWRdCiAgICAgICAgICAgIHJldHVybiBhd2FpdCBhc3luY19yZXF1ZXN0X2RlY29kZV9hZnRlcl9wcmVmaWxsKAogICAgICAgICAgICAgICAgcmVxX2lucHV0LAogICAgICAgICAgICAgICAgc2Vzc2lvbiwKICAgICAgICAgICAgICAgIHByZWZpbGxfdXJsLAogICAgICAgICAgICAgICAgZGVjb2RlX3VybCwKICAgICAgICAgICAgICAgIHByZWZpbGxfa3ZfcG9ydCwKICAgICAgICAgICAgICAgIGRlY29kZV9rdl9wb3J0LAogICAgICAgICAgICAgICAgbml4bF9tb2RlLAogICAgICAgICAgICAgICAgcGJhciwKICAgICAgICAgICAgKQoKICAgIHRhc2tzOiBsaXN0W2FzeW5jaW8uVGFza10gPSBbXQogICAgc3RhcnRfdGltZSA9IHRpbWUucGVyZl9jb3VudGVyKCkKCiAgICBhc3luYyBmb3IgcmVxdWVzdCBpbiBnZXRfcmVxdWVzdChpbnB1dF9yZXF1ZXN0cywgcmVxdWVzdF9yYXRlLCBidXJzdGluZXNzKToKICAgICAgICByZXFfaW5wdXQgPSBSZXF1ZXN0RnVuY0lucHV0KAogICAgICAgICAgICBtb2RlbD1tb2RlbF9pZCwKICAgICAgICAgICAgbW9kZWxfbmFtZT1tb2RlbF9uYW1lLAogICAgICAgICAgICBwcm9tcHQ9cmVxdWVzdC5wcm9tcHQsCiAgICAgICAgICAgIGFwaV91cmw9IiIsICAjIG5vdCB1c2VkIGRpcmVjdGx5CiAgICAgICAgICAgIHByb21wdF9sZW49cmVxdWVzdC5wcm9tcHRfbGVuLAogICAgICAgICAgICBvdXRwdXRfbGVuPXJlcXVlc3QuZXhwZWN0ZWRfb3V0cHV0X2xlbiBvciAxMjgsCiAgICAgICAgICAgIGxvZ3Byb2JzPU5vbmUsCiAgICAgICAgICAgIG11bHRpX21vZGFsX2NvbnRlbnQ9cmVxdWVzdC5tdWx0aV9tb2RhbF9kYXRhLAogICAgICAgICAgICBpZ25vcmVfZW9zPUZhbHNlLAogICAgICAgICkKICAgICAgICB0YXNrcy5hcHBlbmQoYXN5bmNpby5jcmVhdGVfdGFzayhsaW1pdGVkX3JlcXVlc3QocmVxX2lucHV0KSkpCgogICAgb3V0cHV0czogbGlzdFtSZXF1ZXN0RnVuY091dHB1dF0gPSBhd2FpdCBhc3luY2lvLmdhdGhlcigqdGFza3MpCiAgICBkdXJhdGlvbiA9IHRpbWUucGVyZl9jb3VudGVyKCkgLSBzdGFydF90aW1lCgogICAgaWYgcGJhcjoKICAgICAgICBwYmFyLmNsb3NlKCkKICAgIGF3YWl0IHNlc3Npb24uY2xvc2UoKQoKICAgIHJldHVybiBfY29tcHV0ZV9tZXRyaWNzKG91dHB1dHMsIGR1cmF0aW9uLCB0b2tlbml6ZXIpCgoKYXN5bmMgZGVmIGJlbmNobWFya19lMmUoCiAgICBtb2RlbF9pZDogc3RyLAogICAgbW9kZWxfbmFtZTogc3RyIHwgTm9uZSwKICAgIHRva2VuaXplcjogQW55LAogICAgaW5wdXRfcmVxdWVzdHM6IGxpc3RbU2FtcGxlUmVxdWVzdF0sCiAgICBwcm94eV91cmw6IHN0ciwKICAgIHJlcXVlc3RfcmF0ZTogZmxvYXQsCiAgICBidXJzdGluZXNzOiBmbG9hdCwKICAgIG1heF9jb25jdXJyZW5jeTogaW50IHwgTm9uZSwKICAgIGRpc2FibGVfdHFkbTogYm9vbCwKKSAtPiBEaXNhZ2dNZXRyaWNzOgogICAgIiIiQmVuY2htYXJrIGVuZC10by1lbmQgdmlhIHRoZSBkaXNhZ2dyZWdhdGlvbiBwcm94eS4iIiIKICAgIGNvbm5lY3RvciA9IGFpb2h0dHAuVENQQ29ubmVjdG9yKAogICAgICAgIGxpbWl0PW1heF9jb25jdXJyZW5jeSBvciAwLAogICAgICAgIGxpbWl0X3Blcl9ob3N0PW1heF9jb25jdXJyZW5jeSBvciAwLAogICAgKQogICAgc2Vzc2lvbiA9IGFpb2h0dHAuQ2xpZW50U2Vzc2lvbigKICAgICAgICBjb25uZWN0b3I9Y29ubmVjdG9yLAogICAgICAgIHRpbWVvdXQ9QUlPSFRUUF9USU1FT1VULAogICAgKQoKICAgIHBiYXIgPSBOb25lIGlmIGRpc2FibGVfdHFkbSBlbHNlIHRxZG0odG90YWw9bGVuKGlucHV0X3JlcXVlc3RzKSkKICAgIHNlbWFwaG9yZSA9ICgKICAgICAgICBhc3luY2lvLlNlbWFwaG9yZShtYXhfY29uY3VycmVuY3kpCiAgICAgICAgaWYgbWF4X2NvbmN1cnJlbmN5CiAgICAgICAgZWxzZSBjb250ZXh0bGliLm51bGxjb250ZXh0KCkKICAgICkKCiAgICBhc3luYyBkZWYgbGltaXRlZF9yZXF1ZXN0KHJlcV9pbnB1dDogUmVxdWVzdEZ1bmNJbnB1dCkgLT4gUmVxdWVzdEZ1bmNPdXRwdXQ6CiAgICAgICAgYXN5bmMgd2l0aCBzZW1hcGhvcmU6ICAjIHR5cGU6IGlnbm9yZVthdHRyLWRlZmluZWRdCiAgICAgICAgICAgIHJldHVybiBhd2FpdCBBU1lOQ19SRVFVRVNUX0ZVTkNTWyJvcGVuYWktY2hhdCJdKHJlcV9pbnB1dCwgc2Vzc2lvbiwgcGJhcikKCiAgICBhcGlfdXJsID0gcHJveHlfdXJsLnJzdHJpcCgiLyIpICsgIi92MS9jaGF0L2NvbXBsZXRpb25zIgogICAgdGFza3M6IGxpc3RbYXN5bmNpby5UYXNrXSA9IFtdCiAgICBzdGFydF90aW1lID0gdGltZS5wZXJmX2NvdW50ZXIoKQoKICAgIGFzeW5jIGZvciByZXF1ZXN0IGluIGdldF9yZXF1ZXN0KGlucHV0X3JlcXVlc3RzLCByZXF1ZXN0X3JhdGUsIGJ1cnN0aW5lc3MpOgogICAgICAgIHJlcV9pbnB1dCA9IFJlcXVlc3RGdW5jSW5wdXQoCiAgICAgICAgICAgIG1vZGVsPW1vZGVsX2lkLAogICAgICAgICAgICBtb2RlbF9uYW1lPW1vZGVsX25hbWUsCiAgICAgICAgICAgIHByb21wdD1yZXF1ZXN0LnByb21wdCwKICAgICAgICAgICAgYXBpX3VybD1hcGlfdXJsLAogICAgICAgICAgICBwcm9tcHRfbGVuPXJlcXVlc3QucHJvbXB0X2xlbiwKICAgICAgICAgICAgb3V0cHV0X2xlbj1yZXF1ZXN0LmV4cGVjdGVkX291dHB1dF9sZW4gb3IgMTI4LAogICAgICAgICAgICBsb2dwcm9icz1Ob25lLAogICAgICAgICAgICBtdWx0aV9tb2RhbF9jb250ZW50PXJlcXVlc3QubXVsdGlfbW9kYWxfZGF0YSwKICAgICAgICAgICAgaWdub3JlX2Vvcz1GYWxzZSwKICAgICAgICApCiAgICAgICAgdGFza3MuYXBwZW5kKGFzeW5jaW8uY3JlYXRlX3Rhc2sobGltaXRlZF9yZXF1ZXN0KHJlcV9pbnB1dCkpKQoKICAgIG91dHB1dHM6IGxpc3RbUmVxdWVzdEZ1bmNPdXRwdXRdID0gYXdhaXQgYXN5bmNpby5nYXRoZXIoKnRhc2tzKQogICAgZHVyYXRpb24gPSB0aW1lLnBlcmZfY291bnRlcigpIC0gc3RhcnRfdGltZQoKICAgIGlmIHBiYXI6CiAgICAgICAgcGJhci5jbG9zZSgpCiAgICBhd2FpdCBzZXNzaW9uLmNsb3NlKCkKCiAgICByZXR1cm4gX2NvbXB1dGVfbWV0cmljcyhvdXRwdXRzLCBkdXJhdGlvbiwgdG9rZW5pemVyKQoKCmRlZiBfY29tcHV0ZV9tZXRyaWNzKAogICAgb3V0cHV0czogbGlzdFtSZXF1ZXN0RnVuY091dHB1dF0sCiAgICBkdXJhdGlvbjogZmxvYXQsCiAgICB0b2tlbml6ZXI6IEFueSwKKSAtPiBEaXNhZ2dNZXRyaWNzOgogICAgdHRmdHM6IGxpc3RbZmxvYXRdID0gW10KICAgIHRwb3RzOiBsaXN0W2Zsb2F0XSA9IFtdCiAgICBsYXRlbmNpZXM6IGxpc3RbZmxvYXRdID0gW10KICAgIGl0bHM6IGxpc3RbbGlzdFtmbG9hdF1dID0gW10KICAgIHRvdGFsX2lucHV0ID0gMAogICAgdG90YWxfb3V0cHV0ID0gMAogICAgY29tcGxldGVkID0gMAogICAgZmFpbGVkID0gMAoKICAgIGZvciBvdXRwdXQgaW4gb3V0cHV0czoKICAgICAgICBpZiBvdXRwdXQuc3VjY2VzczoKICAgICAgICAgICAgY29tcGxldGVkICs9IDEKICAgICAgICAgICAgdG90YWxfaW5wdXQgKz0gb3V0cHV0LnByb21wdF9sZW4KICAgICAgICAgICAgb3V0cHV0X2xlbiA9IG91dHB1dC5vdXRwdXRfdG9rZW5zCiAgICAgICAgICAgIGlmIG5vdCBvdXRwdXRfbGVuIGFuZCB0b2tlbml6ZXIgaXMgbm90IE5vbmU6CiAgICAgICAgICAgICAgICBvdXRwdXRfbGVuID0gbGVuKAogICAgICAgICAgICAgICAgICAgIHRva2VuaXplcihvdXRwdXQuZ2VuZXJhdGVkX3RleHQsIGFkZF9zcGVjaWFsX3Rva2Vucz1GYWxzZSkuaW5wdXRfaWRzCiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgIHRvdGFsX291dHB1dCArPSBvdXRwdXRfbGVuIG9yIDAKICAgICAgICAgICAgdHRmdHMuYXBwZW5kKG91dHB1dC50dGZ0KQogICAgICAgICAgICBsYXRlbmNpZXMuYXBwZW5kKG91dHB1dC5sYXRlbmN5KQogICAgICAgICAgICBpdGxzLmFwcGVuZChvdXRwdXQuaXRsKQogICAgICAgICAgICBpZiBvdXRwdXRfbGVuIGFuZCBvdXRwdXRfbGVuID4gMToKICAgICAgICAgICAgICAgIHRwb3QgPSAob3V0cHV0LmxhdGVuY3kgLSBvdXRwdXQudHRmdCkgLyAob3V0cHV0X2xlbiAtIDEpCiAgICAgICAgICAgICAgICB0cG90cy5hcHBlbmQodHBvdCkKICAgICAgICBlbHNlOgogICAgICAgICAgICBmYWlsZWQgKz0gMQoKICAgIHJldHVybiBEaXNhZ2dNZXRyaWNzKAogICAgICAgIGNvbXBsZXRlZD1jb21wbGV0ZWQsCiAgICAgICAgZmFpbGVkPWZhaWxlZCwKICAgICAgICB0b3RhbF9pbnB1dF90b2tlbnM9dG90YWxfaW5wdXQsCiAgICAgICAgdG90YWxfb3V0cHV0X3Rva2Vucz10b3RhbF9vdXRwdXQsCiAgICAgICAgbWVhbl90dGZ0X21zPW5wLm1lYW4odHRmdHMpICogMTAwMCBpZiB0dGZ0cyBlbHNlIDAuMCwKICAgICAgICBtZWRpYW5fdHRmdF9tcz1ucC5tZWRpYW4odHRmdHMpICogMTAwMCBpZiB0dGZ0cyBlbHNlIDAuMCwKICAgICAgICBzdGRfdHRmdF9tcz1ucC5zdGQodHRmdHMpICogMTAwMCBpZiB0dGZ0cyBlbHNlIDAuMCwKICAgICAgICBwOTlfdHRmdF9tcz1ucC5wZXJjZW50aWxlKHR0ZnRzLCA5OSkgKiAxMDAwIGlmIHR0ZnRzIGVsc2UgMC4wLAogICAgICAgIG1lYW5fdHBvdF9tcz1ucC5tZWFuKHRwb3RzKSAqIDEwMDAgaWYgdHBvdHMgZWxzZSAwLjAsCiAgICAgICAgbWVkaWFuX3Rwb3RfbXM9bnAubWVkaWFuKHRwb3RzKSAqIDEwMDAgaWYgdHBvdHMgZWxzZSAwLjAsCiAgICAgICAgc3RkX3Rwb3RfbXM9bnAuc3RkKHRwb3RzKSAqIDEwMDAgaWYgdHBvdHMgZWxzZSAwLjAsCiAgICAgICAgcDk5X3Rwb3RfbXM9bnAucGVyY2VudGlsZSh0cG90cywgOTkpICogMTAwMCBpZiB0cG90cyBlbHNlIDAuMCwKICAgICAgICByZXF1ZXN0X3Rocm91Z2hwdXQ9Y29tcGxldGVkIC8gZHVyYXRpb24gaWYgZHVyYXRpb24gPiAwIGVsc2UgMC4wLAogICAgICAgIGlucHV0X3Rva2VuX3Rocm91Z2hwdXQ9dG90YWxfaW5wdXQgLyBkdXJhdGlvbiBpZiBkdXJhdGlvbiA+IDAgZWxzZSAwLjAsCiAgICAgICAgb3V0cHV0X3Rva2VuX3Rocm91Z2hwdXQ9dG90YWxfb3V0cHV0IC8gZHVyYXRpb24gaWYgZHVyYXRpb24gPiAwIGVsc2UgMC4wLAogICAgICAgIHR0ZnRzPXR0ZnRzLAogICAgICAgIHRwb3RzPXRwb3RzLAogICAgICAgIGxhdGVuY2llcz1sYXRlbmNpZXMsCiAgICAgICAgaXRscz1pdGxzLAogICAgKQoKCmRlZiBwcmludF9tZXRyaWNzKG1ldHJpY3M6IERpc2FnZ01ldHJpY3MsIG1vZGU6IHN0cikgLT4gTm9uZToKICAgIHByaW50KGYiXG57Jz0nKjYwfSIpCiAgICBwcmludChmIiAgRGlzYWdncmVnYXRlZCBTZXJ2aW5nIEJlbmNobWFyayBSZXN1bHQgW3ttb2RlLnVwcGVyKCl9XSIpCiAgICBwcmludChmInsnPScqNjB9IikKICAgIHByaW50KGYiICBTdWNjZXNzZnVsIHJlcXVlc3RzOiAgICAgICAge21ldHJpY3MuY29tcGxldGVkfSIpCiAgICBwcmludChmIiAgRmFpbGVkIHJlcXVlc3RzOiAgICAgICAgICAgIHttZXRyaWNzLmZhaWxlZH0iKQogICAgcHJpbnQoZiIgIFRvdGFsIGlucHV0IHRva2VuczogICAgICAgICB7bWV0cmljcy50b3RhbF9pbnB1dF90b2tlbnN9IikKICAgIHByaW50KGYiICBUb3RhbCBvdXRwdXQgdG9rZW5zOiAgICAgICAge21ldHJpY3MudG90YWxfb3V0cHV0X3Rva2Vuc30iKQogICAgcHJpbnQoZiJ7Jy0nKjYwfSIpCiAgICBwcmludChmIiAgUmVxdWVzdCB0aHJvdWdocHV0OiAgICAgICAgIHttZXRyaWNzLnJlcXVlc3RfdGhyb3VnaHB1dDouMmZ9IHJlcS9zIikKICAgIHByaW50KGYiICBJbnB1dCB0b2tlbiB0aHJvdWdocHV0OiAgICAge21ldHJpY3MuaW5wdXRfdG9rZW5fdGhyb3VnaHB1dDouMmZ9IHRvay9zIikKICAgIHByaW50KGYiICBPdXRwdXQgdG9rZW4gdGhyb3VnaHB1dDogICAge21ldHJpY3Mub3V0cHV0X3Rva2VuX3Rocm91Z2hwdXQ6LjJmfSB0b2svcyIpCiAgICBwcmludChmInsnLScqNjB9IikKICAgIHByaW50KGYiICBUVEZUIChtcykiKQogICAgcHJpbnQoZiIgICAgTWVhbjogICB7bWV0cmljcy5tZWFuX3R0ZnRfbXM6LjJmfSIpCiAgICBwcmludChmIiAgICBNZWRpYW46IHttZXRyaWNzLm1lZGlhbl90dGZ0X21zOi4yZn0iKQogICAgcHJpbnQoZiIgICAgU3RkOiAgICB7bWV0cmljcy5zdGRfdHRmdF9tczouMmZ9IikKICAgIHByaW50KGYiICAgIFA5OTogICAge21ldHJpY3MucDk5X3R0ZnRfbXM6LjJmfSIpCiAgICBwcmludChmInsnLScqNjB9IikKICAgIHByaW50KGYiICBUUE9UIChtcykiKQogICAgcHJpbnQoZiIgICAgTWVhbjogICB7bWV0cmljcy5tZWFuX3Rwb3RfbXM6LjJmfSIpCiAgICBwcmludChmIiAgICBNZWRpYW46IHttZXRyaWNzLm1lZGlhbl90cG90X21zOi4yZn0iKQogICAgcHJpbnQoZiIgICAgU3RkOiAgICB7bWV0cmljcy5zdGRfdHBvdF9tczouMmZ9IikKICAgIHByaW50KGYiICAgIFA5OTogICAge21ldHJpY3MucDk5X3Rwb3RfbXM6LjJmfSIpCiAgICBwcmludChmInsnPScqNjB9XG4iKQoKCmRlZiBhZGRfY2xpX2FyZ3MocGFyc2VyOiBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcikgLT4gTm9uZToKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tbW9kZSIsCiAgICAgICAgdHlwZT1zdHIsCiAgICAgICAgcmVxdWlyZWQ9VHJ1ZSwKICAgICAgICBjaG9pY2VzPVsicHJlZmlsbCIsICJkZWNvZGUiLCAiZTJlIl0sCiAgICAgICAgaGVscD0iQmVuY2htYXJrIG1vZGU6IHByZWZpbGwgKHByZWZpbGwgbm9kZSBvbmx5KSwgIgogICAgICAgICJkZWNvZGUgKGRlY29kZSBub2RlIGFmdGVyIHByZWZpbGwgd2FybXVwKSwgZTJlICh2aWEgcHJveHkpIiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tcHJlZmlsbC11cmwiLAogICAgICAgIHR5cGU9c3RyLAogICAgICAgIGRlZmF1bHQ9Imh0dHA6Ly9sb2NhbGhvc3Q6ODEwMCIsCiAgICAgICAgaGVscD0iUHJlZmlsbCBzZXJ2aWNlIFVSTCIsCiAgICApCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KAogICAgICAgICItLWRlY29kZS11cmwiLAogICAgICAgIHR5cGU9c3RyLAogICAgICAgIGRlZmF1bHQ9Imh0dHA6Ly9sb2NhbGhvc3Q6ODIwMCIsCiAgICAgICAgaGVscD0iRGVjb2RlIHNlcnZpY2UgVVJMIiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tcHJveHktdXJsIiwKICAgICAgICB0eXBlPXN0ciwKICAgICAgICBkZWZhdWx0PSJodHRwOi8vbG9jYWxob3N0OjgwMDAiLAogICAgICAgIGhlbHA9IlByb3h5IHNlcnZpY2UgVVJMIChmb3IgZTJlIG1vZGUpIiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tcHJlZmlsbC1rdi1wb3J0IiwKICAgICAgICB0eXBlPWludCwKICAgICAgICBkZWZhdWx0PTE0NTc5LAogICAgICAgIGhlbHA9IlByZWZpbGwgS1YgdHJhbnNmZXIgcG9ydCIsCiAgICApCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KAogICAgICAgICItLWRlY29kZS1rdi1wb3J0IiwKICAgICAgICB0eXBlPWludCwKICAgICAgICBkZWZhdWx0PTE0NTgwLAogICAgICAgIGhlbHA9IkRlY29kZSBLViB0cmFuc2ZlciBwb3J0IiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tbW9kZWwiLAogICAgICAgIHR5cGU9c3RyLAogICAgICAgIHJlcXVpcmVkPVRydWUsCiAgICAgICAgaGVscD0iTW9kZWwgbmFtZSIsCiAgICApCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KAogICAgICAgICItLXJlcXVlc3QtcmF0ZSIsCiAgICAgICAgdHlwZT1mbG9hdCwKICAgICAgICBkZWZhdWx0PWZsb2F0KCJpbmYiKSwKICAgICAgICBoZWxwPSJSZXF1ZXN0IHJhdGUgaW4gcmVxL3MgKGRlZmF1bHQ6IGluZiA9IGJ1cnN0KSIsCiAgICApCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KAogICAgICAgICItLWJ1cnN0aW5lc3MiLAogICAgICAgIHR5cGU9ZmxvYXQsCiAgICAgICAgZGVmYXVsdD0xLjAsCiAgICAgICAgaGVscD0iQnVyc3RpbmVzcyBmYWN0b3IgKDEuMCA9IFBvaXNzb24pIiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tbWF4LWNvbmN1cnJlbmN5IiwKICAgICAgICB0eXBlPWludCwKICAgICAgICBkZWZhdWx0PU5vbmUsCiAgICAgICAgaGVscD0iTWF4aW11bSBjb25jdXJyZW50IHJlcXVlc3RzIiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tZGlzYWJsZS10cWRtIiwKICAgICAgICBhY3Rpb249InN0b3JlX3RydWUiLAogICAgICAgIGhlbHA9IkRpc2FibGUgcHJvZ3Jlc3MgYmFyIiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tc2F2ZS1yZXN1bHQiLAogICAgICAgIGFjdGlvbj0ic3RvcmVfdHJ1ZSIsCiAgICAgICAgaGVscD0iU2F2ZSByZXN1bHRzIHRvIEpTT04gZmlsZSIsCiAgICApCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KAogICAgICAgICItLXJlc3VsdC1kaXIiLAogICAgICAgIHR5cGU9c3RyLAogICAgICAgIGRlZmF1bHQ9Ii4iLAogICAgICAgIGhlbHA9IkRpcmVjdG9yeSB0byBzYXZlIHJlc3VsdHMiLAogICAgKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgKICAgICAgICAiLS1yZXN1bHQtZmlsZW5hbWUiLAogICAgICAgIHR5cGU9c3RyLAogICAgICAgIGRlZmF1bHQ9Tm9uZSwKICAgICAgICBoZWxwPSJSZXN1bHQgZmlsZW5hbWUiLAogICAgKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgKICAgICAgICAiLS10b2tlbml6ZXIiLAogICAgICAgIHR5cGU9c3RyLAogICAgICAgIGRlZmF1bHQ9Tm9uZSwKICAgICAgICBoZWxwPSJUb2tlbml6ZXIgbmFtZSBvciBwYXRoIChkZWZhdWx0cyB0byBtb2RlbCkiLAogICAgKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgKICAgICAgICAiLS1uaXhsLW1vZGUiLAogICAgICAgIGFjdGlvbj0ic3RvcmVfdHJ1ZSIsCiAgICAgICAgaGVscD0iRW5hYmxlIE5JWEwgbW9kZSBmb3IgZGVjb2RlIGJlbmNobWFyayAoaW5qZWN0cyBrdl90cmFuc2Zlcl9wYXJhbXMpIiwKICAgICkKCgphc3luYyBkZWYgbWFpbihhcmdzOiBhcmdwYXJzZS5OYW1lc3BhY2UpIC0+IGRpY3Rbc3RyLCBBbnldOgogICAgIyBMb2FkIHRva2VuaXplcgogICAgdG9rZW5pemVyX25hbWUgPSBhcmdzLnRva2VuaXplciBvciBhcmdzLm1vZGVsCiAgICB0b2tlbml6ZXIgPSBnZXRfdG9rZW5pemVyKHRva2VuaXplcl9uYW1lLCB0cnVzdF9yZW1vdGVfY29kZT1UcnVlKQoKICAgICMgTG9hZCBkYXRhc2V0CiAgICBpbnB1dF9yZXF1ZXN0cyA9IGdldF9zYW1wbGVzKGFyZ3MsIHRva2VuaXplcikKICAgIHByaW50KGYiTG9hZGVkIHtsZW4oaW5wdXRfcmVxdWVzdHMpfSByZXF1ZXN0cyBmcm9tIGRhdGFzZXQiKQoKICAgICMgUnVuIGJlbmNobWFyawogICAgaWYgYXJncy5tb2RlID09ICJwcmVmaWxsIjoKICAgICAgICBtZXRyaWNzID0gYXdhaXQgYmVuY2htYXJrX3ByZWZpbGwoCiAgICAgICAgICAgIG1vZGVsX2lkPWFyZ3MubW9kZWwsCiAgICAgICAgICAgIG1vZGVsX25hbWU9Tm9uZSwKICAgICAgICAgICAgdG9rZW5pemVyPXRva2VuaXplciwKICAgICAgICAgICAgaW5wdXRfcmVxdWVzdHM9aW5wdXRfcmVxdWVzdHMsCiAgICAgICAgICAgIHByZWZpbGxfdXJsPWFyZ3MucHJlZmlsbF91cmwsCiAgICAgICAgICAgIHJlcXVlc3RfcmF0ZT1hcmdzLnJlcXVlc3RfcmF0ZSwKICAgICAgICAgICAgYnVyc3RpbmVzcz1hcmdzLmJ1cnN0aW5lc3MsCiAgICAgICAgICAgIG1heF9jb25jdXJyZW5jeT1hcmdzLm1heF9jb25jdXJyZW5jeSwKICAgICAgICAgICAgZGlzYWJsZV90cWRtPWFyZ3MuZGlzYWJsZV90cWRtLAogICAgICAgICkKICAgIGVsaWYgYXJncy5tb2RlID09ICJkZWNvZGUiOgogICAgICAgIG1ldHJpY3MgPSBhd2FpdCBiZW5jaG1hcmtfZGVjb2RlKAogICAgICAgICAgICBtb2RlbF9pZD1hcmdzLm1vZGVsLAogICAgICAgICAgICBtb2RlbF9uYW1lPU5vbmUsCiAgICAgICAgICAgIHRva2VuaXplcj10b2tlbml6ZXIsCiAgICAgICAgICAgIGlucHV0X3JlcXVlc3RzPWlucHV0X3JlcXVlc3RzLAogICAgICAgICAgICBwcmVmaWxsX3VybD1hcmdzLnByZWZpbGxfdXJsLAogICAgICAgICAgICBkZWNvZGVfdXJsPWFyZ3MuZGVjb2RlX3VybCwKICAgICAgICAgICAgcHJlZmlsbF9rdl9wb3J0PWFyZ3MucHJlZmlsbF9rdl9wb3J0LAogICAgICAgICAgICBkZWNvZGVfa3ZfcG9ydD1hcmdzLmRlY29kZV9rdl9wb3J0LAogICAgICAgICAgICByZXF1ZXN0X3JhdGU9YXJncy5yZXF1ZXN0X3JhdGUsCiAgICAgICAgICAgIGJ1cnN0aW5lc3M9YXJncy5idXJzdGluZXNzLAogICAgICAgICAgICBtYXhfY29uY3VycmVuY3k9YXJncy5tYXhfY29uY3VycmVuY3ksCiAgICAgICAgICAgIGRpc2FibGVfdHFkbT1hcmdzLmRpc2FibGVfdHFkbSwKICAgICAgICAgICAgbml4bF9tb2RlPWFyZ3Mubml4bF9tb2RlLAogICAgICAgICkKICAgIGVsc2U6ICAjIGUyZQogICAgICAgIG1ldHJpY3MgPSBhd2FpdCBiZW5jaG1hcmtfZTJlKAogICAgICAgICAgICBtb2RlbF9pZD1hcmdzLm1vZGVsLAogICAgICAgICAgICBtb2RlbF9uYW1lPU5vbmUsCiAgICAgICAgICAgIHRva2VuaXplcj10b2tlbml6ZXIsCiAgICAgICAgICAgIGlucHV0X3JlcXVlc3RzPWlucHV0X3JlcXVlc3RzLAogICAgICAgICAgICBwcm94eV91cmw9YXJncy5wcm94eV91cmwsCiAgICAgICAgICAgIHJlcXVlc3RfcmF0ZT1hcmdzLnJlcXVlc3RfcmF0ZSwKICAgICAgICAgICAgYnVyc3RpbmVzcz1hcmdzLmJ1cnN0aW5lc3MsCiAgICAgICAgICAgIG1heF9jb25jdXJyZW5jeT1hcmdzLm1heF9jb25jdXJyZW5jeSwKICAgICAgICAgICAgZGlzYWJsZV90cWRtPWFyZ3MuZGlzYWJsZV90cWRtLAogICAgICAgICkKCiAgICBwcmludF9tZXRyaWNzKG1ldHJpY3MsIGFyZ3MubW9kZSkKCiAgICAjIFNhdmUgcmVzdWx0cwogICAgcmVzdWx0OiBkaWN0W3N0ciwgQW55XSA9IHsKICAgICAgICAibW9kZSI6IGFyZ3MubW9kZSwKICAgICAgICAibW9kZWwiOiBhcmdzLm1vZGVsLAogICAgICAgICJudW1fcHJvbXB0cyI6IGxlbihpbnB1dF9yZXF1ZXN0cyksCiAgICAgICAgInJlcXVlc3RfcmF0ZSI6IGFyZ3MucmVxdWVzdF9yYXRlLAogICAgICAgICJjb21wbGV0ZWQiOiBtZXRyaWNzLmNvbXBsZXRlZCwKICAgICAgICAiZmFpbGVkIjogbWV0cmljcy5mYWlsZWQsCiAgICAgICAgInRvdGFsX2lucHV0X3Rva2VucyI6IG1ldHJpY3MudG90YWxfaW5wdXRfdG9rZW5zLAogICAgICAgICJ0b3RhbF9vdXRwdXRfdG9rZW5zIjogbWV0cmljcy50b3RhbF9vdXRwdXRfdG9rZW5zLAogICAgICAgICJyZXF1ZXN0X3Rocm91Z2hwdXQiOiBtZXRyaWNzLnJlcXVlc3RfdGhyb3VnaHB1dCwKICAgICAgICAiaW5wdXRfdG9rZW5fdGhyb3VnaHB1dCI6IG1ldHJpY3MuaW5wdXRfdG9rZW5fdGhyb3VnaHB1dCwKICAgICAgICAib3V0cHV0X3Rva2VuX3Rocm91Z2hwdXQiOiBtZXRyaWNzLm91dHB1dF90b2tlbl90aHJvdWdocHV0LAogICAgICAgICJtZWFuX3R0ZnRfbXMiOiBtZXRyaWNzLm1lYW5fdHRmdF9tcywKICAgICAgICAibWVkaWFuX3R0ZnRfbXMiOiBtZXRyaWNzLm1lZGlhbl90dGZ0X21zLAogICAgICAgICJzdGRfdHRmdF9tcyI6IG1ldHJpY3Muc3RkX3R0ZnRfbXMsCiAgICAgICAgInA5OV90dGZ0X21zIjogbWV0cmljcy5wOTlfdHRmdF9tcywKICAgICAgICAibWVhbl90cG90X21zIjogbWV0cmljcy5tZWFuX3Rwb3RfbXMsCiAgICAgICAgIm1lZGlhbl90cG90X21zIjogbWV0cmljcy5tZWRpYW5fdHBvdF9tcywKICAgICAgICAic3RkX3Rwb3RfbXMiOiBtZXRyaWNzLnN0ZF90cG90X21zLAogICAgICAgICJwOTlfdHBvdF9tcyI6IG1ldHJpY3MucDk5X3Rwb3RfbXMsCiAgICAgICAgInR0ZnRzIjogbWV0cmljcy50dGZ0cywKICAgICAgICAidHBvdHMiOiBtZXRyaWNzLnRwb3RzLAogICAgfQoKICAgIGlmIGFyZ3Muc2F2ZV9yZXN1bHQ6CiAgICAgICAgaW1wb3J0IG9zCgogICAgICAgIG9zLm1ha2VkaXJzKGFyZ3MucmVzdWx0X2RpciwgZXhpc3Rfb2s9VHJ1ZSkKICAgICAgICBmaWxlbmFtZSA9IGFyZ3MucmVzdWx0X2ZpbGVuYW1lIG9yIGYiZGlzYWdnX3thcmdzLm1vZGV9X3Jlc3VsdC5qc29uIgogICAgICAgIGZpbGVwYXRoID0gb3MucGF0aC5qb2luKGFyZ3MucmVzdWx0X2RpciwgZmlsZW5hbWUpCiAgICAgICAgd2l0aCBvcGVuKGZpbGVwYXRoLCAidyIpIGFzIGY6CiAgICAgICAgICAgIGpzb24uZHVtcChyZXN1bHQsIGYsIGluZGVudD0yKQogICAgICAgIHByaW50KGYiUmVzdWx0cyBzYXZlZCB0byB7ZmlsZXBhdGh9IikKCiAgICByZXR1cm4gcmVzdWx0CgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIHBhcnNlciA9IEZsZXhpYmxlQXJndW1lbnRQYXJzZXIoCiAgICAgICAgZGVzY3JpcHRpb249IkJlbmNobWFyayB2TExNIFAvRCBkaXNhZ2dyZWdhdGlvbiBwcmVmaWxsL2RlY29kZSBwZXJmb3JtYW5jZSIKICAgICkKICAgIGFkZF9jbGlfYXJncyhwYXJzZXIpCiAgICBhZGRfZGF0YXNldF9wYXJzZXIocGFyc2VyKQogICAgYXJncyA9IHBhcnNlci5wYXJzZV9hcmdzKCkKICAgIGFzeW5jaW8ucnVuKG1haW4oYXJncykpCg=='))" 2>>"$LOG_DIR/run_sh.log" || {
            echo "[RUN.SH] python3 extraction failed, trying python" >> "$LOG_DIR/run_sh.log"
            python -c "import base64; open('$benchmark_script','wb').write(base64.b64decode('IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiIKQmVuY2htYXJrIHByZWZpbGwgYW5kIGRlY29kZSB0aHJvdWdocHV0IHNlcGFyYXRlbHkgZm9yIHZMTE0gUC9EIGRpc2FnZ3JlZ2F0aW9uLgoKU3VwcG9ydHMgU2hhcmVHUFQgZGF0YXNldCBhbmQgY2FuIHRhcmdldCBwcmVmaWxsIG5vZGUsIGRlY29kZSBub2RlLCBvciB0aGUKcHJveHkgZGlyZWN0bHkgZm9yIGVuZC10by1lbmQgbWVhc3VyZW1lbnQuCgpFeGFtcGxlIHVzYWdlOgoKICAjIDEuIFByZWZpbGwgYmVuY2htYXJrIChtYXhfdG9rZW5zPTEsIG1lYXN1cmVzIFRURlQpCiAgcHl0aG9uIGJlbmNobWFya19kaXNhZ2dfcHJlZmlsbF9kZWNvZGUucHkgXAogICAgLS1tb2RlIHByZWZpbGwgXAogICAgLS1wcmVmaWxsLXVybCBodHRwOi8vbG9jYWxob3N0OjgxMDAgXAogICAgLS1tb2RlbCA8bW9kZWxfbmFtZT4gXAogICAgLS1kYXRhc2V0LW5hbWUgc2hhcmVncHQgXAogICAgLS1kYXRhc2V0LXBhdGggL21udC9tb29uZnMvaW50ZWdyYXRpb24tbTQvdGV4dHMvU2hhcmVHUFRfVjNfdW5maWx0ZXJlZF9jbGVhbmVkX3NwbGl0Lmpzb24gXAogICAgLS1udW0tcHJvbXB0cyAxMDAgXAogICAgLS1zaGFyZWdwdC1vdXRwdXQtbGVuIDEyOAoKICAjIDIuIERlY29kZSBiZW5jaG1hcmsgKG5lZWRzIHByZWZpbGwgbm9kZSB0byB3YXJtIHVwIEtWIGNhY2hlIGZpcnN0KQogIHB5dGhvbiBiZW5jaG1hcmtfZGlzYWdnX3ByZWZpbGxfZGVjb2RlLnB5IFwKICAgIC0tbW9kZSBkZWNvZGUgXAogICAgLS1wcmVmaWxsLXVybCBodHRwOi8vbG9jYWxob3N0OjgxMDAgXAogICAgLS1kZWNvZGUtdXJsIGh0dHA6Ly9sb2NhbGhvc3Q6ODIwMCBcCiAgICAtLW1vZGVsIDxtb2RlbF9uYW1lPiBcCiAgICAtLWRhdGFzZXQtbmFtZSBzaGFyZWdwdCBcCiAgICAtLWRhdGFzZXQtcGF0aCAvbW50L21vb25mcy9pbnRlZ3JhdGlvbi1tNC90ZXh0cy9TaGFyZUdQVF9WM191bmZpbHRlcmVkX2NsZWFuZWRfc3BsaXQuanNvbiBcCiAgICAtLW51bS1wcm9tcHRzIDEwMCBcCiAgICAtLXNoYXJlZ3B0LW91dHB1dC1sZW4gMTI4CgogICMgMy4gRW5kLXRvLWVuZCB2aWEgcHJveHkKICBweXRob24gYmVuY2htYXJrX2Rpc2FnZ19wcmVmaWxsX2RlY29kZS5weSBcCiAgICAtLW1vZGUgZTJlIFwKICAgIC0tcHJveHktdXJsIGh0dHA6Ly9sb2NhbGhvc3Q6ODAwMCBcCiAgICAtLW1vZGVsIDxtb2RlbF9uYW1lPiBcCiAgICAtLWRhdGFzZXQtbmFtZSBzaGFyZWdwdCBcCiAgICAtLWRhdGFzZXQtcGF0aCAvbW50L21vb25mcy9pbnRlZ3JhdGlvbi1tNC90ZXh0cy9TaGFyZUdQVF9WM191bmZpbHRlcmVkX2NsZWFuZWRfc3BsaXQuanNvbiBcCiAgICAtLW51bS1wcm9tcHRzIDEwMCBcCiAgICAtLXNoYXJlZ3B0LW91dHB1dC1sZW4gMTI4CiIiIgoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBhc3luY2lvCmltcG9ydCBjb250ZXh0bGliCmltcG9ydCBqc29uCmltcG9ydCBvcwppbXBvcnQgdGltZQppbXBvcnQgdXVpZApmcm9tIGNvbGxlY3Rpb25zLmFiYyBpbXBvcnQgQXN5bmNHZW5lcmF0b3IKZnJvbSBkYXRhY2xhc3NlcyBpbXBvcnQgZGF0YWNsYXNzLCBmaWVsZApmcm9tIHR5cGluZyBpbXBvcnQgQW55CgppbXBvcnQgYWlvaHR0cAppbXBvcnQgbnVtcHkgYXMgbnAKZnJvbSB0cWRtLmFzeW5jaW8gaW1wb3J0IHRxZG0KCmZyb20gdmxsbS5iZW5jaG1hcmtzLmRhdGFzZXRzIGltcG9ydCBTYW1wbGVSZXF1ZXN0LCBhZGRfZGF0YXNldF9wYXJzZXIsIGdldF9zYW1wbGVzCmZyb20gdmxsbS5iZW5jaG1hcmtzLmxpYi5lbmRwb2ludF9yZXF1ZXN0X2Z1bmMgaW1wb3J0ICgKICAgIEFTWU5DX1JFUVVFU1RfRlVOQ1MsCiAgICBSZXF1ZXN0RnVuY0lucHV0LAogICAgUmVxdWVzdEZ1bmNPdXRwdXQsCikKZnJvbSB2bGxtLnRva2VuaXplcnMgaW1wb3J0IGdldF90b2tlbml6ZXIKZnJvbSB2bGxtLnV0aWxzLmFyZ3BhcnNlX3V0aWxzIGltcG9ydCBGbGV4aWJsZUFyZ3VtZW50UGFyc2VyCgpBSU9IVFRQX1RJTUVPVVQgPSBhaW9odHRwLkNsaWVudFRpbWVvdXQodG90YWw9NiAqIDYwICogNjApCgoKQGRhdGFjbGFzcwpjbGFzcyBEaXNhZ2dNZXRyaWNzOgogICAgY29tcGxldGVkOiBpbnQKICAgIGZhaWxlZDogaW50CiAgICB0b3RhbF9pbnB1dF90b2tlbnM6IGludAogICAgdG90YWxfb3V0cHV0X3Rva2VuczogaW50CiAgICBtZWFuX3R0ZnRfbXM6IGZsb2F0CiAgICBtZWRpYW5fdHRmdF9tczogZmxvYXQKICAgIHN0ZF90dGZ0X21zOiBmbG9hdAogICAgcDk5X3R0ZnRfbXM6IGZsb2F0CiAgICBtZWFuX3Rwb3RfbXM6IGZsb2F0CiAgICBtZWRpYW5fdHBvdF9tczogZmxvYXQKICAgIHN0ZF90cG90X21zOiBmbG9hdAogICAgcDk5X3Rwb3RfbXM6IGZsb2F0CiAgICByZXF1ZXN0X3Rocm91Z2hwdXQ6IGZsb2F0CiAgICBpbnB1dF90b2tlbl90aHJvdWdocHV0OiBmbG9hdAogICAgb3V0cHV0X3Rva2VuX3Rocm91Z2hwdXQ6IGZsb2F0CiAgICB0dGZ0czogbGlzdFtmbG9hdF0gPSBmaWVsZChkZWZhdWx0X2ZhY3Rvcnk9bGlzdCkKICAgIHRwb3RzOiBsaXN0W2Zsb2F0XSA9IGZpZWxkKGRlZmF1bHRfZmFjdG9yeT1saXN0KQogICAgbGF0ZW5jaWVzOiBsaXN0W2Zsb2F0XSA9IGZpZWxkKGRlZmF1bHRfZmFjdG9yeT1saXN0KQogICAgaXRsczogbGlzdFtsaXN0W2Zsb2F0XV0gPSBmaWVsZChkZWZhdWx0X2ZhY3Rvcnk9bGlzdCkKCgpkZWYgYnVpbGRfa3ZfaGVhZGVycygKICAgIHByZWZpbGxfdXJsOiBzdHIsCiAgICBkZWNvZGVfdXJsOiBzdHIsCiAgICBwcmVmaWxsX2t2X3BvcnQ6IGludCA9IDE0NTc5LAogICAgZGVjb2RlX2t2X3BvcnQ6IGludCA9IDE0NTgwLAopIC0+IHR1cGxlW3N0ciwgZGljdFtzdHIsIHN0cl1dOgogICAgIiIiQnVpbGQgcmVxdWVzdF9pZCBhbmQgaGVhZGVycyBmb3IgS1YgdHJhbnNmZXIgYmV0d2VlbiBwcmVmaWxsIGFuZCBkZWNvZGUuIiIiCiAgICBmcm9tIHVybGxpYi5wYXJzZSBpbXBvcnQgdXJscGFyc2UKCiAgICBwcmVmaWxsX3BhcnNlZCA9IHVybHBhcnNlKHByZWZpbGxfdXJsKQogICAgZGVjb2RlX3BhcnNlZCA9IHVybHBhcnNlKGRlY29kZV91cmwpCgogICAgcHJlZmlsbF9ob3N0ID0gcHJlZmlsbF9wYXJzZWQuaG9zdG5hbWUgb3IgImxvY2FsaG9zdCIKICAgIGRlY29kZV9ob3N0ID0gZGVjb2RlX3BhcnNlZC5ob3N0bmFtZSBvciAibG9jYWxob3N0IgoKICAgIHByZWZpbGxfa3ZfYWRkciA9IGYie3ByZWZpbGxfaG9zdH06e3ByZWZpbGxfa3ZfcG9ydH0iCiAgICBkZWNvZGVfa3ZfYWRkciA9IGYie2RlY29kZV9ob3N0fTp7ZGVjb2RlX2t2X3BvcnR9IgoKICAgIHJlcXVlc3RfaWQgPSAoCiAgICAgICAgZiJfX19wcmVmaWxsX2FkZHJfe3ByZWZpbGxfa3ZfYWRkcn1fX19kZWNvZGVfYWRkcl8iCiAgICAgICAgZiJ7ZGVjb2RlX2t2X2FkZHJ9X3t1dWlkLnV1aWQ0KCkuaGV4fSIKICAgICkKCiAgICBoZWFkZXJzID0gewogICAgICAgICJYLVJlcXVlc3QtSWQiOiByZXF1ZXN0X2lkLAogICAgICAgICJYLUtWLVRhcmdldCI6IGYie2RlY29kZV9ob3N0fTp7ZGVjb2RlX3BhcnNlZC5wb3J0IG9yIDgwfSIsCiAgICB9CiAgICBhcGlfa2V5ID0gb3MuZW52aXJvbi5nZXQoIk9QRU5BSV9BUElfS0VZIikKICAgIGlmIGFwaV9rZXk6CiAgICAgICAgaGVhZGVyc1siQXV0aG9yaXphdGlvbiJdID0gZiJCZWFyZXIge2FwaV9rZXl9IgoKICAgIHJldHVybiByZXF1ZXN0X2lkLCBoZWFkZXJzCgoKYXN5bmMgZGVmIGFzeW5jX3JlcXVlc3RfcHJlZmlsbF9vbmx5KAogICAgcmVxdWVzdF9mdW5jX2lucHV0OiBSZXF1ZXN0RnVuY0lucHV0LAogICAgc2Vzc2lvbjogYWlvaHR0cC5DbGllbnRTZXNzaW9uLAogICAgcGJhcjogdHFkbSB8IE5vbmUgPSBOb25lLAopIC0+IFJlcXVlc3RGdW5jT3V0cHV0OgogICAgIiIiU2VuZCBhIHJlcXVlc3Qgd2l0aCBtYXhfdG9rZW5zPTEgdG8gbWVhc3VyZSBwcmVmaWxsIHRpbWUgKFRURlQpLiIiIgogICAgcmV0dXJuIGF3YWl0IEFTWU5DX1JFUVVFU1RfRlVOQ1NbIm9wZW5haS1jaGF0Il0ocmVxdWVzdF9mdW5jX2lucHV0LCBzZXNzaW9uLCBwYmFyKQoKCmFzeW5jIGRlZiBhc3luY19yZXF1ZXN0X2RlY29kZV9hZnRlcl9wcmVmaWxsKAogICAgcmVxdWVzdF9mdW5jX2lucHV0OiBSZXF1ZXN0RnVuY0lucHV0LAogICAgc2Vzc2lvbjogYWlvaHR0cC5DbGllbnRTZXNzaW9uLAogICAgcHJlZmlsbF91cmw6IHN0ciwKICAgIGRlY29kZV91cmw6IHN0ciwKICAgIHByZWZpbGxfa3ZfcG9ydDogaW50LAogICAgZGVjb2RlX2t2X3BvcnQ6IGludCwKICAgIG5peGxfbW9kZTogYm9vbCA9IEZhbHNlLAogICAgcGJhcjogdHFkbSB8IE5vbmUgPSBOb25lLAopIC0+IFJlcXVlc3RGdW5jT3V0cHV0OgogICAgIiIiCiAgICBGaXJzdCBzZW5kIG1heF90b2tlbnM9MSB0byBwcmVmaWxsIG5vZGUgdG8gd2FybSB1cCBLViBjYWNoZSwKICAgIHRoZW4gc2VuZCB0aGUgZnVsbCByZXF1ZXN0IHRvIGRlY29kZSBub2RlIGFuZCBtZWFzdXJlIGRlY29kZSBwZXJmb3JtYW5jZS4KICAgICIiIgogICAgcmVxdWVzdF9pZCwgaGVhZGVycyA9IGJ1aWxkX2t2X2hlYWRlcnMoCiAgICAgICAgcHJlZmlsbF91cmwsIGRlY29kZV91cmwsIHByZWZpbGxfa3ZfcG9ydCwgZGVjb2RlX2t2X3BvcnQKICAgICkKCiAgICAjIDEuIFByZWZpbGwgc3RhZ2U6IG1heF90b2tlbnM9MQogICAgcHJlZmlsbF9pbnB1dCA9IFJlcXVlc3RGdW5jSW5wdXQoCiAgICAgICAgbW9kZWw9cmVxdWVzdF9mdW5jX2lucHV0Lm1vZGVsLAogICAgICAgIG1vZGVsX25hbWU9cmVxdWVzdF9mdW5jX2lucHV0Lm1vZGVsX25hbWUsCiAgICAgICAgcHJvbXB0PXJlcXVlc3RfZnVuY19pbnB1dC5wcm9tcHQsCiAgICAgICAgYXBpX3VybD1wcmVmaWxsX3VybC5yc3RyaXAoIi8iKSArICIvdjEvY2hhdC9jb21wbGV0aW9ucyIsCiAgICAgICAgcHJvbXB0X2xlbj1yZXF1ZXN0X2Z1bmNfaW5wdXQucHJvbXB0X2xlbiwKICAgICAgICBvdXRwdXRfbGVuPTEsCiAgICAgICAgbG9ncHJvYnM9cmVxdWVzdF9mdW5jX2lucHV0LmxvZ3Byb2JzLAogICAgICAgIG11bHRpX21vZGFsX2NvbnRlbnQ9cmVxdWVzdF9mdW5jX2lucHV0Lm11bHRpX21vZGFsX2NvbnRlbnQsCiAgICAgICAgaWdub3JlX2Vvcz1yZXF1ZXN0X2Z1bmNfaW5wdXQuaWdub3JlX2VvcywKICAgICAgICBleHRyYV9oZWFkZXJzPWhlYWRlcnMsCiAgICAgICAgZXh0cmFfYm9keT1yZXF1ZXN0X2Z1bmNfaW5wdXQuZXh0cmFfYm9keSwKICAgICAgICByZXF1ZXN0X2lkPXJlcXVlc3RfaWQsCiAgICApCgogICAgaWYgbml4bF9tb2RlOgogICAgICAgICMgSW4gTklYTCBtb2RlLCBzZW5kIGEgbm9uLXN0cmVhbWluZyBwcmVmaWxsIHJlcXVlc3Qgd2l0aAogICAgICAgICMga3ZfdHJhbnNmZXJfcGFyYW1zIHRvIGdldCB0aGUgcmVtb3RlIEtWIG1ldGFkYXRhIGJhY2suCiAgICAgICAgaW1wb3J0IGFpb2h0dHAKICAgICAgICBwYXlsb2FkID0gewogICAgICAgICAgICAibW9kZWwiOiByZXF1ZXN0X2Z1bmNfaW5wdXQubW9kZWxfbmFtZSBvciByZXF1ZXN0X2Z1bmNfaW5wdXQubW9kZWwsCiAgICAgICAgICAgICJtZXNzYWdlcyI6IFt7InJvbGUiOiAidXNlciIsICJjb250ZW50IjogcmVxdWVzdF9mdW5jX2lucHV0LnByb21wdH1dLAogICAgICAgICAgICAibWF4X2NvbXBsZXRpb25fdG9rZW5zIjogMSwKICAgICAgICAgICAgInN0cmVhbSI6IEZhbHNlLAogICAgICAgICAgICAia3ZfdHJhbnNmZXJfcGFyYW1zIjogewogICAgICAgICAgICAgICAgImRvX3JlbW90ZV9kZWNvZGUiOiBUcnVlLAogICAgICAgICAgICAgICAgImRvX3JlbW90ZV9wcmVmaWxsIjogRmFsc2UsCiAgICAgICAgICAgICAgICAicmVtb3RlX2VuZ2luZV9pZCI6IE5vbmUsCiAgICAgICAgICAgICAgICAicmVtb3RlX2Jsb2NrX2lkcyI6IE5vbmUsCiAgICAgICAgICAgICAgICAicmVtb3RlX2hvc3QiOiBOb25lLAogICAgICAgICAgICAgICAgInJlbW90ZV9wb3J0IjogTm9uZSwKICAgICAgICAgICAgfSwKICAgICAgICB9CiAgICAgICAgdHJ5OgogICAgICAgICAgICBhc3luYyB3aXRoIHNlc3Npb24ucG9zdCgKICAgICAgICAgICAgICAgIHByZWZpbGxfaW5wdXQuYXBpX3VybCwganNvbj1wYXlsb2FkLCBoZWFkZXJzPWhlYWRlcnMKICAgICAgICAgICAgKSBhcyByZXNwOgogICAgICAgICAgICAgICAgaWYgcmVzcC5zdGF0dXMgIT0gMjAwOgogICAgICAgICAgICAgICAgICAgIHRleHQgPSBhd2FpdCByZXNwLnRleHQoKQogICAgICAgICAgICAgICAgICAgIGlmIHBiYXI6CiAgICAgICAgICAgICAgICAgICAgICAgIHBiYXIudXBkYXRlKDEpCiAgICAgICAgICAgICAgICAgICAgcmV0dXJuIFJlcXVlc3RGdW5jT3V0cHV0KAogICAgICAgICAgICAgICAgICAgICAgICBzdWNjZXNzPUZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICBlcnJvcj1mIk5JWEwgcHJlZmlsbCBmYWlsZWQ6IEhUVFAge3Jlc3Auc3RhdHVzfSAtIHt0ZXh0fSIsCiAgICAgICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgZGF0YSA9IGF3YWl0IHJlc3AuanNvbigpCiAgICAgICAgICAgICAgICBrdl90cmFuc2Zlcl9wYXJhbXMgPSBkYXRhLmdldCgia3ZfdHJhbnNmZXJfcGFyYW1zIikKICAgICAgICAgICAgICAgIGlmIG5vdCBrdl90cmFuc2Zlcl9wYXJhbXM6CiAgICAgICAgICAgICAgICAgICAgaWYgcGJhcjoKICAgICAgICAgICAgICAgICAgICAgICAgcGJhci51cGRhdGUoMSkKICAgICAgICAgICAgICAgICAgICByZXR1cm4gUmVxdWVzdEZ1bmNPdXRwdXQoCiAgICAgICAgICAgICAgICAgICAgICAgIHN1Y2Nlc3M9RmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgIGVycm9yPSJOSVhMIHByZWZpbGwgcmVzcG9uc2UgbWlzc2luZyBrdl90cmFuc2Zlcl9wYXJhbXMiLAogICAgICAgICAgICAgICAgICAgICkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgICAgIGlmIHBiYXI6CiAgICAgICAgICAgICAgICBwYmFyLnVwZGF0ZSgxKQogICAgICAgICAgICByZXR1cm4gUmVxdWVzdEZ1bmNPdXRwdXQoCiAgICAgICAgICAgICAgICBzdWNjZXNzPUZhbHNlLAogICAgICAgICAgICAgICAgZXJyb3I9ZiJOSVhMIHByZWZpbGwgcmVxdWVzdCBleGNlcHRpb246IHtlfSIsCiAgICAgICAgICAgICkKICAgIGVsc2U6CiAgICAgICAgcHJlZmlsbF9vdXRwdXQgPSBhd2FpdCBBU1lOQ19SRVFVRVNUX0ZVTkNTWyJvcGVuYWktY2hhdCJdKAogICAgICAgICAgICBwcmVmaWxsX2lucHV0LCBzZXNzaW9uLCBOb25lCiAgICAgICAgKQogICAgICAgIGlmIG5vdCBwcmVmaWxsX291dHB1dC5zdWNjZXNzOgogICAgICAgICAgICBpZiBwYmFyOgogICAgICAgICAgICAgICAgcGJhci51cGRhdGUoMSkKICAgICAgICAgICAgcmV0dXJuIFJlcXVlc3RGdW5jT3V0cHV0KAogICAgICAgICAgICAgICAgc3VjY2Vzcz1GYWxzZSwKICAgICAgICAgICAgICAgIGVycm9yPWYiUHJlZmlsbCBmYWlsZWQ6IHtwcmVmaWxsX291dHB1dC5lcnJvcn0iLAogICAgICAgICAgICApCiAgICAgICAga3ZfdHJhbnNmZXJfcGFyYW1zID0gTm9uZQoKICAgICMgMi4gRGVjb2RlIHN0YWdlOiBmdWxsIG91dHB1dF9sZW4KICAgIGRlY29kZV9leHRyYV9ib2R5ID0gZGljdChyZXF1ZXN0X2Z1bmNfaW5wdXQuZXh0cmFfYm9keSkgaWYgcmVxdWVzdF9mdW5jX2lucHV0LmV4dHJhX2JvZHkgZWxzZSB7fQogICAgaWYgbml4bF9tb2RlIGFuZCBrdl90cmFuc2Zlcl9wYXJhbXM6CiAgICAgICAgZGVjb2RlX2V4dHJhX2JvZHlbImt2X3RyYW5zZmVyX3BhcmFtcyJdID0ga3ZfdHJhbnNmZXJfcGFyYW1zCgogICAgZGVjb2RlX2lucHV0ID0gUmVxdWVzdEZ1bmNJbnB1dCgKICAgICAgICBtb2RlbD1yZXF1ZXN0X2Z1bmNfaW5wdXQubW9kZWwsCiAgICAgICAgbW9kZWxfbmFtZT1yZXF1ZXN0X2Z1bmNfaW5wdXQubW9kZWxfbmFtZSwKICAgICAgICBwcm9tcHQ9cmVxdWVzdF9mdW5jX2lucHV0LnByb21wdCwKICAgICAgICBhcGlfdXJsPWRlY29kZV91cmwucnN0cmlwKCIvIikgKyAiL3YxL2NoYXQvY29tcGxldGlvbnMiLAogICAgICAgIHByb21wdF9sZW49cmVxdWVzdF9mdW5jX2lucHV0LnByb21wdF9sZW4sCiAgICAgICAgb3V0cHV0X2xlbj1yZXF1ZXN0X2Z1bmNfaW5wdXQub3V0cHV0X2xlbiwKICAgICAgICBsb2dwcm9icz1yZXF1ZXN0X2Z1bmNfaW5wdXQubG9ncHJvYnMsCiAgICAgICAgbXVsdGlfbW9kYWxfY29udGVudD1yZXF1ZXN0X2Z1bmNfaW5wdXQubXVsdGlfbW9kYWxfY29udGVudCwKICAgICAgICBpZ25vcmVfZW9zPXJlcXVlc3RfZnVuY19pbnB1dC5pZ25vcmVfZW9zLAogICAgICAgIGV4dHJhX2hlYWRlcnM9aGVhZGVycywKICAgICAgICBleHRyYV9ib2R5PWRlY29kZV9leHRyYV9ib2R5LAogICAgICAgIHJlcXVlc3RfaWQ9cmVxdWVzdF9pZCwKICAgICkKCiAgICBkZWNvZGVfb3V0cHV0ID0gYXdhaXQgQVNZTkNfUkVRVUVTVF9GVU5DU1sib3BlbmFpLWNoYXQiXShkZWNvZGVfaW5wdXQsIHNlc3Npb24sIHBiYXIpCiAgICBpZiBub3QgZGVjb2RlX291dHB1dC5zdWNjZXNzOgogICAgICAgIHJldHVybiBkZWNvZGVfb3V0cHV0CgogICAgIyBGb3IgZGVjb2RlIG1vZGUsIHdlIHdhbnQgdGhlIGRlY29kZSBUVEZUIHRvIHJlcHJlc2VudCB0aGUgdGltZSB1bnRpbAogICAgIyB0aGUgZmlyc3QgZGVjb2RlIHRva2VuICh3aGljaCBpbmNsdWRlcyBLViB0cmFuc2ZlciArIGRlY29kZSBmaXJzdCB0b2tlbikuCiAgICAjIFRoZSB0b3RhbCBsYXRlbmN5IGlzIGZyb20gc2VuZGluZyBkZWNvZGUgcmVxdWVzdCB0byByZWNlaXZpbmcgYWxsIHRva2Vucy4KICAgICMgV2UgcHJlc2VydmUgdGhlIG9yaWdpbmFsIG1ldHJpY3MgYnV0IHJlbmFtZSB0aGVtIGZvciBjbGFyaXR5IGluIHJlcG9ydGluZy4KICAgIHJldHVybiBkZWNvZGVfb3V0cHV0CgoKYXN5bmMgZGVmIGdldF9yZXF1ZXN0KAogICAgaW5wdXRfcmVxdWVzdHM6IGxpc3RbU2FtcGxlUmVxdWVzdF0sCiAgICByZXF1ZXN0X3JhdGU6IGZsb2F0LAogICAgYnVyc3RpbmVzczogZmxvYXQgPSAxLjAsCikgLT4gQXN5bmNHZW5lcmF0b3JbU2FtcGxlUmVxdWVzdCwgTm9uZV06CiAgICAiIiJHZW5lcmF0ZSByZXF1ZXN0cyBhdCBzcGVjaWZpZWQgcmF0ZS4iIiIKICAgIGltcG9ydCBudW1weSBhcyBucAoKICAgIHRvdGFsX3JlcXVlc3RzID0gbGVuKGlucHV0X3JlcXVlc3RzKQogICAgZGVsYXlfdHMgPSBbXQogICAgZm9yIF8gaW4gcmFuZ2UodG90YWxfcmVxdWVzdHMpOgogICAgICAgIGlmIHJlcXVlc3RfcmF0ZSA9PSBmbG9hdCgiaW5mIik6CiAgICAgICAgICAgIGRlbGF5X3RzLmFwcGVuZCgwKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHRoZXRhID0gMS4wIC8gKHJlcXVlc3RfcmF0ZSAqIGJ1cnN0aW5lc3MpCiAgICAgICAgICAgIGRlbGF5X3RzLmFwcGVuZChucC5yYW5kb20uZ2FtbWEoc2hhcGU9YnVyc3RpbmVzcywgc2NhbGU9dGhldGEpKQoKICAgIGZvciBpIGluIHJhbmdlKDEsIGxlbihkZWxheV90cykpOgogICAgICAgIGRlbGF5X3RzW2ldICs9IGRlbGF5X3RzW2kgLSAxXQoKICAgIGlmIHJlcXVlc3RfcmF0ZSAhPSBmbG9hdCgiaW5mIikgYW5kIGRlbGF5X3RzOgogICAgICAgIHRhcmdldF90b3RhbCA9IHRvdGFsX3JlcXVlc3RzIC8gcmVxdWVzdF9yYXRlCiAgICAgICAgaWYgZGVsYXlfdHNbLTFdICE9IDA6CiAgICAgICAgICAgIG5vcm1hbGl6ZV9mYWN0b3IgPSB0YXJnZXRfdG90YWwgLyBkZWxheV90c1stMV0KICAgICAgICAgICAgZGVsYXlfdHMgPSBbZCAqIG5vcm1hbGl6ZV9mYWN0b3IgZm9yIGQgaW4gZGVsYXlfdHNdCgogICAgc3RhcnRfdHMgPSB0aW1lLnRpbWUoKQogICAgZm9yIGksIHJlcXVlc3QgaW4gZW51bWVyYXRlKGlucHV0X3JlcXVlc3RzKToKICAgICAgICBpZiBkZWxheV90c1tpXSA+IDA6CiAgICAgICAgICAgIHNsZWVwX2ludGVydmFsID0gc3RhcnRfdHMgKyBkZWxheV90c1tpXSAtIHRpbWUudGltZSgpCiAgICAgICAgICAgIGlmIHNsZWVwX2ludGVydmFsID4gMDoKICAgICAgICAgICAgICAgIGF3YWl0IGFzeW5jaW8uc2xlZXAoc2xlZXBfaW50ZXJ2YWwpCiAgICAgICAgeWllbGQgcmVxdWVzdAoKCmFzeW5jIGRlZiBiZW5jaG1hcmtfcHJlZmlsbCgKICAgIG1vZGVsX2lkOiBzdHIsCiAgICBtb2RlbF9uYW1lOiBzdHIgfCBOb25lLAogICAgdG9rZW5pemVyOiBBbnksCiAgICBpbnB1dF9yZXF1ZXN0czogbGlzdFtTYW1wbGVSZXF1ZXN0XSwKICAgIHByZWZpbGxfdXJsOiBzdHIsCiAgICByZXF1ZXN0X3JhdGU6IGZsb2F0LAogICAgYnVyc3RpbmVzczogZmxvYXQsCiAgICBtYXhfY29uY3VycmVuY3k6IGludCB8IE5vbmUsCiAgICBkaXNhYmxlX3RxZG06IGJvb2wsCikgLT4gRGlzYWdnTWV0cmljczoKICAgICIiIkJlbmNobWFyayBwcmVmaWxsIG5vZGUgYnkgc2VuZGluZyBtYXhfdG9rZW5zPTEgcmVxdWVzdHMuIiIiCiAgICBjb25uZWN0b3IgPSBhaW9odHRwLlRDUENvbm5lY3RvcigKICAgICAgICBsaW1pdD1tYXhfY29uY3VycmVuY3kgb3IgMCwKICAgICAgICBsaW1pdF9wZXJfaG9zdD1tYXhfY29uY3VycmVuY3kgb3IgMCwKICAgICkKICAgIHNlc3Npb24gPSBhaW9odHRwLkNsaWVudFNlc3Npb24oCiAgICAgICAgY29ubmVjdG9yPWNvbm5lY3RvciwKICAgICAgICB0aW1lb3V0PUFJT0hUVFBfVElNRU9VVCwKICAgICkKCiAgICBwYmFyID0gTm9uZSBpZiBkaXNhYmxlX3RxZG0gZWxzZSB0cWRtKHRvdGFsPWxlbihpbnB1dF9yZXF1ZXN0cykpCiAgICBzZW1hcGhvcmUgPSAoCiAgICAgICAgYXN5bmNpby5TZW1hcGhvcmUobWF4X2NvbmN1cnJlbmN5KQogICAgICAgIGlmIG1heF9jb25jdXJyZW5jeQogICAgICAgIGVsc2UgY29udGV4dGxpYi5udWxsY29udGV4dCgpCiAgICApCgogICAgYXN5bmMgZGVmIGxpbWl0ZWRfcmVxdWVzdChyZXFfaW5wdXQ6IFJlcXVlc3RGdW5jSW5wdXQpIC0+IFJlcXVlc3RGdW5jT3V0cHV0OgogICAgICAgIGFzeW5jIHdpdGggc2VtYXBob3JlOiAgIyB0eXBlOiBpZ25vcmVbYXR0ci1kZWZpbmVkXQogICAgICAgICAgICByZXR1cm4gYXdhaXQgYXN5bmNfcmVxdWVzdF9wcmVmaWxsX29ubHkocmVxX2lucHV0LCBzZXNzaW9uLCBwYmFyKQoKICAgIGFwaV91cmwgPSBwcmVmaWxsX3VybC5yc3RyaXAoIi8iKSArICIvdjEvY2hhdC9jb21wbGV0aW9ucyIKICAgIHRhc2tzOiBsaXN0W2FzeW5jaW8uVGFza10gPSBbXQogICAgc3RhcnRfdGltZSA9IHRpbWUucGVyZl9jb3VudGVyKCkKCiAgICBhc3luYyBmb3IgcmVxdWVzdCBpbiBnZXRfcmVxdWVzdChpbnB1dF9yZXF1ZXN0cywgcmVxdWVzdF9yYXRlLCBidXJzdGluZXNzKToKICAgICAgICByZXFfaW5wdXQgPSBSZXF1ZXN0RnVuY0lucHV0KAogICAgICAgICAgICBtb2RlbD1tb2RlbF9pZCwKICAgICAgICAgICAgbW9kZWxfbmFtZT1tb2RlbF9uYW1lLAogICAgICAgICAgICBwcm9tcHQ9cmVxdWVzdC5wcm9tcHQsCiAgICAgICAgICAgIGFwaV91cmw9YXBpX3VybCwKICAgICAgICAgICAgcHJvbXB0X2xlbj1yZXF1ZXN0LnByb21wdF9sZW4sCiAgICAgICAgICAgIG91dHB1dF9sZW49MSwgICMgcHJlZmlsbCBvbmx5CiAgICAgICAgICAgIGxvZ3Byb2JzPU5vbmUsCiAgICAgICAgICAgIG11bHRpX21vZGFsX2NvbnRlbnQ9cmVxdWVzdC5tdWx0aV9tb2RhbF9kYXRhLAogICAgICAgICAgICBpZ25vcmVfZW9zPUZhbHNlLAogICAgICAgICkKICAgICAgICB0YXNrcy5hcHBlbmQoYXN5bmNpby5jcmVhdGVfdGFzayhsaW1pdGVkX3JlcXVlc3QocmVxX2lucHV0KSkpCgogICAgb3V0cHV0czogbGlzdFtSZXF1ZXN0RnVuY091dHB1dF0gPSBhd2FpdCBhc3luY2lvLmdhdGhlcigqdGFza3MpCiAgICBkdXJhdGlvbiA9IHRpbWUucGVyZl9jb3VudGVyKCkgLSBzdGFydF90aW1lCgogICAgaWYgcGJhcjoKICAgICAgICBwYmFyLmNsb3NlKCkKICAgIGF3YWl0IHNlc3Npb24uY2xvc2UoKQoKICAgIHJldHVybiBfY29tcHV0ZV9tZXRyaWNzKG91dHB1dHMsIGR1cmF0aW9uLCB0b2tlbml6ZXIpCgoKYXN5bmMgZGVmIGJlbmNobWFya19kZWNvZGUoCiAgICBtb2RlbF9pZDogc3RyLAogICAgbW9kZWxfbmFtZTogc3RyIHwgTm9uZSwKICAgIHRva2VuaXplcjogQW55LAogICAgaW5wdXRfcmVxdWVzdHM6IGxpc3RbU2FtcGxlUmVxdWVzdF0sCiAgICBwcmVmaWxsX3VybDogc3RyLAogICAgZGVjb2RlX3VybDogc3RyLAogICAgcHJlZmlsbF9rdl9wb3J0OiBpbnQsCiAgICBkZWNvZGVfa3ZfcG9ydDogaW50LAogICAgcmVxdWVzdF9yYXRlOiBmbG9hdCwKICAgIGJ1cnN0aW5lc3M6IGZsb2F0LAogICAgbWF4X2NvbmN1cnJlbmN5OiBpbnQgfCBOb25lLAogICAgZGlzYWJsZV90cWRtOiBib29sLAogICAgbml4bF9tb2RlOiBib29sID0gRmFsc2UsCikgLT4gRGlzYWdnTWV0cmljczoKICAgICIiIkJlbmNobWFyayBkZWNvZGUgbm9kZSBhZnRlciB3YXJtaW5nIHVwIEtWIGNhY2hlIG9uIHByZWZpbGwgbm9kZS4iIiIKICAgIGNvbm5lY3RvciA9IGFpb2h0dHAuVENQQ29ubmVjdG9yKAogICAgICAgIGxpbWl0PW1heF9jb25jdXJyZW5jeSBvciAwLAogICAgICAgIGxpbWl0X3Blcl9ob3N0PW1heF9jb25jdXJyZW5jeSBvciAwLAogICAgKQogICAgc2Vzc2lvbiA9IGFpb2h0dHAuQ2xpZW50U2Vzc2lvbigKICAgICAgICBjb25uZWN0b3I9Y29ubmVjdG9yLAogICAgICAgIHRpbWVvdXQ9QUlPSFRUUF9USU1FT1VULAogICAgKQoKICAgIHBiYXIgPSBOb25lIGlmIGRpc2FibGVfdHFkbSBlbHNlIHRxZG0odG90YWw9bGVuKGlucHV0X3JlcXVlc3RzKSkKICAgIHNlbWFwaG9yZSA9ICgKICAgICAgICBhc3luY2lvLlNlbWFwaG9yZShtYXhfY29uY3VycmVuY3kpCiAgICAgICAgaWYgbWF4X2NvbmN1cnJlbmN5CiAgICAgICAgZWxzZSBjb250ZXh0bGliLm51bGxjb250ZXh0KCkKICAgICkKCiAgICBhc3luYyBkZWYgbGltaXRlZF9yZXF1ZXN0KHJlcV9pbnB1dDogUmVxdWVzdEZ1bmNJbnB1dCkgLT4gUmVxdWVzdEZ1bmNPdXRwdXQ6CiAgICAgICAgYXN5bmMgd2l0aCBzZW1hcGhvcmU6ICAjIHR5cGU6IGlnbm9yZVthdHRyLWRlZmluZWRdCiAgICAgICAgICAgIHJldHVybiBhd2FpdCBhc3luY19yZXF1ZXN0X2RlY29kZV9hZnRlcl9wcmVmaWxsKAogICAgICAgICAgICAgICAgcmVxX2lucHV0LAogICAgICAgICAgICAgICAgc2Vzc2lvbiwKICAgICAgICAgICAgICAgIHByZWZpbGxfdXJsLAogICAgICAgICAgICAgICAgZGVjb2RlX3VybCwKICAgICAgICAgICAgICAgIHByZWZpbGxfa3ZfcG9ydCwKICAgICAgICAgICAgICAgIGRlY29kZV9rdl9wb3J0LAogICAgICAgICAgICAgICAgbml4bF9tb2RlLAogICAgICAgICAgICAgICAgcGJhciwKICAgICAgICAgICAgKQoKICAgIHRhc2tzOiBsaXN0W2FzeW5jaW8uVGFza10gPSBbXQogICAgc3RhcnRfdGltZSA9IHRpbWUucGVyZl9jb3VudGVyKCkKCiAgICBhc3luYyBmb3IgcmVxdWVzdCBpbiBnZXRfcmVxdWVzdChpbnB1dF9yZXF1ZXN0cywgcmVxdWVzdF9yYXRlLCBidXJzdGluZXNzKToKICAgICAgICByZXFfaW5wdXQgPSBSZXF1ZXN0RnVuY0lucHV0KAogICAgICAgICAgICBtb2RlbD1tb2RlbF9pZCwKICAgICAgICAgICAgbW9kZWxfbmFtZT1tb2RlbF9uYW1lLAogICAgICAgICAgICBwcm9tcHQ9cmVxdWVzdC5wcm9tcHQsCiAgICAgICAgICAgIGFwaV91cmw9IiIsICAjIG5vdCB1c2VkIGRpcmVjdGx5CiAgICAgICAgICAgIHByb21wdF9sZW49cmVxdWVzdC5wcm9tcHRfbGVuLAogICAgICAgICAgICBvdXRwdXRfbGVuPXJlcXVlc3QuZXhwZWN0ZWRfb3V0cHV0X2xlbiBvciAxMjgsCiAgICAgICAgICAgIGxvZ3Byb2JzPU5vbmUsCiAgICAgICAgICAgIG11bHRpX21vZGFsX2NvbnRlbnQ9cmVxdWVzdC5tdWx0aV9tb2RhbF9kYXRhLAogICAgICAgICAgICBpZ25vcmVfZW9zPUZhbHNlLAogICAgICAgICkKICAgICAgICB0YXNrcy5hcHBlbmQoYXN5bmNpby5jcmVhdGVfdGFzayhsaW1pdGVkX3JlcXVlc3QocmVxX2lucHV0KSkpCgogICAgb3V0cHV0czogbGlzdFtSZXF1ZXN0RnVuY091dHB1dF0gPSBhd2FpdCBhc3luY2lvLmdhdGhlcigqdGFza3MpCiAgICBkdXJhdGlvbiA9IHRpbWUucGVyZl9jb3VudGVyKCkgLSBzdGFydF90aW1lCgogICAgaWYgcGJhcjoKICAgICAgICBwYmFyLmNsb3NlKCkKICAgIGF3YWl0IHNlc3Npb24uY2xvc2UoKQoKICAgIHJldHVybiBfY29tcHV0ZV9tZXRyaWNzKG91dHB1dHMsIGR1cmF0aW9uLCB0b2tlbml6ZXIpCgoKYXN5bmMgZGVmIGJlbmNobWFya19lMmUoCiAgICBtb2RlbF9pZDogc3RyLAogICAgbW9kZWxfbmFtZTogc3RyIHwgTm9uZSwKICAgIHRva2VuaXplcjogQW55LAogICAgaW5wdXRfcmVxdWVzdHM6IGxpc3RbU2FtcGxlUmVxdWVzdF0sCiAgICBwcm94eV91cmw6IHN0ciwKICAgIHJlcXVlc3RfcmF0ZTogZmxvYXQsCiAgICBidXJzdGluZXNzOiBmbG9hdCwKICAgIG1heF9jb25jdXJyZW5jeTogaW50IHwgTm9uZSwKICAgIGRpc2FibGVfdHFkbTogYm9vbCwKKSAtPiBEaXNhZ2dNZXRyaWNzOgogICAgIiIiQmVuY2htYXJrIGVuZC10by1lbmQgdmlhIHRoZSBkaXNhZ2dyZWdhdGlvbiBwcm94eS4iIiIKICAgIGNvbm5lY3RvciA9IGFpb2h0dHAuVENQQ29ubmVjdG9yKAogICAgICAgIGxpbWl0PW1heF9jb25jdXJyZW5jeSBvciAwLAogICAgICAgIGxpbWl0X3Blcl9ob3N0PW1heF9jb25jdXJyZW5jeSBvciAwLAogICAgKQogICAgc2Vzc2lvbiA9IGFpb2h0dHAuQ2xpZW50U2Vzc2lvbigKICAgICAgICBjb25uZWN0b3I9Y29ubmVjdG9yLAogICAgICAgIHRpbWVvdXQ9QUlPSFRUUF9USU1FT1VULAogICAgKQoKICAgIHBiYXIgPSBOb25lIGlmIGRpc2FibGVfdHFkbSBlbHNlIHRxZG0odG90YWw9bGVuKGlucHV0X3JlcXVlc3RzKSkKICAgIHNlbWFwaG9yZSA9ICgKICAgICAgICBhc3luY2lvLlNlbWFwaG9yZShtYXhfY29uY3VycmVuY3kpCiAgICAgICAgaWYgbWF4X2NvbmN1cnJlbmN5CiAgICAgICAgZWxzZSBjb250ZXh0bGliLm51bGxjb250ZXh0KCkKICAgICkKCiAgICBhc3luYyBkZWYgbGltaXRlZF9yZXF1ZXN0KHJlcV9pbnB1dDogUmVxdWVzdEZ1bmNJbnB1dCkgLT4gUmVxdWVzdEZ1bmNPdXRwdXQ6CiAgICAgICAgYXN5bmMgd2l0aCBzZW1hcGhvcmU6ICAjIHR5cGU6IGlnbm9yZVthdHRyLWRlZmluZWRdCiAgICAgICAgICAgIHJldHVybiBhd2FpdCBBU1lOQ19SRVFVRVNUX0ZVTkNTWyJvcGVuYWktY2hhdCJdKHJlcV9pbnB1dCwgc2Vzc2lvbiwgcGJhcikKCiAgICBhcGlfdXJsID0gcHJveHlfdXJsLnJzdHJpcCgiLyIpICsgIi92MS9jaGF0L2NvbXBsZXRpb25zIgogICAgdGFza3M6IGxpc3RbYXN5bmNpby5UYXNrXSA9IFtdCiAgICBzdGFydF90aW1lID0gdGltZS5wZXJmX2NvdW50ZXIoKQoKICAgIGFzeW5jIGZvciByZXF1ZXN0IGluIGdldF9yZXF1ZXN0KGlucHV0X3JlcXVlc3RzLCByZXF1ZXN0X3JhdGUsIGJ1cnN0aW5lc3MpOgogICAgICAgIHJlcV9pbnB1dCA9IFJlcXVlc3RGdW5jSW5wdXQoCiAgICAgICAgICAgIG1vZGVsPW1vZGVsX2lkLAogICAgICAgICAgICBtb2RlbF9uYW1lPW1vZGVsX25hbWUsCiAgICAgICAgICAgIHByb21wdD1yZXF1ZXN0LnByb21wdCwKICAgICAgICAgICAgYXBpX3VybD1hcGlfdXJsLAogICAgICAgICAgICBwcm9tcHRfbGVuPXJlcXVlc3QucHJvbXB0X2xlbiwKICAgICAgICAgICAgb3V0cHV0X2xlbj1yZXF1ZXN0LmV4cGVjdGVkX291dHB1dF9sZW4gb3IgMTI4LAogICAgICAgICAgICBsb2dwcm9icz1Ob25lLAogICAgICAgICAgICBtdWx0aV9tb2RhbF9jb250ZW50PXJlcXVlc3QubXVsdGlfbW9kYWxfZGF0YSwKICAgICAgICAgICAgaWdub3JlX2Vvcz1GYWxzZSwKICAgICAgICApCiAgICAgICAgdGFza3MuYXBwZW5kKGFzeW5jaW8uY3JlYXRlX3Rhc2sobGltaXRlZF9yZXF1ZXN0KHJlcV9pbnB1dCkpKQoKICAgIG91dHB1dHM6IGxpc3RbUmVxdWVzdEZ1bmNPdXRwdXRdID0gYXdhaXQgYXN5bmNpby5nYXRoZXIoKnRhc2tzKQogICAgZHVyYXRpb24gPSB0aW1lLnBlcmZfY291bnRlcigpIC0gc3RhcnRfdGltZQoKICAgIGlmIHBiYXI6CiAgICAgICAgcGJhci5jbG9zZSgpCiAgICBhd2FpdCBzZXNzaW9uLmNsb3NlKCkKCiAgICByZXR1cm4gX2NvbXB1dGVfbWV0cmljcyhvdXRwdXRzLCBkdXJhdGlvbiwgdG9rZW5pemVyKQoKCmRlZiBfY29tcHV0ZV9tZXRyaWNzKAogICAgb3V0cHV0czogbGlzdFtSZXF1ZXN0RnVuY091dHB1dF0sCiAgICBkdXJhdGlvbjogZmxvYXQsCiAgICB0b2tlbml6ZXI6IEFueSwKKSAtPiBEaXNhZ2dNZXRyaWNzOgogICAgdHRmdHM6IGxpc3RbZmxvYXRdID0gW10KICAgIHRwb3RzOiBsaXN0W2Zsb2F0XSA9IFtdCiAgICBsYXRlbmNpZXM6IGxpc3RbZmxvYXRdID0gW10KICAgIGl0bHM6IGxpc3RbbGlzdFtmbG9hdF1dID0gW10KICAgIHRvdGFsX2lucHV0ID0gMAogICAgdG90YWxfb3V0cHV0ID0gMAogICAgY29tcGxldGVkID0gMAogICAgZmFpbGVkID0gMAoKICAgIGZvciBvdXRwdXQgaW4gb3V0cHV0czoKICAgICAgICBpZiBvdXRwdXQuc3VjY2VzczoKICAgICAgICAgICAgY29tcGxldGVkICs9IDEKICAgICAgICAgICAgdG90YWxfaW5wdXQgKz0gb3V0cHV0LnByb21wdF9sZW4KICAgICAgICAgICAgb3V0cHV0X2xlbiA9IG91dHB1dC5vdXRwdXRfdG9rZW5zCiAgICAgICAgICAgIGlmIG5vdCBvdXRwdXRfbGVuIGFuZCB0b2tlbml6ZXIgaXMgbm90IE5vbmU6CiAgICAgICAgICAgICAgICBvdXRwdXRfbGVuID0gbGVuKAogICAgICAgICAgICAgICAgICAgIHRva2VuaXplcihvdXRwdXQuZ2VuZXJhdGVkX3RleHQsIGFkZF9zcGVjaWFsX3Rva2Vucz1GYWxzZSkuaW5wdXRfaWRzCiAgICAgICAgICAgICAgICApCiAgICAgICAgICAgIHRvdGFsX291dHB1dCArPSBvdXRwdXRfbGVuIG9yIDAKICAgICAgICAgICAgdHRmdHMuYXBwZW5kKG91dHB1dC50dGZ0KQogICAgICAgICAgICBsYXRlbmNpZXMuYXBwZW5kKG91dHB1dC5sYXRlbmN5KQogICAgICAgICAgICBpdGxzLmFwcGVuZChvdXRwdXQuaXRsKQogICAgICAgICAgICBpZiBvdXRwdXRfbGVuIGFuZCBvdXRwdXRfbGVuID4gMToKICAgICAgICAgICAgICAgIHRwb3QgPSAob3V0cHV0LmxhdGVuY3kgLSBvdXRwdXQudHRmdCkgLyAob3V0cHV0X2xlbiAtIDEpCiAgICAgICAgICAgICAgICB0cG90cy5hcHBlbmQodHBvdCkKICAgICAgICBlbHNlOgogICAgICAgICAgICBmYWlsZWQgKz0gMQoKICAgIHJldHVybiBEaXNhZ2dNZXRyaWNzKAogICAgICAgIGNvbXBsZXRlZD1jb21wbGV0ZWQsCiAgICAgICAgZmFpbGVkPWZhaWxlZCwKICAgICAgICB0b3RhbF9pbnB1dF90b2tlbnM9dG90YWxfaW5wdXQsCiAgICAgICAgdG90YWxfb3V0cHV0X3Rva2Vucz10b3RhbF9vdXRwdXQsCiAgICAgICAgbWVhbl90dGZ0X21zPW5wLm1lYW4odHRmdHMpICogMTAwMCBpZiB0dGZ0cyBlbHNlIDAuMCwKICAgICAgICBtZWRpYW5fdHRmdF9tcz1ucC5tZWRpYW4odHRmdHMpICogMTAwMCBpZiB0dGZ0cyBlbHNlIDAuMCwKICAgICAgICBzdGRfdHRmdF9tcz1ucC5zdGQodHRmdHMpICogMTAwMCBpZiB0dGZ0cyBlbHNlIDAuMCwKICAgICAgICBwOTlfdHRmdF9tcz1ucC5wZXJjZW50aWxlKHR0ZnRzLCA5OSkgKiAxMDAwIGlmIHR0ZnRzIGVsc2UgMC4wLAogICAgICAgIG1lYW5fdHBvdF9tcz1ucC5tZWFuKHRwb3RzKSAqIDEwMDAgaWYgdHBvdHMgZWxzZSAwLjAsCiAgICAgICAgbWVkaWFuX3Rwb3RfbXM9bnAubWVkaWFuKHRwb3RzKSAqIDEwMDAgaWYgdHBvdHMgZWxzZSAwLjAsCiAgICAgICAgc3RkX3Rwb3RfbXM9bnAuc3RkKHRwb3RzKSAqIDEwMDAgaWYgdHBvdHMgZWxzZSAwLjAsCiAgICAgICAgcDk5X3Rwb3RfbXM9bnAucGVyY2VudGlsZSh0cG90cywgOTkpICogMTAwMCBpZiB0cG90cyBlbHNlIDAuMCwKICAgICAgICByZXF1ZXN0X3Rocm91Z2hwdXQ9Y29tcGxldGVkIC8gZHVyYXRpb24gaWYgZHVyYXRpb24gPiAwIGVsc2UgMC4wLAogICAgICAgIGlucHV0X3Rva2VuX3Rocm91Z2hwdXQ9dG90YWxfaW5wdXQgLyBkdXJhdGlvbiBpZiBkdXJhdGlvbiA+IDAgZWxzZSAwLjAsCiAgICAgICAgb3V0cHV0X3Rva2VuX3Rocm91Z2hwdXQ9dG90YWxfb3V0cHV0IC8gZHVyYXRpb24gaWYgZHVyYXRpb24gPiAwIGVsc2UgMC4wLAogICAgICAgIHR0ZnRzPXR0ZnRzLAogICAgICAgIHRwb3RzPXRwb3RzLAogICAgICAgIGxhdGVuY2llcz1sYXRlbmNpZXMsCiAgICAgICAgaXRscz1pdGxzLAogICAgKQoKCmRlZiBwcmludF9tZXRyaWNzKG1ldHJpY3M6IERpc2FnZ01ldHJpY3MsIG1vZGU6IHN0cikgLT4gTm9uZToKICAgIHByaW50KGYiXG57Jz0nKjYwfSIpCiAgICBwcmludChmIiAgRGlzYWdncmVnYXRlZCBTZXJ2aW5nIEJlbmNobWFyayBSZXN1bHQgW3ttb2RlLnVwcGVyKCl9XSIpCiAgICBwcmludChmInsnPScqNjB9IikKICAgIHByaW50KGYiICBTdWNjZXNzZnVsIHJlcXVlc3RzOiAgICAgICAge21ldHJpY3MuY29tcGxldGVkfSIpCiAgICBwcmludChmIiAgRmFpbGVkIHJlcXVlc3RzOiAgICAgICAgICAgIHttZXRyaWNzLmZhaWxlZH0iKQogICAgcHJpbnQoZiIgIFRvdGFsIGlucHV0IHRva2VuczogICAgICAgICB7bWV0cmljcy50b3RhbF9pbnB1dF90b2tlbnN9IikKICAgIHByaW50KGYiICBUb3RhbCBvdXRwdXQgdG9rZW5zOiAgICAgICAge21ldHJpY3MudG90YWxfb3V0cHV0X3Rva2Vuc30iKQogICAgcHJpbnQoZiJ7Jy0nKjYwfSIpCiAgICBwcmludChmIiAgUmVxdWVzdCB0aHJvdWdocHV0OiAgICAgICAgIHttZXRyaWNzLnJlcXVlc3RfdGhyb3VnaHB1dDouMmZ9IHJlcS9zIikKICAgIHByaW50KGYiICBJbnB1dCB0b2tlbiB0aHJvdWdocHV0OiAgICAge21ldHJpY3MuaW5wdXRfdG9rZW5fdGhyb3VnaHB1dDouMmZ9IHRvay9zIikKICAgIHByaW50KGYiICBPdXRwdXQgdG9rZW4gdGhyb3VnaHB1dDogICAge21ldHJpY3Mub3V0cHV0X3Rva2VuX3Rocm91Z2hwdXQ6LjJmfSB0b2svcyIpCiAgICBwcmludChmInsnLScqNjB9IikKICAgIHByaW50KGYiICBUVEZUIChtcykiKQogICAgcHJpbnQoZiIgICAgTWVhbjogICB7bWV0cmljcy5tZWFuX3R0ZnRfbXM6LjJmfSIpCiAgICBwcmludChmIiAgICBNZWRpYW46IHttZXRyaWNzLm1lZGlhbl90dGZ0X21zOi4yZn0iKQogICAgcHJpbnQoZiIgICAgU3RkOiAgICB7bWV0cmljcy5zdGRfdHRmdF9tczouMmZ9IikKICAgIHByaW50KGYiICAgIFA5OTogICAge21ldHJpY3MucDk5X3R0ZnRfbXM6LjJmfSIpCiAgICBwcmludChmInsnLScqNjB9IikKICAgIHByaW50KGYiICBUUE9UIChtcykiKQogICAgcHJpbnQoZiIgICAgTWVhbjogICB7bWV0cmljcy5tZWFuX3Rwb3RfbXM6LjJmfSIpCiAgICBwcmludChmIiAgICBNZWRpYW46IHttZXRyaWNzLm1lZGlhbl90cG90X21zOi4yZn0iKQogICAgcHJpbnQoZiIgICAgU3RkOiAgICB7bWV0cmljcy5zdGRfdHBvdF9tczouMmZ9IikKICAgIHByaW50KGYiICAgIFA5OTogICAge21ldHJpY3MucDk5X3Rwb3RfbXM6LjJmfSIpCiAgICBwcmludChmInsnPScqNjB9XG4iKQoKCmRlZiBhZGRfY2xpX2FyZ3MocGFyc2VyOiBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcikgLT4gTm9uZToKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tbW9kZSIsCiAgICAgICAgdHlwZT1zdHIsCiAgICAgICAgcmVxdWlyZWQ9VHJ1ZSwKICAgICAgICBjaG9pY2VzPVsicHJlZmlsbCIsICJkZWNvZGUiLCAiZTJlIl0sCiAgICAgICAgaGVscD0iQmVuY2htYXJrIG1vZGU6IHByZWZpbGwgKHByZWZpbGwgbm9kZSBvbmx5KSwgIgogICAgICAgICJkZWNvZGUgKGRlY29kZSBub2RlIGFmdGVyIHByZWZpbGwgd2FybXVwKSwgZTJlICh2aWEgcHJveHkpIiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tcHJlZmlsbC11cmwiLAogICAgICAgIHR5cGU9c3RyLAogICAgICAgIGRlZmF1bHQ9Imh0dHA6Ly9sb2NhbGhvc3Q6ODEwMCIsCiAgICAgICAgaGVscD0iUHJlZmlsbCBzZXJ2aWNlIFVSTCIsCiAgICApCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KAogICAgICAgICItLWRlY29kZS11cmwiLAogICAgICAgIHR5cGU9c3RyLAogICAgICAgIGRlZmF1bHQ9Imh0dHA6Ly9sb2NhbGhvc3Q6ODIwMCIsCiAgICAgICAgaGVscD0iRGVjb2RlIHNlcnZpY2UgVVJMIiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tcHJveHktdXJsIiwKICAgICAgICB0eXBlPXN0ciwKICAgICAgICBkZWZhdWx0PSJodHRwOi8vbG9jYWxob3N0OjgwMDAiLAogICAgICAgIGhlbHA9IlByb3h5IHNlcnZpY2UgVVJMIChmb3IgZTJlIG1vZGUpIiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tcHJlZmlsbC1rdi1wb3J0IiwKICAgICAgICB0eXBlPWludCwKICAgICAgICBkZWZhdWx0PTE0NTc5LAogICAgICAgIGhlbHA9IlByZWZpbGwgS1YgdHJhbnNmZXIgcG9ydCIsCiAgICApCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KAogICAgICAgICItLWRlY29kZS1rdi1wb3J0IiwKICAgICAgICB0eXBlPWludCwKICAgICAgICBkZWZhdWx0PTE0NTgwLAogICAgICAgIGhlbHA9IkRlY29kZSBLViB0cmFuc2ZlciBwb3J0IiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tbW9kZWwiLAogICAgICAgIHR5cGU9c3RyLAogICAgICAgIHJlcXVpcmVkPVRydWUsCiAgICAgICAgaGVscD0iTW9kZWwgbmFtZSIsCiAgICApCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KAogICAgICAgICItLXJlcXVlc3QtcmF0ZSIsCiAgICAgICAgdHlwZT1mbG9hdCwKICAgICAgICBkZWZhdWx0PWZsb2F0KCJpbmYiKSwKICAgICAgICBoZWxwPSJSZXF1ZXN0IHJhdGUgaW4gcmVxL3MgKGRlZmF1bHQ6IGluZiA9IGJ1cnN0KSIsCiAgICApCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KAogICAgICAgICItLWJ1cnN0aW5lc3MiLAogICAgICAgIHR5cGU9ZmxvYXQsCiAgICAgICAgZGVmYXVsdD0xLjAsCiAgICAgICAgaGVscD0iQnVyc3RpbmVzcyBmYWN0b3IgKDEuMCA9IFBvaXNzb24pIiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tbWF4LWNvbmN1cnJlbmN5IiwKICAgICAgICB0eXBlPWludCwKICAgICAgICBkZWZhdWx0PU5vbmUsCiAgICAgICAgaGVscD0iTWF4aW11bSBjb25jdXJyZW50IHJlcXVlc3RzIiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tZGlzYWJsZS10cWRtIiwKICAgICAgICBhY3Rpb249InN0b3JlX3RydWUiLAogICAgICAgIGhlbHA9IkRpc2FibGUgcHJvZ3Jlc3MgYmFyIiwKICAgICkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoCiAgICAgICAgIi0tc2F2ZS1yZXN1bHQiLAogICAgICAgIGFjdGlvbj0ic3RvcmVfdHJ1ZSIsCiAgICAgICAgaGVscD0iU2F2ZSByZXN1bHRzIHRvIEpTT04gZmlsZSIsCiAgICApCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KAogICAgICAgICItLXJlc3VsdC1kaXIiLAogICAgICAgIHR5cGU9c3RyLAogICAgICAgIGRlZmF1bHQ9Ii4iLAogICAgICAgIGhlbHA9IkRpcmVjdG9yeSB0byBzYXZlIHJlc3VsdHMiLAogICAgKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgKICAgICAgICAiLS1yZXN1bHQtZmlsZW5hbWUiLAogICAgICAgIHR5cGU9c3RyLAogICAgICAgIGRlZmF1bHQ9Tm9uZSwKICAgICAgICBoZWxwPSJSZXN1bHQgZmlsZW5hbWUiLAogICAgKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgKICAgICAgICAiLS10b2tlbml6ZXIiLAogICAgICAgIHR5cGU9c3RyLAogICAgICAgIGRlZmF1bHQ9Tm9uZSwKICAgICAgICBoZWxwPSJUb2tlbml6ZXIgbmFtZSBvciBwYXRoIChkZWZhdWx0cyB0byBtb2RlbCkiLAogICAgKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgKICAgICAgICAiLS1uaXhsLW1vZGUiLAogICAgICAgIGFjdGlvbj0ic3RvcmVfdHJ1ZSIsCiAgICAgICAgaGVscD0iRW5hYmxlIE5JWEwgbW9kZSBmb3IgZGVjb2RlIGJlbmNobWFyayAoaW5qZWN0cyBrdl90cmFuc2Zlcl9wYXJhbXMpIiwKICAgICkKCgphc3luYyBkZWYgbWFpbihhcmdzOiBhcmdwYXJzZS5OYW1lc3BhY2UpIC0+IGRpY3Rbc3RyLCBBbnldOgogICAgIyBMb2FkIHRva2VuaXplcgogICAgdG9rZW5pemVyX25hbWUgPSBhcmdzLnRva2VuaXplciBvciBhcmdzLm1vZGVsCiAgICB0b2tlbml6ZXIgPSBnZXRfdG9rZW5pemVyKHRva2VuaXplcl9uYW1lLCB0cnVzdF9yZW1vdGVfY29kZT1UcnVlKQoKICAgICMgTG9hZCBkYXRhc2V0CiAgICBpbnB1dF9yZXF1ZXN0cyA9IGdldF9zYW1wbGVzKGFyZ3MsIHRva2VuaXplcikKICAgIHByaW50KGYiTG9hZGVkIHtsZW4oaW5wdXRfcmVxdWVzdHMpfSByZXF1ZXN0cyBmcm9tIGRhdGFzZXQiKQoKICAgICMgUnVuIGJlbmNobWFyawogICAgaWYgYXJncy5tb2RlID09ICJwcmVmaWxsIjoKICAgICAgICBtZXRyaWNzID0gYXdhaXQgYmVuY2htYXJrX3ByZWZpbGwoCiAgICAgICAgICAgIG1vZGVsX2lkPWFyZ3MubW9kZWwsCiAgICAgICAgICAgIG1vZGVsX25hbWU9Tm9uZSwKICAgICAgICAgICAgdG9rZW5pemVyPXRva2VuaXplciwKICAgICAgICAgICAgaW5wdXRfcmVxdWVzdHM9aW5wdXRfcmVxdWVzdHMsCiAgICAgICAgICAgIHByZWZpbGxfdXJsPWFyZ3MucHJlZmlsbF91cmwsCiAgICAgICAgICAgIHJlcXVlc3RfcmF0ZT1hcmdzLnJlcXVlc3RfcmF0ZSwKICAgICAgICAgICAgYnVyc3RpbmVzcz1hcmdzLmJ1cnN0aW5lc3MsCiAgICAgICAgICAgIG1heF9jb25jdXJyZW5jeT1hcmdzLm1heF9jb25jdXJyZW5jeSwKICAgICAgICAgICAgZGlzYWJsZV90cWRtPWFyZ3MuZGlzYWJsZV90cWRtLAogICAgICAgICkKICAgIGVsaWYgYXJncy5tb2RlID09ICJkZWNvZGUiOgogICAgICAgIG1ldHJpY3MgPSBhd2FpdCBiZW5jaG1hcmtfZGVjb2RlKAogICAgICAgICAgICBtb2RlbF9pZD1hcmdzLm1vZGVsLAogICAgICAgICAgICBtb2RlbF9uYW1lPU5vbmUsCiAgICAgICAgICAgIHRva2VuaXplcj10b2tlbml6ZXIsCiAgICAgICAgICAgIGlucHV0X3JlcXVlc3RzPWlucHV0X3JlcXVlc3RzLAogICAgICAgICAgICBwcmVmaWxsX3VybD1hcmdzLnByZWZpbGxfdXJsLAogICAgICAgICAgICBkZWNvZGVfdXJsPWFyZ3MuZGVjb2RlX3VybCwKICAgICAgICAgICAgcHJlZmlsbF9rdl9wb3J0PWFyZ3MucHJlZmlsbF9rdl9wb3J0LAogICAgICAgICAgICBkZWNvZGVfa3ZfcG9ydD1hcmdzLmRlY29kZV9rdl9wb3J0LAogICAgICAgICAgICByZXF1ZXN0X3JhdGU9YXJncy5yZXF1ZXN0X3JhdGUsCiAgICAgICAgICAgIGJ1cnN0aW5lc3M9YXJncy5idXJzdGluZXNzLAogICAgICAgICAgICBtYXhfY29uY3VycmVuY3k9YXJncy5tYXhfY29uY3VycmVuY3ksCiAgICAgICAgICAgIGRpc2FibGVfdHFkbT1hcmdzLmRpc2FibGVfdHFkbSwKICAgICAgICAgICAgbml4bF9tb2RlPWFyZ3Mubml4bF9tb2RlLAogICAgICAgICkKICAgIGVsc2U6ICAjIGUyZQogICAgICAgIG1ldHJpY3MgPSBhd2FpdCBiZW5jaG1hcmtfZTJlKAogICAgICAgICAgICBtb2RlbF9pZD1hcmdzLm1vZGVsLAogICAgICAgICAgICBtb2RlbF9uYW1lPU5vbmUsCiAgICAgICAgICAgIHRva2VuaXplcj10b2tlbml6ZXIsCiAgICAgICAgICAgIGlucHV0X3JlcXVlc3RzPWlucHV0X3JlcXVlc3RzLAogICAgICAgICAgICBwcm94eV91cmw9YXJncy5wcm94eV91cmwsCiAgICAgICAgICAgIHJlcXVlc3RfcmF0ZT1hcmdzLnJlcXVlc3RfcmF0ZSwKICAgICAgICAgICAgYnVyc3RpbmVzcz1hcmdzLmJ1cnN0aW5lc3MsCiAgICAgICAgICAgIG1heF9jb25jdXJyZW5jeT1hcmdzLm1heF9jb25jdXJyZW5jeSwKICAgICAgICAgICAgZGlzYWJsZV90cWRtPWFyZ3MuZGlzYWJsZV90cWRtLAogICAgICAgICkKCiAgICBwcmludF9tZXRyaWNzKG1ldHJpY3MsIGFyZ3MubW9kZSkKCiAgICAjIFNhdmUgcmVzdWx0cwogICAgcmVzdWx0OiBkaWN0W3N0ciwgQW55XSA9IHsKICAgICAgICAibW9kZSI6IGFyZ3MubW9kZSwKICAgICAgICAibW9kZWwiOiBhcmdzLm1vZGVsLAogICAgICAgICJudW1fcHJvbXB0cyI6IGxlbihpbnB1dF9yZXF1ZXN0cyksCiAgICAgICAgInJlcXVlc3RfcmF0ZSI6IGFyZ3MucmVxdWVzdF9yYXRlLAogICAgICAgICJjb21wbGV0ZWQiOiBtZXRyaWNzLmNvbXBsZXRlZCwKICAgICAgICAiZmFpbGVkIjogbWV0cmljcy5mYWlsZWQsCiAgICAgICAgInRvdGFsX2lucHV0X3Rva2VucyI6IG1ldHJpY3MudG90YWxfaW5wdXRfdG9rZW5zLAogICAgICAgICJ0b3RhbF9vdXRwdXRfdG9rZW5zIjogbWV0cmljcy50b3RhbF9vdXRwdXRfdG9rZW5zLAogICAgICAgICJyZXF1ZXN0X3Rocm91Z2hwdXQiOiBtZXRyaWNzLnJlcXVlc3RfdGhyb3VnaHB1dCwKICAgICAgICAiaW5wdXRfdG9rZW5fdGhyb3VnaHB1dCI6IG1ldHJpY3MuaW5wdXRfdG9rZW5fdGhyb3VnaHB1dCwKICAgICAgICAib3V0cHV0X3Rva2VuX3Rocm91Z2hwdXQiOiBtZXRyaWNzLm91dHB1dF90b2tlbl90aHJvdWdocHV0LAogICAgICAgICJtZWFuX3R0ZnRfbXMiOiBtZXRyaWNzLm1lYW5fdHRmdF9tcywKICAgICAgICAibWVkaWFuX3R0ZnRfbXMiOiBtZXRyaWNzLm1lZGlhbl90dGZ0X21zLAogICAgICAgICJzdGRfdHRmdF9tcyI6IG1ldHJpY3Muc3RkX3R0ZnRfbXMsCiAgICAgICAgInA5OV90dGZ0X21zIjogbWV0cmljcy5wOTlfdHRmdF9tcywKICAgICAgICAibWVhbl90cG90X21zIjogbWV0cmljcy5tZWFuX3Rwb3RfbXMsCiAgICAgICAgIm1lZGlhbl90cG90X21zIjogbWV0cmljcy5tZWRpYW5fdHBvdF9tcywKICAgICAgICAic3RkX3Rwb3RfbXMiOiBtZXRyaWNzLnN0ZF90cG90X21zLAogICAgICAgICJwOTlfdHBvdF9tcyI6IG1ldHJpY3MucDk5X3Rwb3RfbXMsCiAgICAgICAgInR0ZnRzIjogbWV0cmljcy50dGZ0cywKICAgICAgICAidHBvdHMiOiBtZXRyaWNzLnRwb3RzLAogICAgfQoKICAgIGlmIGFyZ3Muc2F2ZV9yZXN1bHQ6CiAgICAgICAgaW1wb3J0IG9zCgogICAgICAgIG9zLm1ha2VkaXJzKGFyZ3MucmVzdWx0X2RpciwgZXhpc3Rfb2s9VHJ1ZSkKICAgICAgICBmaWxlbmFtZSA9IGFyZ3MucmVzdWx0X2ZpbGVuYW1lIG9yIGYiZGlzYWdnX3thcmdzLm1vZGV9X3Jlc3VsdC5qc29uIgogICAgICAgIGZpbGVwYXRoID0gb3MucGF0aC5qb2luKGFyZ3MucmVzdWx0X2RpciwgZmlsZW5hbWUpCiAgICAgICAgd2l0aCBvcGVuKGZpbGVwYXRoLCAidyIpIGFzIGY6CiAgICAgICAgICAgIGpzb24uZHVtcChyZXN1bHQsIGYsIGluZGVudD0yKQogICAgICAgIHByaW50KGYiUmVzdWx0cyBzYXZlZCB0byB7ZmlsZXBhdGh9IikKCiAgICByZXR1cm4gcmVzdWx0CgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIHBhcnNlciA9IEZsZXhpYmxlQXJndW1lbnRQYXJzZXIoCiAgICAgICAgZGVzY3JpcHRpb249IkJlbmNobWFyayB2TExNIFAvRCBkaXNhZ2dyZWdhdGlvbiBwcmVmaWxsL2RlY29kZSBwZXJmb3JtYW5jZSIKICAgICkKICAgIGFkZF9jbGlfYXJncyhwYXJzZXIpCiAgICBhZGRfZGF0YXNldF9wYXJzZXIocGFyc2VyKQogICAgYXJncyA9IHBhcnNlci5wYXJzZV9hcmdzKCkKICAgIGFzeW5jaW8ucnVuKG1haW4oYXJncykpCg=='))" 2>>"$LOG_DIR/run_sh.log"
        }
        if [ -f "$benchmark_script" ]; then
            echo "[RUN.SH] File extracted, size=$(stat -c%s "$benchmark_script") bytes" >> "$LOG_DIR/run_sh.log"
            echo "Using benchmark script: $benchmark_script"

            echo "=== Running NIXL prefill benchmark (direct to prefill) ==="
            set +e
            set -o pipefail
            python "$benchmark_script" \
                --mode prefill \
                --prefill-url "http://${PREFILL_HOST}:${PREFILL_PORT}" \
                --model "$SERVED_MODEL_NAME" \
                --tokenizer "${MODEL_PATH_EFFECTIVE}" \
                --dataset-name "$BENCH_DATASET_NAME" \
                --dataset-path "$BENCH_DATASET_PATH" \
                --num-prompts "$BENCH_NUM_PROMPTS" \
                --sharegpt-output-len "$BENCH_OUTPUT_LEN" \
                --request-rate "$BENCH_REQUEST_RATE" \
                --max-concurrency "$BENCH_MAX_CONCURRENCY" \
                --disable-tqdm \
                --save-result \
                --result-dir "$RESULT_DIR" \
                --result-filename "nixl_prefill_result.json" \
                2>&1 | tee "$LOG_DIR/bench_nixl_prefill.log"
            prefill_exit=$?
            set +o pipefail
            set -e
            if [ $prefill_exit -ne 0 ]; then
                echo "ERROR: Prefill benchmark failed with exit code $prefill_exit"
                cat "$LOG_DIR/bench_nixl_prefill.log" 2>/dev/null | tail -n 30
                sync_results_to_persistent "prefill_benchmark_failed"
            fi

            echo "=== Running NIXL decode benchmark (prefill -> decode) ==="
            set +e
            set -o pipefail
            python "$benchmark_script" \
                --mode decode \
                --prefill-url "http://${PREFILL_HOST}:${PREFILL_PORT}" \
                --decode-url "http://${DECODE_HOST}:${DECODE_PORT}" \
                --model "$SERVED_MODEL_NAME" \
                --tokenizer "${MODEL_PATH_EFFECTIVE}" \
                --dataset-name "$BENCH_DATASET_NAME" \
                --dataset-path "$BENCH_DATASET_PATH" \
                --num-prompts "$BENCH_NUM_PROMPTS" \
                --sharegpt-output-len "$BENCH_OUTPUT_LEN" \
                --request-rate "$BENCH_REQUEST_RATE" \
                --max-concurrency "$BENCH_MAX_CONCURRENCY" \
                --nixl-mode \
                --disable-tqdm \
                --save-result \
                --result-dir "$RESULT_DIR" \
                --result-filename "nixl_decode_result.json" \
                2>&1 | tee "$LOG_DIR/bench_nixl_decode.log"
            decode_exit=$?
            set +o pipefail
            set -e
            if [ $decode_exit -ne 0 ]; then
                echo "ERROR: Decode benchmark failed with exit code $decode_exit"
                cat "$LOG_DIR/bench_nixl_decode.log" 2>/dev/null | tail -n 30
                sync_results_to_persistent "decode_benchmark_failed"
            fi
        else
            echo "[RUN.SH] File NOT found at $benchmark_script" >> "$LOG_DIR/run_sh.log"
            echo "[RUN.SH] Searching /app for file:" >> "$LOG_DIR/run_sh.log"
            find /app -name "benchmark_disagg_prefill_decode.py" 2>/dev/null >> "$LOG_DIR/run_sh.log" || true
            echo "WARNING: benchmark_disagg_prefill_decode.py not found, skipping separate prefill/decode benchmarks"
        fi
        sync_results_to_persistent "before_e2e"

        echo "=== Running NIXL e2e benchmark via proxy ==="
        set +e
        set -o pipefail
        run_proxy_benchmark "nixl_e2e_result.json" "$LOG_DIR/bench_nixl_e2e.log"
        e2e_exit=$?
        set +o pipefail
        set -e
        if [ $e2e_exit -ne 0 ]; then
            echo "ERROR: E2E benchmark failed with exit code $e2e_exit"
            cat "$LOG_DIR/bench_nixl_e2e.log" | tail -n 30
            sync_results_to_persistent "e2e_benchmark_failed"
        fi

        echo "=== NIXL benchmark JSON Results ==="
        for result_file in "$RESULT_DIR"/nixl_*.json; do
            if [ -f "$result_file" ]; then
                echo "--- $(basename "$result_file") ---"
                cat "$result_file"
                echo ""
            fi
        done
        sync_results_to_persistent "benchmark_complete"

        kill "${PROXY_PID}" 2>/dev/null || true
        wait "${PROXY_PID}" 2>/dev/null || true
        kill "${PREFILL_PID}" 2>/dev/null || true
        wait "${PREFILL_PID}" 2>/dev/null || true
        exit 0
    fi

    if [ "$NODE_RANK" = "1" ]; then
        require_python_module nixl

        echo "=== Starting NIXL DECODE node on ${DECODE_HOST}:${DECODE_PORT} ==="
        start_vllm_server \
            "decode" \
            "${DECODE_HOST}" \
            "${DECODE_PORT}" \
            "$(build_nixl_kv_config)" \
            "$LOG_DIR/vllm_decode.log" \
            "${DECODE_SIDE_CHANNEL_PORT}"

        wait_for_http "decode" "http://localhost:${DECODE_PORT}/health" 3600 "$LOG_DIR/vllm_decode.log"
        wait_for_prefill_controller_exit "$LOG_DIR/vllm_decode.log"

        if kill -0 "${DECODE_PID}" 2>/dev/null; then
            kill "${DECODE_PID}" 2>/dev/null || true
            wait "${DECODE_PID}" 2>/dev/null || true
            exit 0
        fi

        echo "ERROR: Decode process exited unexpectedly"
        debug_pause_on_failure "$LOG_DIR/vllm_decode.log"
        exit 1
    fi
}

prepare_model_path
print_debug_context
build_server_extra_args

case "$PD_ROUTE" in
    p2p)
        run_p2p_route
        ;;
    nixl)
        run_nixl_route
        ;;
    *)
        echo "ERROR: Unsupported PD_ROUTE=${PD_ROUTE}"
        exit 1
        ;;
esac

echo "ERROR: Unsupported NODE_RANK=${NODE_RANK}"
exit 1
