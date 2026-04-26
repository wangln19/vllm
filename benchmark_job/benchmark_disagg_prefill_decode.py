#!/usr/bin/env python3
"""
Benchmark prefill and decode throughput separately for vLLM P/D disaggregation.

Supports ShareGPT dataset and can target prefill node, decode node, or the
proxy directly for end-to-end measurement.

Example usage:

  # 1. Prefill benchmark (max_tokens=1, measures TTFT)
  python benchmark_disagg_prefill_decode.py \
    --mode prefill \
    --prefill-url http://localhost:8100 \
    --model <model_name> \
    --dataset-name sharegpt \
    --dataset-path /mnt/moonfs/integration-m4/texts/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 100 \
    --sharegpt-output-len 128

  # 2. Decode benchmark (needs prefill node to warm up KV cache first)
  python benchmark_disagg_prefill_decode.py \
    --mode decode \
    --prefill-url http://localhost:8100 \
    --decode-url http://localhost:8200 \
    --model <model_name> \
    --dataset-name sharegpt \
    --dataset-path /mnt/moonfs/integration-m4/texts/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 100 \
    --sharegpt-output-len 128

  # 3. End-to-end via proxy
  python benchmark_disagg_prefill_decode.py \
    --mode e2e \
    --proxy-url http://localhost:8000 \
    --model <model_name> \
    --dataset-name sharegpt \
    --dataset-path /mnt/moonfs/integration-m4/texts/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 100 \
    --sharegpt-output-len 128
"""

import argparse
import asyncio
import contextlib
import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from vllm.benchmarks.datasets import SampleRequest, add_dataset_parser, get_samples
from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)
from vllm.tokenizers import get_tokenizer
from vllm.utils.argparse_utils import FlexibleArgumentParser

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class DisaggMetrics:
    completed: int
    failed: int
    total_input_tokens: int
    total_output_tokens: int
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    request_throughput: float
    input_token_throughput: float
    output_token_throughput: float
    ttfts: list[float] = field(default_factory=list)
    tpots: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    itls: list[list[float]] = field(default_factory=list)


def build_kv_headers(
    prefill_url: str,
    decode_url: str,
    prefill_kv_port: int = 14579,
    decode_kv_port: int = 14580,
) -> tuple[str, dict[str, str]]:
    """Build request_id and headers for KV transfer between prefill and decode."""
    from urllib.parse import urlparse

    prefill_parsed = urlparse(prefill_url)
    decode_parsed = urlparse(decode_url)

    prefill_host = prefill_parsed.hostname or "localhost"
    decode_host = decode_parsed.hostname or "localhost"

    prefill_kv_addr = f"{prefill_host}:{prefill_kv_port}"
    decode_kv_addr = f"{decode_host}:{decode_kv_port}"

    request_id = (
        f"___prefill_addr_{prefill_kv_addr}___decode_addr_"
        f"{decode_kv_addr}_{uuid.uuid4().hex}"
    )

    headers = {
        "X-Request-Id": request_id,
        "X-KV-Target": f"{decode_host}:{decode_parsed.port or 80}",
    }
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return request_id, headers


async def async_request_prefill_only(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """Send a request with max_tokens=1 to measure prefill time (TTFT)."""
    return await ASYNC_REQUEST_FUNCS["openai-chat"](request_func_input, session, pbar)


async def async_request_decode_after_prefill(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    prefill_url: str,
    decode_url: str,
    prefill_kv_port: int,
    decode_kv_port: int,
    nixl_mode: bool = False,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """
    First send max_tokens=1 to prefill node to warm up KV cache,
    then send the full request to decode node and measure decode performance.
    """
    request_id, headers = build_kv_headers(
        prefill_url, decode_url, prefill_kv_port, decode_kv_port
    )

    # 1. Prefill stage: max_tokens=1
    prefill_input = RequestFuncInput(
        model=request_func_input.model,
        model_name=request_func_input.model_name,
        prompt=request_func_input.prompt,
        api_url=prefill_url.rstrip("/") + "/v1/chat/completions",
        prompt_len=request_func_input.prompt_len,
        output_len=1,
        logprobs=request_func_input.logprobs,
        multi_modal_content=request_func_input.multi_modal_content,
        ignore_eos=request_func_input.ignore_eos,
        extra_headers=headers,
        extra_body=request_func_input.extra_body,
        request_id=request_id,
    )

    if nixl_mode:
        # In NIXL mode, send a non-streaming prefill request with
        # kv_transfer_params to get the remote KV metadata back.
        import aiohttp
        payload = {
            "model": request_func_input.model_name or request_func_input.model,
            "messages": [{"role": "user", "content": request_func_input.prompt}],
            "max_completion_tokens": 1,
            "stream": False,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            },
        }
        try:
            async with session.post(
                prefill_input.api_url, json=payload, headers=headers
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    if pbar:
                        pbar.update(1)
                    return RequestFuncOutput(
                        success=False,
                        error=f"NIXL prefill failed: HTTP {resp.status} - {text}",
                    )
                data = await resp.json()
                kv_transfer_params = data.get("kv_transfer_params")
                if not kv_transfer_params:
                    if pbar:
                        pbar.update(1)
                    return RequestFuncOutput(
                        success=False,
                        error="NIXL prefill response missing kv_transfer_params",
                    )
        except Exception as e:
            if pbar:
                pbar.update(1)
            return RequestFuncOutput(
                success=False,
                error=f"NIXL prefill request exception: {e}",
            )
    else:
        prefill_output = await ASYNC_REQUEST_FUNCS["openai-chat"](
            prefill_input, session, None
        )
        if not prefill_output.success:
            if pbar:
                pbar.update(1)
            return RequestFuncOutput(
                success=False,
                error=f"Prefill failed: {prefill_output.error}",
            )
        kv_transfer_params = None

    # 2. Decode stage: full output_len
    decode_extra_body = dict(request_func_input.extra_body) if request_func_input.extra_body else {}
    if nixl_mode and kv_transfer_params:
        decode_extra_body["kv_transfer_params"] = kv_transfer_params

    decode_input = RequestFuncInput(
        model=request_func_input.model,
        model_name=request_func_input.model_name,
        prompt=request_func_input.prompt,
        api_url=decode_url.rstrip("/") + "/v1/chat/completions",
        prompt_len=request_func_input.prompt_len,
        output_len=request_func_input.output_len,
        logprobs=request_func_input.logprobs,
        multi_modal_content=request_func_input.multi_modal_content,
        ignore_eos=request_func_input.ignore_eos,
        extra_headers=headers,
        extra_body=decode_extra_body,
        request_id=request_id,
    )

    decode_output = await ASYNC_REQUEST_FUNCS["openai-chat"](decode_input, session, pbar)
    if not decode_output.success:
        return decode_output

    # For decode mode, we want the decode TTFT to represent the time until
    # the first decode token (which includes KV transfer + decode first token).
    # The total latency is from sending decode request to receiving all tokens.
    # We preserve the original metrics but rename them for clarity in reporting.
    return decode_output


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[SampleRequest, None]:
    """Generate requests at specified rate."""
    import numpy as np

    total_requests = len(input_requests)
    delay_ts = []
    for _ in range(total_requests):
        if request_rate == float("inf"):
            delay_ts.append(0)
        else:
            theta = 1.0 / (request_rate * burstiness)
            delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))

    for i in range(1, len(delay_ts)):
        delay_ts[i] += delay_ts[i - 1]

    if request_rate != float("inf") and delay_ts:
        target_total = total_requests / request_rate
        if delay_ts[-1] != 0:
            normalize_factor = target_total / delay_ts[-1]
            delay_ts = [d * normalize_factor for d in delay_ts]

    start_ts = time.time()
    for i, request in enumerate(input_requests):
        if delay_ts[i] > 0:
            sleep_interval = start_ts + delay_ts[i] - time.time()
            if sleep_interval > 0:
                await asyncio.sleep(sleep_interval)
        yield request


async def benchmark_prefill(
    model_id: str,
    model_name: str | None,
    tokenizer: Any,
    input_requests: list[SampleRequest],
    prefill_url: str,
    request_rate: float,
    burstiness: float,
    max_concurrency: int | None,
    disable_tqdm: bool,
) -> DisaggMetrics:
    """Benchmark prefill node by sending max_tokens=1 requests."""
    connector = aiohttp.TCPConnector(
        limit=max_concurrency or 0,
        limit_per_host=max_concurrency or 0,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=AIOHTTP_TIMEOUT,
    )

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency
        else contextlib.nullcontext()
    )

    async def limited_request(req_input: RequestFuncInput) -> RequestFuncOutput:
        async with semaphore:  # type: ignore[attr-defined]
            return await async_request_prefill_only(req_input, session, pbar)

    api_url = prefill_url.rstrip("/") + "/v1/chat/completions"
    tasks: list[asyncio.Task] = []
    start_time = time.perf_counter()

    async for request in get_request(input_requests, request_rate, burstiness):
        req_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=1,  # prefill only
            logprobs=None,
            multi_modal_content=request.multi_modal_data,
            ignore_eos=False,
        )
        tasks.append(asyncio.create_task(limited_request(req_input)))

    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time

    if pbar:
        pbar.close()
    await session.close()

    return _compute_metrics(outputs, duration, tokenizer)


async def benchmark_decode(
    model_id: str,
    model_name: str | None,
    tokenizer: Any,
    input_requests: list[SampleRequest],
    prefill_url: str,
    decode_url: str,
    prefill_kv_port: int,
    decode_kv_port: int,
    request_rate: float,
    burstiness: float,
    max_concurrency: int | None,
    disable_tqdm: bool,
    nixl_mode: bool = False,
) -> DisaggMetrics:
    """Benchmark decode node after warming up KV cache on prefill node."""
    connector = aiohttp.TCPConnector(
        limit=max_concurrency or 0,
        limit_per_host=max_concurrency or 0,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=AIOHTTP_TIMEOUT,
    )

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency
        else contextlib.nullcontext()
    )

    async def limited_request(req_input: RequestFuncInput) -> RequestFuncOutput:
        async with semaphore:  # type: ignore[attr-defined]
            return await async_request_decode_after_prefill(
                req_input,
                session,
                prefill_url,
                decode_url,
                prefill_kv_port,
                decode_kv_port,
                nixl_mode,
                pbar,
            )

    tasks: list[asyncio.Task] = []
    start_time = time.perf_counter()

    async for request in get_request(input_requests, request_rate, burstiness):
        req_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=request.prompt,
            api_url="",  # not used directly
            prompt_len=request.prompt_len,
            output_len=request.expected_output_len or 128,
            logprobs=None,
            multi_modal_content=request.multi_modal_data,
            ignore_eos=False,
        )
        tasks.append(asyncio.create_task(limited_request(req_input)))

    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time

    if pbar:
        pbar.close()
    await session.close()

    return _compute_metrics(outputs, duration, tokenizer)


async def benchmark_e2e(
    model_id: str,
    model_name: str | None,
    tokenizer: Any,
    input_requests: list[SampleRequest],
    proxy_url: str,
    request_rate: float,
    burstiness: float,
    max_concurrency: int | None,
    disable_tqdm: bool,
) -> DisaggMetrics:
    """Benchmark end-to-end via the disaggregation proxy."""
    connector = aiohttp.TCPConnector(
        limit=max_concurrency or 0,
        limit_per_host=max_concurrency or 0,
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=AIOHTTP_TIMEOUT,
    )

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency
        else contextlib.nullcontext()
    )

    async def limited_request(req_input: RequestFuncInput) -> RequestFuncOutput:
        async with semaphore:  # type: ignore[attr-defined]
            return await ASYNC_REQUEST_FUNCS["openai-chat"](req_input, session, pbar)

    api_url = proxy_url.rstrip("/") + "/v1/chat/completions"
    tasks: list[asyncio.Task] = []
    start_time = time.perf_counter()

    async for request in get_request(input_requests, request_rate, burstiness):
        req_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.expected_output_len or 128,
            logprobs=None,
            multi_modal_content=request.multi_modal_data,
            ignore_eos=False,
        )
        tasks.append(asyncio.create_task(limited_request(req_input)))

    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time

    if pbar:
        pbar.close()
    await session.close()

    return _compute_metrics(outputs, duration, tokenizer)


def _compute_metrics(
    outputs: list[RequestFuncOutput],
    duration: float,
    tokenizer: Any,
) -> DisaggMetrics:
    ttfts: list[float] = []
    tpots: list[float] = []
    latencies: list[float] = []
    itls: list[list[float]] = []
    total_input = 0
    total_output = 0
    completed = 0
    failed = 0

    for output in outputs:
        if output.success:
            completed += 1
            total_input += output.prompt_len
            output_len = output.output_tokens
            if not output_len and tokenizer is not None:
                output_len = len(
                    tokenizer(output.generated_text, add_special_tokens=False).input_ids
                )
            total_output += output_len or 0
            ttfts.append(output.ttft)
            latencies.append(output.latency)
            itls.append(output.itl)
            if output_len and output_len > 1:
                tpot = (output.latency - output.ttft) / (output_len - 1)
                tpots.append(tpot)
        else:
            failed += 1

    return DisaggMetrics(
        completed=completed,
        failed=failed,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        mean_ttft_ms=np.mean(ttfts) * 1000 if ttfts else 0.0,
        median_ttft_ms=np.median(ttfts) * 1000 if ttfts else 0.0,
        std_ttft_ms=np.std(ttfts) * 1000 if ttfts else 0.0,
        p99_ttft_ms=np.percentile(ttfts, 99) * 1000 if ttfts else 0.0,
        mean_tpot_ms=np.mean(tpots) * 1000 if tpots else 0.0,
        median_tpot_ms=np.median(tpots) * 1000 if tpots else 0.0,
        std_tpot_ms=np.std(tpots) * 1000 if tpots else 0.0,
        p99_tpot_ms=np.percentile(tpots, 99) * 1000 if tpots else 0.0,
        request_throughput=completed / duration if duration > 0 else 0.0,
        input_token_throughput=total_input / duration if duration > 0 else 0.0,
        output_token_throughput=total_output / duration if duration > 0 else 0.0,
        ttfts=ttfts,
        tpots=tpots,
        latencies=latencies,
        itls=itls,
    )


def print_metrics(metrics: DisaggMetrics, mode: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Disaggregated Serving Benchmark Result [{mode.upper()}]")
    print(f"{'='*60}")
    print(f"  Successful requests:        {metrics.completed}")
    print(f"  Failed requests:            {metrics.failed}")
    print(f"  Total input tokens:         {metrics.total_input_tokens}")
    print(f"  Total output tokens:        {metrics.total_output_tokens}")
    print(f"{'-'*60}")
    print(f"  Request throughput:         {metrics.request_throughput:.2f} req/s")
    print(f"  Input token throughput:     {metrics.input_token_throughput:.2f} tok/s")
    print(f"  Output token throughput:    {metrics.output_token_throughput:.2f} tok/s")
    print(f"{'-'*60}")
    print(f"  TTFT (ms)")
    print(f"    Mean:   {metrics.mean_ttft_ms:.2f}")
    print(f"    Median: {metrics.median_ttft_ms:.2f}")
    print(f"    Std:    {metrics.std_ttft_ms:.2f}")
    print(f"    P99:    {metrics.p99_ttft_ms:.2f}")
    print(f"{'-'*60}")
    print(f"  TPOT (ms)")
    print(f"    Mean:   {metrics.mean_tpot_ms:.2f}")
    print(f"    Median: {metrics.median_tpot_ms:.2f}")
    print(f"    Std:    {metrics.std_tpot_ms:.2f}")
    print(f"    P99:    {metrics.p99_tpot_ms:.2f}")
    print(f"{'='*60}\n")


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["prefill", "decode", "e2e"],
        help="Benchmark mode: prefill (prefill node only), "
        "decode (decode node after prefill warmup), e2e (via proxy)",
    )
    parser.add_argument(
        "--prefill-url",
        type=str,
        default="http://localhost:8100",
        help="Prefill service URL",
    )
    parser.add_argument(
        "--decode-url",
        type=str,
        default="http://localhost:8200",
        help="Decode service URL",
    )
    parser.add_argument(
        "--proxy-url",
        type=str,
        default="http://localhost:8000",
        help="Proxy service URL (for e2e mode)",
    )
    parser.add_argument(
        "--prefill-kv-port",
        type=int,
        default=14579,
        help="Prefill KV transfer port",
    )
    parser.add_argument(
        "--decode-kv-port",
        type=int,
        default=14580,
        help="Decode KV transfer port",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request rate in req/s (default: inf = burst)",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor (1.0 = Poisson)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent requests",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=".",
        help="Directory to save results",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Result filename",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or path (defaults to model)",
    )
    parser.add_argument(
        "--nixl-mode",
        action="store_true",
        help="Enable NIXL mode for decode benchmark (injects kv_transfer_params)",
    )


async def main(args: argparse.Namespace) -> dict[str, Any]:
    # Load tokenizer
    tokenizer_name = args.tokenizer or args.model
    tokenizer = get_tokenizer(tokenizer_name, trust_remote_code=True)

    # Load dataset
    input_requests = get_samples(args, tokenizer)
    print(f"Loaded {len(input_requests)} requests from dataset")

    # Run benchmark
    if args.mode == "prefill":
        metrics = await benchmark_prefill(
            model_id=args.model,
            model_name=None,
            tokenizer=tokenizer,
            input_requests=input_requests,
            prefill_url=args.prefill_url,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
        )
    elif args.mode == "decode":
        metrics = await benchmark_decode(
            model_id=args.model,
            model_name=None,
            tokenizer=tokenizer,
            input_requests=input_requests,
            prefill_url=args.prefill_url,
            decode_url=args.decode_url,
            prefill_kv_port=args.prefill_kv_port,
            decode_kv_port=args.decode_kv_port,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
            nixl_mode=args.nixl_mode,
        )
    else:  # e2e
        metrics = await benchmark_e2e(
            model_id=args.model,
            model_name=None,
            tokenizer=tokenizer,
            input_requests=input_requests,
            proxy_url=args.proxy_url,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
        )

    print_metrics(metrics, args.mode)

    # Save results
    result: dict[str, Any] = {
        "mode": args.mode,
        "model": args.model,
        "num_prompts": len(input_requests),
        "request_rate": args.request_rate,
        "completed": metrics.completed,
        "failed": metrics.failed,
        "total_input_tokens": metrics.total_input_tokens,
        "total_output_tokens": metrics.total_output_tokens,
        "request_throughput": metrics.request_throughput,
        "input_token_throughput": metrics.input_token_throughput,
        "output_token_throughput": metrics.output_token_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "ttfts": metrics.ttfts,
        "tpots": metrics.tpots,
    }

    if args.save_result:
        import os

        os.makedirs(args.result_dir, exist_ok=True)
        filename = args.result_filename or f"disagg_{args.mode}_result.json"
        filepath = os.path.join(args.result_dir, filename)
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {filepath}")

    return result


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark vLLM P/D disaggregation prefill/decode performance"
    )
    add_cli_args(parser)
    add_dataset_parser(parser)
    args = parser.parse_args()
    asyncio.run(main(args))
