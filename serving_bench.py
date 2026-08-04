import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np
from tqdm.auto import tqdm


DEFAULT_MIN_INPUT_LEN = 100


@dataclass
class RequestMetrics:
    request_id: int
    input_len: int
    requested_output_len: int
    arrival_time: float
    enqueue_time: float
    first_token_time: float | None = None
    completion_time: float | None = None
    output_len: int | None = None

    def record_new_tokens(self, timestamp: float, count: int) -> None:
        if self.first_token_time is None:
            self.first_token_time = timestamp
        self.output_len = (self.output_len or 0) + count

    def record_completion(self, timestamp: float) -> None:
        self.completion_time = timestamp

    def to_dict(self, start_time: float) -> dict:
        return {
            "request_id": self.request_id,
            "input_len": self.input_len,
            "requested_output_len": self.requested_output_len,
            "output_len": self.output_len,
            "arrival_time_s": self.arrival_time - start_time,
            "enqueue_time_s": self.enqueue_time - start_time,
            "ttft_s": self.first_token_time - self.arrival_time if self.first_token_time is not None else None,
            "tpot_s": self.tpot,
            "latency_s": self.completion_time - self.arrival_time if self.completion_time is not None else None,
        }

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def tpot(self) -> float | None:
        if self.output_len is None or self.output_len <= 1 or self.first_token_time is None or self.completion_time is None:
            return None
        return (self.completion_time - self.first_token_time) / (self.output_len - 1)

    @property
    def latency(self) -> float | None:
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline and serving benchmark for nano-vllm.")
    parser.add_argument("--model", required=True, help="Path to the local Hugging Face model directory.")
    parser.add_argument("--mode", choices=["offline", "serving"], default="serving", help="Benchmark mode.")
    parser.add_argument("--num-requests", type=int, default=256, help="Number of requests to process.")
    parser.add_argument("--request-rate", type=float, default=8.0, help="Poisson arrival rate in requests per second.")
    parser.add_argument("--prompt-file", type=Path, help="Optional UTF-8 text file with one prompt per line.")
    parser.add_argument("--length-distribution", choices=["fixed", "uniform"], default="uniform")
    parser.add_argument("--min-input-len", type=int, default=DEFAULT_MIN_INPUT_LEN)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=1)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, help="Optional path for aggregate and per-request results.")
    args = parser.parse_args()

    if args.num_requests <= 0:
        parser.error("--num-requests must be positive")
    if args.mode == "serving" and args.request_rate <= 0:
        parser.error("--request-rate must be positive in serving mode")
    if args.min_input_len <= 0 or args.min_input_len > args.max_input_len:
        parser.error("input lengths must satisfy 0 < --min-input-len <= --max-input-len")
    if args.min_output_len <= 0 or args.min_output_len > args.max_output_len:
        parser.error("output lengths must satisfy 0 < --min-output-len <= --max-output-len")
    if args.temperature <= 1e-10:
        parser.error("--temperature must be greater than 1e-10")
    if args.max_model_len <= 0:
        parser.error("--max-model-len must be positive")
    if args.min_input_len + args.min_output_len > args.max_model_len:
        parser.error("minimum input and output lengths exceed --max-model-len")
    if args.max_num_seqs <= 0 or args.max_num_batched_tokens <= 0:
        parser.error("batch limits must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must not be negative")
    if not Path(args.model).expanduser().is_dir():
        parser.error(f"model directory does not exist: {Path(args.model).expanduser()}")
    if args.prompt_file is not None and not args.prompt_file.is_file():
        parser.error(f"prompt file does not exist: {args.prompt_file}")
    return args


def sample_length(random_generator: Random, minimum: int, maximum: int, distribution: str) -> int:
    return maximum if distribution == "fixed" else random_generator.randint(minimum, maximum)


def load_prompts(args: argparse.Namespace, random_generator: Random, vocab_size: int) -> list[str | list[int]]:
    if args.prompt_file is not None:
        prompts = [line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(prompts) < args.num_requests:
            raise ValueError("--prompt-file contains fewer non-empty prompts than --num-requests")
        return prompts[:args.num_requests]
    return [
        [
            random_generator.randrange(vocab_size)
            for _ in range(sample_length(random_generator, args.min_input_len, args.max_input_len, args.length_distribution))
        ]
        for _ in range(args.num_requests)
    ]


def arrival_times(args: argparse.Namespace, random_generator: np.random.Generator) -> np.ndarray:
    if args.mode == "offline":
        return np.zeros(args.num_requests)
    intervals = random_generator.exponential(1.0 / args.request_rate, args.num_requests - 1)
    return np.concatenate(([0.0], np.cumsum(intervals)))


def summarize(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def print_latency_summary(name: str, summary: dict | None, unit: str) -> None:
    if summary is None:
        print(f"{name}: unavailable")
        return
    multiplier = 1000 if unit == "ms" else 1
    suffix = unit
    print(
        f"{name}: mean={summary['mean'] * multiplier:.2f}{suffix}, "
        f"p50={summary['p50'] * multiplier:.2f}{suffix}, "
        f"p95={summary['p95'] * multiplier:.2f}{suffix}, "
        f"p99={summary['p99'] * multiplier:.2f}{suffix}"
    )


def build_results(
    args: argparse.Namespace,
    metrics: dict[int, RequestMetrics],
    start_time: float,
    end_time: float,
    metadata: dict,
) -> dict:
    ordered_metrics = [metrics[request_id] for request_id in sorted(metrics)]
    completed_metrics = [metric for metric in ordered_metrics if metric.completion_time is not None and metric.output_len is not None]
    total_input_tokens = sum(metric.input_len for metric in completed_metrics)
    total_output_tokens = sum(metric.output_len for metric in completed_metrics)
    duration = end_time - start_time
    aggregate = {
        "requests_submitted": len(metrics),
        "requests_completed": len(completed_metrics),
        "total_duration_s": duration,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput_rps": len(completed_metrics) / duration,
        "input_throughput_tps": total_input_tokens / duration,
        "output_throughput_tps": total_output_tokens / duration,
        "ttft_s": summarize([metric.ttft for metric in completed_metrics if metric.ttft is not None]),
        "tpot_s": summarize([metric.tpot for metric in completed_metrics if metric.tpot is not None]),
        "latency_s": summarize([metric.latency for metric in completed_metrics if metric.latency is not None]),
    }
    return {
        "metadata": metadata,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "aggregate": aggregate,
        "requests": [metric.to_dict(start_time) for metric in ordered_metrics],
    }


def print_results(results: dict) -> None:
    aggregate = results["aggregate"]
    throughput_label = "Engine" if results["metadata"]["throughput_kind"] == "engine" else "End-to-end goodput"
    print("--- Benchmark Results ---")
    print(f"Mode: {results['metadata']['mode']}")
    print(f"Measurement: {results['metadata']['measurement_scope']}")
    print(f"Total time: {aggregate['total_duration_s']:.2f}s")
    print(f"Requests: {aggregate['requests_completed']}/{aggregate['requests_submitted']}")
    print(f"Input tokens: {aggregate['total_input_tokens']}")
    print(f"Output tokens: {aggregate['total_output_tokens']}")
    print(f"{throughput_label} request throughput: {aggregate['request_throughput_rps']:.2f} req/s")
    print(f"{throughput_label} input throughput: {aggregate['input_throughput_tps']:.2f} tok/s")
    print(f"{throughput_label} output throughput: {aggregate['output_throughput_tps']:.2f} tok/s")
    print_latency_summary("TTFT", aggregate["ttft_s"], "ms")
    print_latency_summary("TPOT", aggregate["tpot_s"], "ms")
    print_latency_summary("Latency", aggregate["latency_s"], "ms")
    print("-------------------------")


def main() -> None:
    args = parse_args()
    model_path = str(Path(args.model).expanduser().resolve())

    import torch
    from nanovllm import LLM, SamplingParams

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nano-vllm")

    print(f"\n--- Running {args.mode} benchmark with {args.num_requests} requests ---")
    llm = LLM(
        model_path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    engine = llm
    effective_max_model_len = engine.model_runner.config.max_model_len
    if args.max_input_len + args.max_output_len > effective_max_model_len and args.prompt_file is None:
        raise ValueError(
            "The requested maximum input and output lengths exceed the model's effective context length "
            f"({effective_max_model_len} tokens)."
        )

    if args.warmup_requests:
        llm.generate([[0]] * args.warmup_requests, SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=1), use_tqdm=False)

    prompt_rng = Random(args.seed)
    arrival_rng = np.random.default_rng(args.seed)
    prompts = load_prompts(args, prompt_rng, engine.model_runner.config.hf_config.vocab_size)
    sampling_params = [
        SamplingParams(
            temperature=args.temperature,
            ignore_eos=args.ignore_eos,
            max_tokens=sample_length(prompt_rng, args.min_output_len, args.max_output_len, args.length_distribution),
        )
        for _ in range(args.num_requests)
    ]
    scheduled_arrivals = arrival_times(args, arrival_rng)

    metadata = {
        "mode": args.mode,
        "throughput_kind": "engine" if args.mode == "offline" else "end_to_end_goodput",
        "measurement_scope": (
            "model execution after all requests are queued"
            if args.mode == "offline"
            else "theoretical request arrival through completion"
        ),
        "model": model_path,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn": __import__("flash_attn").__version__,
        "effective_max_model_len": effective_max_model_len,
    }
    metrics: dict[int, RequestMetrics] = {}
    requests_sent = 0

    if args.mode == "offline":
        for request_index, (prompt, sampling_param) in enumerate(zip(prompts, sampling_params)):
            input_len = len(engine.tokenizer.encode(prompt)) if isinstance(prompt, str) else len(prompt)
            if input_len + sampling_param.max_tokens > effective_max_model_len:
                raise ValueError(
                    f"request {request_index} needs {input_len + sampling_param.max_tokens} tokens, "
                    f"exceeding the effective context length of {effective_max_model_len}"
                )
            engine.add_request(prompt, sampling_param)
            sequence = engine.scheduler.waiting[-1]
            metrics[sequence.seq_id] = RequestMetrics(
                request_id=sequence.seq_id,
                input_len=input_len,
                requested_output_len=sampling_param.max_tokens,
                arrival_time=0.0,
                enqueue_time=0.0,
            )
        requests_sent = args.num_requests
        start_time = time.perf_counter()
        for metric in metrics.values():
            metric.arrival_time = start_time
            metric.enqueue_time = start_time
    else:
        start_time = time.perf_counter()

    with tqdm(total=args.num_requests, desc="Processing Requests") as progress_bar:
        while requests_sent < args.num_requests or not engine.is_finished():
            current_time = time.perf_counter()
            while (
                args.mode == "serving"
                and requests_sent < args.num_requests
                and current_time - start_time >= scheduled_arrivals[requests_sent]
            ):
                prompt = prompts[requests_sent]
                sampling_param = sampling_params[requests_sent]
                input_len = len(engine.tokenizer.encode(prompt)) if isinstance(prompt, str) else len(prompt)
                if input_len + sampling_param.max_tokens > effective_max_model_len:
                    raise ValueError(
                        f"request {requests_sent} needs {input_len + sampling_param.max_tokens} tokens, "
                        f"exceeding the effective context length of {effective_max_model_len}"
                    )
                enqueue_time = time.perf_counter()
                engine.add_request(prompt, sampling_param)
                sequence = engine.scheduler.waiting[-1]
                metrics[sequence.seq_id] = RequestMetrics(
                    request_id=sequence.seq_id,
                    input_len=input_len,
                    requested_output_len=sampling_param.max_tokens,
                    arrival_time=start_time + scheduled_arrivals[requests_sent],
                    enqueue_time=enqueue_time,
                )
                requests_sent += 1

            if engine.scheduler.waiting or engine.scheduler.running:
                step_outputs, _ = engine.step()
                step_end_time = time.perf_counter()
                for sequence_id, new_token_ids, finished in step_outputs:
                    metric = metrics[sequence_id]
                    metric.record_new_tokens(step_end_time, len(new_token_ids))
                    if finished:
                        metric.record_completion(step_end_time)
                        progress_bar.update(1)
            elif requests_sent < args.num_requests:
                next_arrival = start_time + scheduled_arrivals[requests_sent]
                time.sleep(min(0.01, max(0.0, next_arrival - time.perf_counter())))

    end_time = time.perf_counter()
    results = build_results(args, metrics, start_time, end_time, metadata)
    print_results(results)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results written to: {args.output_json}")


if __name__ == "__main__":
    main()
