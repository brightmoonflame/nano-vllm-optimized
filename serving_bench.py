import os
import time
import numpy as np
import argparse
from random import Random
from tqdm.auto import tqdm

MIN_INPUT_LEN = 100

class RequestMetrics:
    """Stores metrics for a single request."""
    def __init__(self, request_id, input_len, max_output_len):
        self.request_id = request_id
        self.input_len = input_len
        self.max_output_len = max_output_len
        self.submission_time = -1
        self.first_token_time = -1
        self.completion_time = -1
        self.output_len = -1

    def record_submission(self):
        self.submission_time = time.perf_counter()

    def record_first_token(self):
        if self.first_token_time == -1:
            self.first_token_time = time.perf_counter()

    def record_completion(self, output_ids):
        self.completion_time = time.perf_counter()
        self.output_len = len(output_ids)

    @property
    def ttft(self):
        return self.first_token_time - self.submission_time

    @property
    def tpot(self):
        if self.output_len > 1:
            return (self.completion_time - self.first_token_time) / (self.output_len - 1)
        return float('nan')

    @property
    def latency(self):
        return self.completion_time - self.submission_time

def main():
    """Main function to run the serving benchmark."""
    parser = argparse.ArgumentParser(description="Serving benchmark for nano-vllm.")
    parser.add_argument("--model", required=True, help="Path to the local Hugging Face model directory.")
    parser.add_argument("--num-requests", type=int, default=256, help="Number of requests to process.")
    parser.add_argument("--request-rate", type=float, default=8.0, help="Request rate (requests per second).")
    parser.add_argument("--max-input-len", type=int, default=1024, help="Maximum prompt length in tokens.")
    parser.add_argument("--max-output-len", type=int, default=1024, help="Maximum generated length in tokens.")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model context length in tokens.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for workload generation.")
    args = parser.parse_args()

    if args.num_requests <= 0:
        parser.error("--num-requests must be positive")
    if args.request_rate <= 0:
        parser.error("--request-rate must be positive")
    if args.max_input_len < MIN_INPUT_LEN:
        parser.error(f"--max-input-len must be at least {MIN_INPUT_LEN}")
    if args.max_output_len < MIN_INPUT_LEN:
        parser.error(f"--max-output-len must be at least {MIN_INPUT_LEN}")
    if args.max_model_len <= 0:
        parser.error("--max-model-len must be positive")
    if args.max_input_len + args.max_output_len > args.max_model_len:
        parser.error("--max-input-len plus --max-output-len must not exceed --max-model-len")

    model_path = os.path.abspath(os.path.expanduser(args.model))
    if not os.path.isdir(model_path):
        parser.error(f"model directory does not exist: {model_path}")

    from nanovllm import LLM, SamplingParams

    print(f"\n--- Running benchmark with --num-requests {args.num_requests} --request-rate {args.request_rate} ---")
    llm = LLM(model_path, enforce_eager=False, max_model_len=args.max_model_len)
    engine = llm
    effective_max_model_len = engine.model_runner.config.max_model_len
    if args.max_input_len + args.max_output_len > effective_max_model_len:
        raise ValueError(
            "The requested input and output limits exceed the model's effective context length "
            f"({effective_max_model_len} tokens)."
        )

    prompt_rng = Random(args.seed)
    arrival_rng = np.random.default_rng(args.seed)
    vocab_size = engine.model_runner.config.hf_config.vocab_size

    # --- Generate random prompts ---
    prompts = [
        [prompt_rng.randrange(vocab_size) for _ in range(prompt_rng.randint(MIN_INPUT_LEN, args.max_input_len))]
        for _ in range(args.num_requests)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=prompt_rng.randint(MIN_INPUT_LEN, args.max_output_len),
        )
        for _ in range(args.num_requests)
    ]

    # --- Generate request arrival times ---
    request_intervals = arrival_rng.exponential(1.0 / args.request_rate, args.num_requests)
    arrival_times = np.cumsum(request_intervals)

    # --- Benchmark loop ---
    metrics = {}
    requests_sent = 0
    start_time = time.perf_counter()
    completed_latencies = []

    with tqdm(total=args.num_requests, desc="Processing Requests") as pbar:
        while requests_sent < args.num_requests or not engine.is_finished():
            # --- Send new requests ---
            current_time = time.perf_counter()
            while requests_sent < args.num_requests and current_time - start_time >= arrival_times[requests_sent]:
                prompt = prompts[requests_sent]
                sp = sampling_params[requests_sent]
                
                engine.add_request(prompt, sp)
                
                new_seq = engine.scheduler.waiting[-1]
                seq_id = new_seq.seq_id
                req_metrics = RequestMetrics(seq_id, len(prompt), sp.max_tokens)
                req_metrics.submission_time = start_time + arrival_times[requests_sent]
                metrics[seq_id] = req_metrics
                
                requests_sent += 1

            # --- Engine step ---
            if engine.scheduler.waiting or engine.scheduler.running:
                finished_outputs, _ = engine.step()

                # Record first token time for all processed sequences
                all_processed_seqs = list(engine.scheduler.running)
                for seq in all_processed_seqs:
                    if seq.seq_id in metrics:
                        metrics[seq.seq_id].record_first_token()

                for seq_id, output_ids in finished_outputs:
                    if seq_id in metrics:
                        metrics[seq_id].record_first_token() # Ensure first token time is recorded
                        metrics[seq_id].record_completion(output_ids)
                        
                        completed_latencies.append(metrics[seq_id].latency)
                        avg_latency = np.mean(completed_latencies)
                        pbar.set_postfix({"Avg Latency": f"{avg_latency:.2f}s"})
                        pbar.update(1)
            else:
                # If no requests are running or waiting, sleep briefly
                time.sleep(0.01)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # --- Calculate and print metrics ---
    total_input_tokens = sum(m.input_len for m in metrics.values())
    total_output_tokens = sum(m.output_len for m in metrics.values() if m.output_len != -1)
    
    avg_ttft = np.mean([m.ttft for m in metrics.values() if m.first_token_time != -1])
    avg_tpot = np.mean([m.tpot for m in metrics.values() if not np.isnan(m.tpot)])
    avg_latency = np.mean([m.latency for m in metrics.values() if m.completion_time != -1])
    throughput = total_output_tokens / total_time

    print("--- Benchmark Results ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests sent: {requests_sent}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Average TTFT: {avg_ttft * 1000:.2f} ms")
    print(f"Average TPOT: {avg_tpot * 1000:.2f} ms/token")
    print(f"Average latency: {avg_latency:.2f} s")
    print("-------------------------\n")

if __name__ == "__main__":
    main()
