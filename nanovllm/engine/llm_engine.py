import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # Collect all EOS token ids: hf_config.eos_token_id (may be int or list) plus
        # tokenizer.eos_token_id. They can differ — e.g. Gemma 3 uses <eos> (id=1) in
        # config but <end_of_turn> (id=106) as the tokenizer's eos_token.
        eos_ids = set()
        hf_eos = getattr(config.hf_config, "eos_token_id", None)
        if isinstance(hf_eos, (list, tuple)):
            eos_ids.update(hf_eos)
        elif isinstance(hf_eos, int):
            eos_ids.add(hf_eos)
        eos_ids.add(self.tokenizer.eos_token_id)
        config.eos = eos_ids
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, 'model_runner'):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def _drop_spec_state(self, seqs: list[Sequence]):
        """Free draft-side state for seqs that finished or were preempted
        this step (spec decode only).

        Finished seqs are done for good; preempted seqs re-prefill from
        scratch and would otherwise extend on top of stale draft KV.
        """
        ids = [seq.seq_id for seq in seqs if seq.is_finished]
        if self.scheduler.preempted_seq_ids:
            ids += self.scheduler.preempted_seq_ids
            self.scheduler.preempted_seq_ids.clear()
        if ids:
            self.model_runner.call("drop_proposer_state", ids)

    def step(self):
        if self.config.enable_chunked_prefill:
            seqs, seqlen_this_time = self.scheduler.schedule_chunked()
            has_prefill = any(seq.is_prefill for seq in seqs)
            prev_counts = [seq.num_completion_tokens for seq in seqs]
            if has_prefill:
                # Mixed batch: prefill chunk + decode, walk varlen path.
                num_prefill = sum(seqlen_this_time[seq.seq_id] for seq in seqs if seq.is_prefill)
                token_ids = self.model_runner.call("run_chunked", seqs, seqlen_this_time)
                self.scheduler.postprocess_chunked(seqs, token_ids)
                num_tokens = num_prefill
            else:
                # Pure decode: spec decode if enabled, else CUDA graph path.
                token_ids = self.model_runner.call("run", seqs, False)
                if self.config.speculative_config is not None:
                    self.scheduler.postprocess_spec(seqs, token_ids)
                    num_tokens = -sum(seq.num_completion_tokens - prev for seq, prev in zip(seqs, prev_counts))
                else:
                    self.scheduler.postprocess(seqs, token_ids, False)
                    num_tokens = -len(seqs)
            if self.config.speculative_config is not None:
                self._drop_spec_state(seqs)
            outputs = [
                (seq.seq_id, seq.completion_token_ids[prev:], seq.is_finished)
                for seq, prev in zip(seqs, prev_counts)
                if seq.num_completion_tokens > prev
            ]
            return outputs, num_tokens

        seqs, is_prefill = self.scheduler.schedule()
        use_spec = not is_prefill and self.config.speculative_config is not None
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        prev_counts = [seq.num_completion_tokens for seq in seqs]
        if use_spec:
            self.scheduler.postprocess_spec(seqs, token_ids)
            num_tokens = -sum(seq.num_completion_tokens - prev for seq, prev in zip(seqs, prev_counts))
        else:
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        if self.config.speculative_config is not None:
            self._drop_spec_state(seqs)
        outputs = [
            (seq.seq_id, seq.completion_token_ids[prev:], seq.is_finished)
            for seq, prev in zip(seqs, prev_counts)
            if seq.num_completion_tokens > prev
        ]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate_stream(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ):
        """Yield (prompt_index, new_token_ids, is_finished) incrementally after each step."""
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        index_of = {}
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            seq = self.add_request(prompt, sp)
            index_of[seq.seq_id] = i
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            outputs, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, new_token_ids, finished in outputs:
                if finished:
                    pbar.update(1)
                yield index_of[seq_id], new_token_ids, finished
        pbar.close()

    def generate_stream_text(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ):
        """Yield (prompt_index, text_delta, is_finished) with incremental detokenization.

        Text is decoded from the accumulated token ids so that BPE pieces
        spanning multiple tokens render correctly; a trailing half-formed
        character is held back until the remaining tokens arrive.
        """
        buffers, emitted = {}, {}
        for index, new_token_ids, finished in self.generate_stream(prompts, sampling_params, use_tqdm):
            buf = buffers.setdefault(index, [])
            buf.extend(new_token_ids)
            text = self.tokenizer.decode(buf)
            start = emitted.get(index, 0)
            delta = text[start:]
            if not finished and delta.endswith("\ufffd"):
                delta = delta[:-1]
            emitted[index] = start + len(delta)
            yield index, delta, finished

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        collected = {}
        for index, new_token_ids, _ in self.generate_stream(prompts, sampling_params, use_tqdm):
            collected.setdefault(index, []).extend(new_token_ids)
        outputs = [collected[i] for i in sorted(collected)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
