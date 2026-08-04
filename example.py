import os
import sys
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/root/model/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # Each request owns one block; on every delta the cursor returns to the top
    # and both blocks are redrawn in place, like two typewriter-style chat boxes.
    print("--- Streaming Output ---\n")
    for i in range(len(prompts)):
        print(f"[req {i}] \n")
    texts = [""] * len(prompts)
    for index, delta, _ in llm.generate_stream_text(prompts, sampling_params, use_tqdm=False):
        texts[index] += delta
        frame = ["\x1b[H--- Streaming Output ---\x1b[K\n"]   # cursor home, redraw title
        for i, text in enumerate(texts):
            frame.append(f"\n[req {i}] {text}\x1b[K\n")     # \x1b[K clears leftovers on the line
        sys.stdout.write("".join(frame) + "\x1b[J")          # \x1b[J clears anything below
        sys.stdout.flush()


if __name__ == "__main__":
    main()
