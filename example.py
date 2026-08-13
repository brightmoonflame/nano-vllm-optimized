import os
import shutil
import sys
import unicodedata
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Special tokens appended by the chat template / produced at end-of-turn.
# Stripping them only affects this demo's display, not engine output.
SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>")


def strip_special(text: str) -> str:
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


def display_width(text: str) -> int:
    """Terminal column width of a line (CJK chars and most emoji take 2 columns)."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def main():
    target_path = os.path.expanduser("/root/model/Qwen3-1.7B/")
    draft_path = os.path.expanduser("/root/model/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(target_path)

    # Set draft_path=None to disable speculative decoding.
    speculative_config = {"model": draft_path, "num_spec_tokens": 5} if draft_path else None
    llm = LLM(target_path, enforce_eager=True, tensor_parallel_size=1,
              speculative_config=speculative_config)

    # temperature=0 (greedy) so spec output is identical to non-spec output.
    sampling_params = SamplingParams(temperature=0, max_tokens=256)
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

    # Each request owns one block that grows in place, like a chat box.
    # The cursor moves up relative to the previous frame height (\x1b[nA),
    # never absolute (\x1b[H), so scrollback history above stays untouched.
    print("--- Streaming Output ---")
    texts = [""] * len(prompts)
    frame_lines = 0  # physical terminal lines occupied by the previous frame

    for index, delta, _ in llm.generate_stream_text(prompts, sampling_params, use_tqdm=False):
        texts[index] = strip_special(texts[index] + delta)   # strip on full text, handles split tokens
        width = shutil.get_terminal_size().columns
        lines = []
        for i, text in enumerate(texts):
            lines.extend(f"[req {i}] {text}".expandtabs().split("\n"))  # embedded \n and \t count as lines/columns
            lines.append("")
        sys.stdout.write("\r" + (f"\x1b[{frame_lines}A" if frame_lines else ""))
        frame_lines = 0
        for line in lines:
            sys.stdout.write(line + "\x1b[K\n")
            frame_lines += max(1, -(-display_width(line) // width))  # wrapped lines
        sys.stdout.write("\x1b[J")  # clear leftovers below if this frame is shorter
        sys.stdout.flush()
    print("------------------------")


if __name__ == "__main__":
    main()
