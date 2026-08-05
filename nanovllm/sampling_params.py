from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1          # -1 disables top-k filtering; >0 keeps the top-k tokens
    top_p: float = 1.0       # 1.0 disables top-p filtering; (0, 1) keeps the smallest set whose cumulative probability reaches top_p
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert self.top_k == -1 or self.top_k > 0, "top_k must be -1 or positive"
        assert 0 < self.top_p <= 1, "top_p must be in (0, 1]"
