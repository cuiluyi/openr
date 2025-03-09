from typing import List, Optional, Callable, Type
from dataclasses import dataclass, fields
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ConcatedLMGenResult:
    text: List[str]
    prompt_tokens: List[int]
    num_tokens: List[int]
    cumulative_logprob: List[float]
    logp_avg_by_len: List[float]
    finish_reason: List[str]

    def __post_init__(self):
        self.completion_tokens = sum(self.num_tokens)


from typing import Optional, List, Tuple


class CorrectFilter:
    def __init__(
        self,
        env,
        reward_model_fn: Optional[Callable],
    ):
        self.env = env
        self.reward_model_fn = reward_model_fn
        self.max_step: Optional[ConcatedLMGenResult] = None
        self.max_prm_value: float = 0.0

    @staticmethod
    def _filter_fields(source: ConcatedLMGenResult, target_indices: List[int]) -> dict:
        data = {}
        for field in fields(source):
            if field.name == "completion_tokens":
                continue
            value = getattr(source, field.name)
            data[field.name] = (
                value
                if field.name == "prompt_tokens"
                else [value[i] for i in target_indices]
            )
        return data

    def filter(
        self,
        result: ConcatedLMGenResult,
        thresh: float = 0.8,
    ) -> Optional[ConcatedLMGenResult]:
        """过滤生成结果，保留高于阈值的条目，并更新历史最佳记录"""
        indices, max_index, max_prm_value = self._get_filtered_indices(
            result.text, thresh
        )

        max_index_data = self._filter_fields(result, [max_index])
        self.update_max_step(ConcatedLMGenResult(**max_index_data), max_prm_value)

        if len(indices) == 0:
            return None

        filtered_data = self._filter_fields(result, indices)
        return ConcatedLMGenResult(**filtered_data)

    def _get_filtered_indices(
        self,
        steps: List[str],
        thresh: float,
    ) -> Tuple[List[int], int, float]:
        if not steps:
            return [], 0, 0.0

        prms: List[List[float]] = self.reward_model_fn(
            [(self.env.question, self.env.answer + step) for step in steps]
        )

        prms: List[float] = [prm[-1] for prm in prms]
        final_indices = [i for i, prm in enumerate(prms) if prm >= thresh]
        max_prm = max(prms)
        max_index = prms.index(max_prm)
        return final_indices, max_index, max_prm

    def update_max_step(
        self,
        result: ConcatedLMGenResult,
        prm_value: float,
    ) -> None:
        if self.max_step is None or prm_value > self.max_prm_value:
            self.max_step = result
            self.max_prm_value = prm_value


if __name__ == "__main__":
    # 原始生成结果
    gen_result = ConcatedLMGenResult(
        text=[
            "Now, we need to find the area of triangle \\(Submit\\). Since \\(MN \\parallel BC\\), triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle \\( complete triangle"
        ],
        prompt_tokens=497,
        num_tokens=[2048],
        cumulative_logprob=[-21.730792524293065],
        logp_avg_by_len=[-0.010610738537252473],
        finish_reason=["length"],
    )
