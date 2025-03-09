from typing import List
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


class DiversityFilter:
    def __init__(self):
        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.cache = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        if text not in self.cache:
            self.cache[text] = self.model.encode(text)
        return self.cache[text]

    def filter(
        self,
        result: ConcatedLMGenResult,
        semantic_thresh: float = 0.95,
        ngram_thresh: float = 0.95,
    ) -> ConcatedLMGenResult:
        """同步过滤所有相关字段"""
        # 获取原始索引映射
        indices = self._get_filtered_indices(result.text, semantic_thresh, ngram_thresh)

        # 创建过滤后的数据字典
        filtered_data = {}
        for field in fields(result):
            if field.name == "completion_tokens":
                continue  # 该字段会自动重新计算
            # 过滤所有列表字段
            if field.name == "prompt_tokens":
                filtered_data[field.name] = getattr(result, field.name)
            else:
                filtered_data[field.name] = [
                    getattr(result, field.name)[i] for i in indices
                ]

        # 创建新实例并触发post_init
        return ConcatedLMGenResult(**filtered_data)

    def _get_filtered_indices(
        self, texts: List[str], semantic_thresh: float, ngram_thresh: float
    ) -> List[int]:
        """获取需要保留的索引列表"""
        # 第一阶段：N-gram过滤
        candidate_indices = []
        existing_ngrams = []

        for idx in sorted(range(len(texts)), key=lambda x: len(texts[x]), reverse=True):
            text = texts[idx]
            tokens = text.split()
            if len(tokens) < 3:
                current_ngrams = set()
            else:
                current_ngrams = {
                    " ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)
                }

            max_overlap = 0
            for exist_ngrams in existing_ngrams:
                intersection = current_ngrams & exist_ngrams
                overlap = (
                    len(intersection) / len(current_ngrams) if current_ngrams else 0
                )
                max_overlap = max(max_overlap, overlap)
                if max_overlap >= ngram_thresh:
                    break

            if max_overlap < ngram_thresh:
                candidate_indices.append(idx)
                existing_ngrams.append(current_ngrams)

        # 第二阶段：语义过滤
        final_indices = []
        embeddings = [self._get_embedding(texts[i]) for i in candidate_indices]

        for i in range(len(candidate_indices)):
            unique = True
            for j in range(i):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim > semantic_thresh:
                    unique = False
                    break
            if unique:
                final_indices.append(candidate_indices[i])

        return final_indices


if __name__ == "__main__":
    # 原始生成结果
    raw_result = ConcatedLMGenResult(
        # text=[
        #     "To determine which student has the greatest average speed, we need to calculate the average speed for each student. Average speed is given by the formula:\n\n",
        #     "To determine which student has the greatest average speed, we need to calculate the average speed for each student. The average speed is given by the formula:\n\n",
        #     "To determine which student has the greatest average speed, we need to calculate the average speed for each student. The average speed is given by the formula:\n\n",
        #     "To determine which student has the greatest average speed, we need to calculate the average speed for each student using the formula for average speed:\n\n",
        # ],
        text=[
            "To find \\( (a_9)^9 \\), we first need to determine the periodicity of the sequence \\( (a_n) \\). Given \\( a_3 = a_1 \\), we will find the values of \\( a_4, a_5, \\) and so on, to see if the sequence is periodic.\n\n",
            "Given the recurrence relation \\( a_{i+1} = \\frac{1}{1 - a_i} \\), we need to determine the behavior of the sequence. Let's compute the first few terms of the sequence in terms of \\( a_1 \\):\n\n",
            "To solve the problem, we first need to determine the periodicity of the sequence \\( (a_n) \\). We start by expressing \\( a_2 \\) in terms of \\( a_1 \\):\n\\[ a_2 = \\frac{1}{1 - a_1}. \\]\nNext, we express \\( a_3 \\) in terms of \\( a_2 \\):\n\\[ a_3 = \\frac{1}{1 - a_2} = \\frac{1}{1 - \\frac{1}{1 - a_1}} = \\frac{1 - a_1}{1 - a_1 - 1} = \\frac{1 - a_1}{-a_1} = \\frac{a_1 - 1}{a_1} = 1 - \\frac{1}{a_1}. \\]\nSince we know that \\( a_3 = a_1 \\), we can set up the equation:\n\\[ a_1 = 1 - \\frac{1}{a_1}. \\]\nMultiplying both sides by \\( a_1 \\) gives:\n\\[ a_1^2 = a_1 - 1. \\]\nRearranging terms, we get a quadratic equation:\n\\[ a_1^2 - a_1 + 1 = 0. \\]\nThe solutions to this quadratic equation are:\n\\[ a_1 = \\frac{1 \\pm \\sqrt{1 - 4}}{2} = \\frac{1 \\pm \\sqrt{-3}}{2} = \\frac{1 \\pm i\\sqrt{3}}{2}. \\]\nLet's denote these solutions by \\( \\omega = \\frac{1 + i\\sqrt{3}}{2} \\) and \\( \\omega^2 = \\frac{1 - i\\sqrt{3}}{2} \\). These are the non-real cube roots of unity, satisfying \\( \\omega^3 = 1 \\) and \\( (\\omega^2)^3 = 1 \\), and \\( 1 + \\omega + \\omega^2 = 0 \\).\n\n",
            "To solve for \\( (a_9)^9 \\), we first need to understand the periodicity of the sequence. Given that \\( a_3 = a_1 \\), we will investigate the sequence further to find its periodicity.\n\n",
            "To solve for \\( (a_9)^9 \\) in the sequence \\( (a_n) \\) defined by \\( a_{i+1} = \\frac{1}{1 - a_i} \\) and given that \\( a_3 = a_1 \\), we will first determine the periodicity of the sequence. Let's start by expressing the first few terms of the sequence in terms of \\( a_1 \\).\n\n",
        ],
        prompt_tokens=369,
        num_tokens=[29, 30, 30, 27],
        cumulative_logprob=[
            -0.6994021248507494,
            -1.1192965058767683,
            -1.1192965058767683,
            -3.198860161170664,
        ],
        logp_avg_by_len=[
            -0.02411731465002584,
            -0.03730988352922561,
            -0.03730988352922561,
            -0.11847630226558016,
        ],
        finish_reason=["stop", "stop", "stop", "stop"],
    )

    # 过滤处理
    filter = DiversityFilter()
    filtered_result = filter.filter(raw_result)

    # 验证过滤结果
    print(len(filtered_result.text))
    print(filtered_result.text)  # ["方法A", "方法B"]
    print(filtered_result.prompt_tokens)  # [10, 10]
    print(filtered_result.finish_reason)  # ["stop", "length"]

    # [
    #     "Next, we need to find \\(\\theta\\), the angle that the line from the origin to the point makes with the positive \\(x\\)-axis. The formula for \\(\\theta\\) is:\n\\[\n\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\n\\]\nHowever, since \\(x = 0\\), we need to consider the position of the point in the coordinate plane. The point \\((0,3)\\) lies on the positive \\(y\\)-axis, so the angle \\(\\theta\\) is \\(\\frac{\\pi}{2}\\) radians.\n\n",
    #     "Next, we calculate \\(\\theta\\), which is the angle formed with the positive \\(x\\)-axis. The formula for \\(\\theta\\) is:\n\\[\n\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\n\\]\nHowever, we need to consider the quadrant in which the point lies. Since the point \\((0,3)\\) is on the positive \\(y\\)-axis, \\(\\theta\\) is \\(\\frac{\\pi}{2}\\) (or 90 degrees).\n\n",
    # ]
    # [3, 12]
    # ["stop", "stop"]
