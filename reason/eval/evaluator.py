import ray
import numpy as np

from typing import Any, Callable, Dict, Optional, List, Tuple
from dataclasses import dataclass
from math_verify import parse, verify

from reason.infer.lm_call import LanguageModelCallingFunction
from reason.infer.rm_call import RewardModelCallingFunction
from utils.answer_vote import (
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_VOTE,
    PRM_LAST_MAX,
    AGG_FN_MAP,
)


CHOSEN_AGGR_METHODS = [
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_MAX,
    PRM_LAST_VOTE,
]


def judge_ans(
    extracted_groundtruth: str,
    ans_list: List[str],
    v_list: List[float],
    aggration_mode: str,
    normalize=False,
):
    if len(ans_list) == 0:
        return 0

    if "orm" in aggration_mode and normalize:
        # score_normalization: this is only necessary for [-1, 1] values
        v_list = np.array(v_list)
        v_list -= v_list.min()
        v_list /= v_list.max() + 1e-3
        v_list = v_list.tolist()
    aggregated_ans = AGG_FN_MAP[aggration_mode](ans_list, v_list)

    return 1 if verify(extracted_groundtruth, aggregated_ans) else 0


@dataclass
class SolutionOutput:
    solutions: List[str]
    completion_tokens: List[int]
    values: Optional[List[float]]


@dataclass
class TreeSearchSolutionOutput(SolutionOutput):
    tree_completion_tokens: List[int]


@ray.remote
class MathEvaluator:
    def __init__(
        self,
        lm_call: LanguageModelCallingFunction,
        rm_call: RewardModelCallingFunction,
    ):
        self.lm_call = lm_call
        self.rm_call = rm_call

    def evaluate_problem(
        self,
        problem_inst: Dict[str, str],
        solver_fn: Callable,
    ) -> List[str]:
        solution: SolutionOutput = solver_fn(problem_inst, self.lm_call, self.rm_call)

        result, output = self.analyze_output(problem_inst, solution.solutions, solution.values)

        total_completion_token = 0
        for i, o in enumerate(output):
            o["completion_tokens"] = solution.completion_tokens[i]
            if isinstance(solution, TreeSearchSolutionOutput):
                o["tree_completion_tokens"] = solution.tree_completion_tokens[i]

            # We define the completion_tokens as the tokens comsumed between two generated
            #  answers, therefore we need to take sum here.
            total_completion_token += solution.completion_tokens[i]
        result["total_completion_tokens"] = total_completion_token
        return problem_inst, result, output

    def analyze_output(
        self,
        problem_inst: Dict[str, str],
        gen_answers: List[str],
        values_list: List[List[float]],
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        parsed_groundtruth = parse(problem_inst["answer"])

        output_list = [
            {
                "path_idx": i,
                "text": answer,
                "value": values,
            }
            for i, (answer, values) in enumerate(zip(gen_answers, values_list))
        ]
        parsed_ans_list = [parse(txt) for txt in gen_answers]

        res = {
            agg_method: judge_ans(
                parsed_groundtruth,
                parsed_ans_list,
                values_list,
                agg_method,
            )
            for agg_method in CHOSEN_AGGR_METHODS
        }
        return res, output_list
