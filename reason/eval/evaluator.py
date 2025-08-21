import ray

from typing import Any, Callable, Dict, Optional, List, Tuple, TypeVar
from dataclasses import dataclass
from math_verify import parse, verify

from reason.infer.lm_call import LanguageModelCallingFunction
from reason.infer.rm_call import RewardModelCallingFunction
from utils import DataItem
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

ParsedAnswer = List[Any]


def judge_ans(
    parsed_gold: ParsedAnswer,
    parsed_ans_list: List[ParsedAnswer],
    v_list: List[List[float]],
    aggration_mode: str,
):
    if len(parsed_ans_list) == 0:
        return 0
    aggregated_ans = AGG_FN_MAP[aggration_mode](parsed_ans_list, v_list)

    return int(verify(parsed_gold, aggregated_ans))


@dataclass
class TreeSearchSolutionOutput:
    solutions: List[str]
    completion_tokens: List[int]
    values: List[List[float]]
    accs: List[float]
    scores: List[float]


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
        input_inst: DataItem,
        solver_fn: Callable,
    ) -> List[str]:
        # try:
        solution, tree_step_data = solver_fn(input_inst, self.lm_call, self.rm_call)
        solution: TreeSearchSolutionOutput
        result, output = self.analyze_output(input_inst, solution)

        for i, o in enumerate(output):
            o["completion_tokens"] = solution.completion_tokens[i]
        return input_inst, result, output, tree_step_data
        # except Exception as e:
        #     print(f"Error evaluating problem {input_inst.question}: {e}")
        #     return input_inst, None, None, None

    def analyze_output(
        self,
        input_inst: DataItem,
        solution: TreeSearchSolutionOutput,
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        parsed_gold = parse(input_inst.gold)
        output_list, parsed_ans_list = [], []
        for i, (output, values, acc, score) in enumerate(
            zip(solution.solutions, solution.values, solution.accs, solution.scores)
        ):
            parsed_answer = parse(output)
            parsed_ans_list.append(parsed_answer)
            output_list.append(
                {
                    "path_idx": i,
                    "text": output,
                    "value": values,
                    "acc": acc,
                    "score": score,
                }
            )

        res = {
            # agg_method: judge_ans(parsed_gold, parsed_ans_list, solution.values, agg_method)
            # for agg_method in CHOSEN_AGGR_METHODS
        }
        return res, output_list
