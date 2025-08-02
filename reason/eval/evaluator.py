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
        input_inst: DataItem,
        solver_fn: Callable,
    ) -> List[str]:
        try:
            solution, tree_step_data = solver_fn(input_inst, self.lm_call, self.rm_call)
            solution: SolutionOutput

            result, output = self.analyze_output(input_inst, solution.solutions, solution.values)

            total_completion_token = 0
            for i, o in enumerate(output):
                o["completion_tokens"] = solution.completion_tokens[i]
                o["tree_completion_tokens"] = solution.tree_completion_tokens[i]
                # We define the completion_tokens as the tokens comsumed between two generated
                #  answers, therefore we need to take sum here.
                total_completion_token += solution.completion_tokens[i]
            result["total_completion_tokens"] = total_completion_token
            return input_inst, result, output, tree_step_data
        except Exception as e:
            print(f"Error evaluating problem {input_inst.question}: {e}")
            return input_inst, None, None, None

    def analyze_output(
        self,
        input_inst: DataItem,
        gen_answers: List[str],
        values_list: List[List[float]],
    ) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        parsed_gold = parse(input_inst.gold)
        output_list, parsed_ans_list = [], []
        for i, (output, values) in enumerate(zip(gen_answers, values_list)):
            parsed_answer = parse(output)
            parsed_ans_list.append(parsed_answer)
            output_list.append(
                {
                    "path_idx": i,
                    "text": output,
                    "value": values,
                    "acc": int(verify(parsed_gold, parsed_answer)),
                }
            )

        res = {
            agg_method: judge_ans(parsed_gold, parsed_ans_list, values_list, agg_method)
            for agg_method in CHOSEN_AGGR_METHODS
        }
        return res, output_list
