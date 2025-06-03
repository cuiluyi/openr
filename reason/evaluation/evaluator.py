import importlib
from datetime import datetime
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Callable, Dict, Optional, List, Union, Tuple


import numpy as np
import ray
from math_verify import parse, verify


from envs import get_default_query_str_builder, get_env_datasets
from envs.base_env import INVALID_ANS
from reason.inference.lm_call import LanguageModelCallingFunction
from reason.inference.rm_call import RewardModelCallingFunction
from reason.reranking.vote_utils import (
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_VOTE,
    PRM_LAST_MAX,
    AGG_FN_MAP,
)


class Task:
    def __init__(self, task_name: str, dataset_id: str, is_few_shot: bool = False):
        SUPPORTED_TASKS = ["MATH", "rstar", "Online"]
        if task_name not in SUPPORTED_TASKS:
            raise NotImplementedError(f"Task {task_name} is not supported")

        self.task_name = "MATH" if task_name == "Online" else task_name
        task_module = importlib.import_module(f"envs.{self.task_name}")
        self.extract_answer = task_module.extract_answer
        self.extract_groundtruth = task_module.extract_groundtruth
        self.judge_correct = task_module.judge_correct

        self._is_few_shot = is_few_shot
        self.env_fn = task_module.Env

        if task_name != "Online":
            self.dataset_id = dataset_id
            self._train_ds, self._test_ds = get_env_datasets(
                self.task_name, self.dataset_id
            )

    def prompt_fn(self, problem_input: str):
        return get_default_query_str_builder(self.task_name)(
            problem_input,
            is_few_shot=self._is_few_shot,
        )

    @property
    def test_ds(self):
        return self._test_ds

    @property
    def train_ds(self):
        return self._train_ds


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
    valid_ans_list, valid_v_list = [], []
    for i, ans in enumerate(ans_list):
        if ans != INVALID_ANS:
            valid_ans_list.append(ans)
            valid_v_list.append(v_list[i])
    if len(valid_ans_list) == 0:
        return 0

    if "orm" in aggration_mode and normalize:
        # score_normalization: this is only necessary for [-1, 1] values
        valid_v_list = np.array(valid_v_list)
        valid_v_list -= valid_v_list.min()
        valid_v_list /= valid_v_list.max() + 1e-3
        valid_v_list = valid_v_list.tolist()
    aggregated_ans = AGG_FN_MAP[aggration_mode](valid_ans_list, valid_v_list)

    return 1 if verify(extracted_groundtruth, aggregated_ans) else 0


@dataclass
class SolutionOutput:
    solutions: List[str]
    # Define the completion tokens for each solution
    #  For best_of_n, it's a list of int, indicate how many tokens in each generation
    #  for beam search, it's a list of zeros, except the last element indicates total tokens
    #  for mcts, it's a list of int, indicate how many tokens comsumed between two paths
    completion_tokens: List[int]
    values: Optional[List[float]]


@dataclass
class TreeSearchSolutionOutput(SolutionOutput):
    tree_completion_tokens: List[int]


class MathEvaluator:
    def __init__(
        self,
        task: Task,
        lm_call: LanguageModelCallingFunction,
        rm_call: RewardModelCallingFunction,
    ):
        self._task = task
        self.lm_call = lm_call
        self.rm_call = rm_call

    def evaluate_problem(
        self,
        problem_inst: Dict[str, str],
        solver_fn: Callable,
    ) -> List[str]:
        # try:
        solution: SolutionOutput = solver_fn(
            problem_inst,
            self.lm_call,
            self.rm_call,
        )
            
        result, output = self.analyze_output(
            problem_inst,
            solution.solutions,
            solution.values,
        )
        
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
        # except Exception as e:
        #     return problem_inst, {"error": str(e)}, []

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


@ray.remote
class RemoteMathEvaluator(MathEvaluator):
    def __init__(
        self,
        task: str,
        lm_call: LanguageModelCallingFunction,
        rm_call: RewardModelCallingFunction,
    ):
        super().__init__(task, lm_call, rm_call)
