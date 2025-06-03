from dataclasses import dataclass
import functools
from typing import Dict

from math_verify import parse, verify

from reason.inference.lm_call import LMCallingConfig, LanguageModelCallingFunction
from reason.inference.rm_call import RewardModelCallingFunction
from reason.evaluation.evaluator import SolutionOutput, Task, TreeSearchSolutionOutput
from reason.guided_search.tree import SearchTree
from reason.guided_search.rstar import RstarSearchTree


@dataclass
class BasicConfig:
    task_name: str


@dataclass
class CoTConfig(BasicConfig):
    pass


def cot(
    config: CoTConfig,
    gen_config: LMCallingConfig,
    task: Task,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> SolutionOutput:
    config = BestOfNConfig(**config.__dict__, num_sequence=1)

    if gen_config.n != 1:
        print("Warning: generation config n is set to 1 for CoT method")
    gen_config.n = config.num_sequence

    return best_of_n(config, gen_config, task, problem_inst, lm_call, rm_call)


@dataclass
class BestOfNConfig(BasicConfig):
    num_sequence: int = 32


def best_of_n(
    config: BestOfNConfig,
    gen_config: LMCallingConfig,
    task: Task,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> SolutionOutput:
    if gen_config.max_new_tokens < 256:
        print("Warning: max_new_tokens is less than 256")

    gen_config.n = config.num_sequence
    prompt = task.prompt_fn(problem_inst["question"])
    output = lm_call(prompt, gen_config)
    completion_tokens = output.num_tokens

    question_answer_pairs = [(problem_inst["question"], txt) for txt in output.text]
    values = rm_call(
        question_answer_pairs=question_answer_pairs,
        lm_step_tag=lm_call.lm_step_tag,
    )

    return SolutionOutput(
        solutions=output.text,
        completion_tokens=completion_tokens,
        values=values,
    )


# # best_of_n with correct filter
# def best_of_n(
#     config: BestOfNConfig,
#     gen_config: LMCallingConfig,
#     task: Task,
#     problem_inst: Dict[str, str],
#     lm_call: LanguageModelCallingFunction,
#     rm_call: RewardModelCallingFunction,
# ) -> SolutionOutput:
#     num_sequence = config.num_sequence
#     beam_search_config = BeamSearchConfig(
#         task_name=config.task_name,
#         tree_max_depth=50,
#         tree_max_width=1,
#         beam_size=1,
#     )
#     def merge_tree_solutions(
#         output1: TreeSearchSolutionOutput,
#         output2: TreeSearchSolutionOutput,
#     ) -> TreeSearchSolutionOutput:
#         return TreeSearchSolutionOutput(
#             tree_completion_tokens=output1.tree_completion_tokens
#             + output2.tree_completion_tokens,
#             solutions=output1.solutions + output2.solutions,
#             completion_tokens=output1.completion_tokens + output2.completion_tokens,
#         )
#     result = beam_search(
#         beam_search_config,
#         gen_config,
#         task,
#         problem_inst,
#         lm_call,
#         rm_call,
#     )
#     for _ in range(num_sequence - 1):
#         temp = beam_search(
#             beam_search_config,
#             gen_config,
#             task,
#             problem_inst,
#             lm_call,
#             rm_call,
#         )
#         result = merge_tree_solutions(result, temp)
#     return result


@dataclass
class TreeSearchConfig(BasicConfig):
    # construction config
    tree_max_width: int = 10
    tree_max_depth: int = 10
    # node config
    init_critic_value: bool = True

    def __post_init__(self):
        assert self.tree_max_width > 0, "Tree width must be greater than 0"
        assert self.tree_max_depth > 0, "Tree depth must be greater than 0"


@dataclass
class BeamSearchConfig(TreeSearchConfig):
    beam_size: int = 1

    def __post_init__(self):
        super().__post_init__()
        assert self.beam_size > 0, "Beam size must be greater than 0"
        assert self.init_critic_value, "BeamSearch should set init_critic_value to True"


def beam_search(
    config: BeamSearchConfig,
    gen_config: LMCallingConfig,
    task: Task,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> TreeSearchSolutionOutput:
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,
            "max_length": config.tree_max_depth,
            "stop_str": "The answer is ",
            "generation_config": {
                "max_new_tokens": gen_config.max_new_tokens,
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,
            },
        },
        math_problems=[
            {
                "question": problem_inst["question"],
                "answer": parse(problem_inst["answer"]),
            }
        ],
        llm_gen_fn=lm_call,
        reward_model_fn=rm_call_fn,
        # TODO(ziyu): set sep by lm_call.lm_step_tag
    )

    search_tree = SearchTree(cfg={})
    traj_list = search_tree.beam_search(
        simulate_env=env,
        beam_size=config.beam_size,
        max_step=config.tree_max_depth,
        reward_model_fn=rm_call_fn,
    )
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
        values=[t["values"] for t in traj_list],
    )


@dataclass
class MCTSBaseConfig(TreeSearchConfig):
    # PUCT hparams
    pb_c_base: float = 19652
    pb_c_init: float = 1.25


@dataclass
class VanilaMCTSConfig(MCTSBaseConfig):
    # rollout step strategy, if `select_by_prior` is False,
    #  then select by the initial critic value
    # otherwise, random choice by the prior probability
    select_by_prior: bool = False
    num_path: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not self.select_by_prior:
            assert (
                self.init_critic_value
            ), "VanilaMCTS with greedy as rollout method should set init_critic_value to True"
        assert self.num_path > 0


def vanila_mcts(
    config: VanilaMCTSConfig,
    gen_config: LMCallingConfig,
    task: Task,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> TreeSearchSolutionOutput:
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,
            "max_length": config.tree_max_depth,
            "stop_str": "The answer is ",
            "generation_config": {
                "max_new_tokens": gen_config.max_new_tokens,
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,
            },
        },
        math_problems=[
            {
                "question": problem_inst["question"],
                "answer": task.extract_groundtruth(problem_inst["answer"]),
            }
        ],
        llm_gen_fn=lm_call,
        reward_model_fn=rm_call_fn,
    )

    search_tree = SearchTree(
        cfg={
            "pb_c_base": config.pb_c_base,
            "pb_c_init": config.pb_c_init,
            "init_critic_value": config.init_critic_value,
        }
    )

    traj_list = search_tree.vanila_mcts(
        simulate_env=env,
        num_path=config.num_path,
        reward_model_fn=rm_call_fn,
        select_by_prior=config.select_by_prior,
    )
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
        values=[t["values"] for t in traj_list],
    )


@dataclass
class MCTSConfig(MCTSBaseConfig):
    # rollout step strategy, if `select_by_prior` is False,
    #  then select by the initial critic value
    # otherwise, random choice by the prior probability
    select_by_prior: bool = False
    num_path: int = 1
    simulate_num: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not self.select_by_prior:
            assert (
                self.init_critic_value
            ), "MCTS with greedy as rollout method should set init_critic_value to True"
        assert self.num_path > 0


def mcts(
    config: MCTSConfig,
    gen_config: LMCallingConfig,
    task: Task,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> TreeSearchSolutionOutput:
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,
            "max_length": config.tree_max_depth,
            "stop_str": "The answer is ",
            "generation_config": {
                "max_new_tokens": gen_config.max_new_tokens,
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,
            },
        },
        math_problems=[
            {
                "question": problem_inst["question"],
                "answer": task.extract_groundtruth(problem_inst["answer"]),
            }
        ],
        llm_gen_fn=lm_call,
        reward_model_fn=rm_call_fn,
    )

    search_tree = SearchTree(
        cfg={
            "pb_c_base": config.pb_c_base,
            "pb_c_init": config.pb_c_init,
            "init_critic_value": config.init_critic_value,
        }
    )

    traj_list = search_tree.mcts(
        simulate_env=env,
        num_path=config.num_path,
        reward_model_fn=rm_call_fn,
        select_by_prior=config.select_by_prior,
        simulate_num=config.simulate_num,
    )
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
        values=[t["values"] for t in traj_list],
    )


@dataclass
class RStarMCTSConfig(MCTSBaseConfig):
    # rollout step strategy, if `select_by_prior` is False,
    #  then select by the initial critic value
    # otherwise, random choice by the prior probability
    select_by_prior: bool = False
    num_path: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not self.select_by_prior:
            assert (
                self.init_critic_value
            ), "VanilaMCTS with greedy as rollout method should set init_critic_value to True"
        assert self.num_path > 0


def rstar_mcts(
    config: RStarMCTSConfig,
    gen_config: LMCallingConfig,
    task: Task,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> TreeSearchSolutionOutput:
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,
            "max_length": config.tree_max_depth,
            "stop_str": "The answer is ",
            "generation_config": {
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,  # this is fixed for each llm call
            },
        },
        math_problems=[
            {
                "question": problem_inst["question"],
                "answer": task.extract_groundtruth(problem_inst["answer"]),
            }
        ],
        llm_gen_fn=lm_call,
        reward_model_fn=rm_call_fn,
    )

    search_tree = RstarSearchTree(
        cfg={
            "pb_c_base": config.pb_c_base,
            "pb_c_init": config.pb_c_init,
            "init_critic_value": config.init_critic_value,
        }
    )

    traj_list = search_tree.rstar_mcts(
        simulate_env=env,
        num_path=config.num_path,
        reward_model_fn=rm_call_fn,
        select_by_prior=config.select_by_prior,
    )
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
        values=[t["values"] for t in traj_list],
    )


@dataclass
class MCTSBeamSearchConfig(MCTSBaseConfig):
    # rollout step strategy, if `select_by_prior` is False,
    #  then select by the initial critic value
    # otherwise, random choice by the prior probability
    num_path: int = 1
    select_by_prior: bool = False
    simulate_num: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not self.select_by_prior:
            assert (
                self.init_critic_value
            ), "MCTSBeamSearch method should set init_critic_value to True"
        assert self.num_path > 0
        assert (
            self.init_critic_value
        ), "MCTSBeamSearch should set init_critic_value to True"


def mcts_beam_search(
    config: MCTSBeamSearchConfig,
    gen_config: LMCallingConfig,
    task: Task,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> TreeSearchSolutionOutput:
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,
            "max_length": config.tree_max_depth,
            "stop_str": "The answer is ",
            "generation_config": {
                "max_new_tokens": gen_config.max_new_tokens,
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,
            },
        },
        math_problems=[
            {
                "question": problem_inst["question"],
                "answer": task.extract_groundtruth(problem_inst["answer"]),
            }
        ],
        llm_gen_fn=lm_call,
        reward_model_fn=rm_call_fn,
    )

    search_tree = SearchTree(
        cfg={
            "pb_c_base": config.pb_c_base,
            "pb_c_init": config.pb_c_init,
            "init_critic_value": config.init_critic_value,
        }
    )

    traj_list = search_tree.mcts_beam_search(
        initial_env=env,
        num_path=config.num_path,
        # max_step=config.tree_max_depth,
        reward_model_fn=rm_call_fn,
        select_by_prior=config.select_by_prior,
        simulate_num=config.simulate_num,
    )
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
        values=[t["values"] for t in traj_list],
    )
