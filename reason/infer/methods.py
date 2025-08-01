import functools

from typing import Dict, Optional
from dataclasses import dataclass
from jsonlines import Writer

from utils import DataItem
from reason.env import Env
from reason.eval.evaluator import TreeSearchSolutionOutput
from reason.infer.tree import SearchTree
from reason.infer.rm_call import RewardModelCallingFunction
from reason.infer.lm_call import LMCallingConfig, LanguageModelCallingFunction


@dataclass
class TreeSearchConfig:
    # construction config
    max_width: int = 10
    max_steps: int = 10

    def __post_init__(self):
        assert self.max_width > 0, "Tree width must be greater than 0"
        assert self.max_steps > 0, "Tree depth must be greater than 0"


@dataclass
class BeamSearchConfig(TreeSearchConfig):
    beam_size: int = 1

    def __post_init__(self):
        super().__post_init__()
        assert self.beam_size > 0, "Beam size must be greater than 0"


def beam_search(
    config: BeamSearchConfig,
    gen_config: LMCallingConfig,
    input_inst: DataItem,
    lm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> TreeSearchSolutionOutput:
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    env = Env(
        config={
            "max_width": config.max_width,
            "max_steps": config.max_steps,
            "stop_str": gen_config.stop_str,
            "generation_config": {
                "max_new_tokens": gen_config.max_new_tokens,
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,
            },
        },
        input_inst=input_inst,
        lm_call=lm_call,
        rm_call=rm_call_fn,
    )

    search_tree = SearchTree()

    traj_list, tree_step_data = search_tree.beam_search(
        simulate_env=env,
        beam_size=config.beam_size,
        max_step=config.max_steps,
        rm_call=rm_call_fn,
    )
    solution = TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
        values=[t["values"] for t in traj_list],
    )
    return solution, tree_step_data


@dataclass
class VanillaMCTSConfig(TreeSearchConfig):
    num_path: int = 1
    # PUCT hparams
    pb_c_base: float = 19652
    pb_c_init: float = 1.25

    def __post_init__(self):
        super().__post_init__()
        assert self.num_path > 0


def vanilla_mcts(
    config: VanillaMCTSConfig,
    gen_config: LMCallingConfig,
    input_inst: DataItem,
    lm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> TreeSearchSolutionOutput:
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    env = Env(
        config={
            "max_width": config.max_width,
            "max_steps": config.max_steps,
            "stop_str": gen_config.stop_str,
            "generation_config": {
                "max_new_tokens": gen_config.max_new_tokens,
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,
            },
        },
        input_inst=input_inst,
        lm_call=lm_call,
        rm_call=rm_call_fn,
    )

    search_tree = SearchTree(
        pb_c_base=config.pb_c_base,
        pb_c_init=config.pb_c_init,
    )

    traj_list, tree_step_data = search_tree.vanilla_mcts(
        simulate_env=env,
        num_path=config.num_path,
        rm_call=rm_call_fn,
    )

    solution = TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
        values=[t["values"] for t in traj_list],
    )
    return solution, tree_step_data
