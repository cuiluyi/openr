from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from config.config_utils import str2bool
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller
from reason.inference.rm_call import (
    RMRemoteCaller,
    DummyRewardModelCaller,
    RewardModelBaseConfig,
    RemoteRewardModelConfig,
)
from reason.evaluation.evaluator import SolutionOutput, Task, RemoteMathEvaluator
from reason.evaluation.utils import setup_seed, jsonl_to_json
import torch
from functools import partial
import json
import jsonlines
import time
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import os
import random
from multiprocessing import Pool
import tree
from ray.util.actor_pool import ActorPool
from reason.evaluation.methods import *
import ray

from config import get_args


def parallel_evaluate_test_dataset(
    method_name: str,
    solver_fn: Callable,
    save_dir: Optional[Path] = None,
    record_writer: Optional[jsonlines.Writer] = None,
) -> List[Dict[str, Any]]:
    test_ds = task.test_ds
    # test_ds = task.train_ds

    results = []
    if args.resume_dir is not None:
        answered_questions = set()
        with jsonlines.open(Path(args.resume_dir) / "record.jsonl", "r") as reader:
            cnt = 0
            for obj in reader:
                results.append(obj["result"])
                answered_questions.add(obj["question"])
                if record_writer is not None:
                    record_writer.write(obj)
                    cnt += 1
        print(f"Resumed {cnt} questions from {args.resume_dir}")
        total_cnt = len(test_ds)
        test_ds = [
            problem_inst
            for problem_inst in test_ds
            if problem_inst["question"] not in answered_questions
        ]
        new_cnt = len(test_ds)
        print(
            f"After resuming, there are {new_cnt}/{total_cnt} new questions to answer."
        )

    actor_pool = ActorPool(
        [
            # RemoteMathEvaluator.remote(config.task_name, lm_call, rm_call)
            RemoteMathEvaluator.remote(task, lm_call, rm_call)
            for _ in range(args.num_worker)
        ]
    )
    res_q = actor_pool.map_unordered(
        lambda p, x: p.evaluate_problem.remote(x, solver_fn), test_ds
    )
    # Distributes tasks from the test_ds dataset across the worker pool asynchronously and
    # collects results in any order as they complete. Every worker has a new searching tree as we reset the
    # tree in solver_fn
    for i, item in enumerate(tqdm(res_q, total=len(test_ds))):
        problem_inst, result, output = item
        results.append(result)
        if record_writer:
            obj = {
                # "i": i,
                "question": problem_inst["question"],
                "groundtruth": problem_inst["answer"],
                "result": result,
                "output": output,
            }
            record_writer.write(obj)
    avg_res = (tree.map_structure(lambda *xs: np.mean(xs), *results),)
    if record_writer:
        json.dump(avg_res, open(save_dir / "avg_result.json", "w"))
    print("Method: {}. Average result: {}".format(method_name, avg_res))
    return results


if __name__ == "__main__":
    args = get_args()
    setup_seed(args.seed)

    if args.local:
        print("run in pure local mode for debug only")
        args.num_worker = 1
        ray.init(local_mode=True)

    # TODO(ziyu): move into some configuration file
    if "mistral" in args.RM.lower():
        prm_step_tag = "ки\n"
        prm_format_str = "{question} {answer}"
    elif "qwen2.5-math-prm" in args.RM.lower():
        prm_step_tag = "<extra_0>"
        prm_format_str = "{question}<this is qwen2.5-math-prm seperation &&&&& >{answer}"
    else:
        # assume qwen
        prm_step_tag = "\n\n\n\n\n "
        prm_format_str = "{question} {answer}"

    if "mistral" in args.LM.lower():
        lm_step_tag = "ки\n"
    else:
        # assume qwen
        lm_step_tag = "\n\n"

    lm_call = VLLMRemoteCaller(
        args.LM, args.controller_addr, lm_step_tag=lm_step_tag
    )

    if args.RM == "dummy":
        rm_config = RewardModelBaseConfig(
            step_tag=prm_step_tag, format_str=prm_format_str
        )
        rm_call = DummyRewardModelCaller(rm_config)
    else:
        rm_config = RemoteRewardModelConfig(
            step_tag=prm_step_tag,
            format_str=prm_format_str,
            model_name=args.RM,
            controller_addr=args.controller_addr,
        )
        rm_call = RMRemoteCaller(rm_config)

    task = Task(
        task_name=args.task_name,
        dataset_id=args.dataset,
        is_few_shot=args.is_few_shot,
    )

    cfg_dict_record = dict()
    # XXX: qwen-2.5 requires add more stop words
    # not do it now.
    # stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    gen_config = LMCallingConfig(
        n=args.num_sequence,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    cfg_dict_record["gen_config"] = gen_config.__dict__

    if args.method == "cot":
        method_config = CoTConfig(
            args.task_name,
        )
        solver_fn = partial(cot, method_config, gen_config, task)
    elif args.method == "best_of_n":
        method_config = BestOfNConfig(
            args.task_name,
            num_sequence=args.num_sequence,
        )
        solver_fn = partial(best_of_n, method_config, gen_config, task)
    elif args.method == "beam_search":
        method_config = BeamSearchConfig(
            task_name=args.task_name,
            tree_max_depth=args.tree_max_depth,
            tree_max_width=args.tree_max_width,
            beam_size=args.num_sequence,
        )
        solver_fn = partial(beam_search, method_config, gen_config, task)
    elif args.method == "vanila_mcts":
        method_config = VanilaMCTSConfig(
            task_name=args.task_name,
            tree_max_width=args.tree_max_width,
            tree_max_depth=args.tree_max_depth,
            select_by_prior=False,
            num_path=args.num_sequence,
        )
        solver_fn = partial(vanila_mcts, method_config, gen_config, task)
    elif args.method == "mcts":
        method_config = MCTSConfig(
            task_name=args.task_name,
            tree_max_width=args.tree_max_width,
            tree_max_depth=args.tree_max_depth,
            select_by_prior=False,
            num_path=args.num_sequence,
            simulate_num=args.simulate_num,
        )
        solver_fn = partial(mcts, method_config, gen_config, task)
    elif args.method == "rstar_mcts":
        method_config = RStarMCTSConfig(
            task_name=args.task_name,
            tree_max_width=args.tree_max_width,
            tree_max_depth=args.tree_max_depth,
            select_by_prior=False,
            num_path=args.num_sequence,
        )
        solver_fn = partial(rstar_mcts, method_config, gen_config, task)
    elif args.method == "mcts_beam_search":
        method_config = MCTSBeamSearchConfig(
            task_name=args.task_name,
            tree_max_width=args.tree_max_width,
            tree_max_depth=args.tree_max_depth,
            select_by_prior=False,
            num_path=args.num_sequence,  # TODO: add optional argument in script
            # beam_size=config.num_sequence,  # TODO: add optional argument in script
            simulate_num=args.simulate_num,
        )
        solver_fn = partial(mcts_beam_search, method_config, gen_config, task)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    cfg_dict_record["method"] = args.method
    cfg_dict_record["method_config"] = method_config.__dict__

    if args.local or args.save_dir is None:
        save_dir, record_writer = None, None
    else:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(args.save_dir) / task.task_name / args.method / datetime_str
        save_dir.mkdir(parents=True)
        record_writer = jsonlines.open(save_dir / f"record.jsonl", mode="w")
        cfg_dict_record["LM"] = args.LM
        cfg_dict_record["RM"] = args.RM
        json.dump(cfg_dict_record, open(save_dir / "config.json", "w"))

    parallel_evaluate_test_dataset(args.method, solver_fn, save_dir, record_writer)
    jsonl_to_json(save_dir / "record.jsonl", save_dir / "record.json")