import tree
from pathlib import Path
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple

import jsonlines
import numpy as np
from loguru import logger
from ray.util.actor_pool import ActorPool

from utils import (
    write_json,
    jsonl_to_json,
    copy_file_to_dir,
    ensure_newline,
    DataItem,
)
from reason.eval.evaluator import MathEvaluator


import jsonlines
from pathlib import Path
from typing import List, Set, Optional


def resume_from_record(
    dataset: List[DataItem],
    resume_dir: Path,
    save_dir: Path,
) -> Tuple[List[DataItem], List[Dict[str, Any]]]:
    """
    Resume from previous records by reading answered questions and appending results.
    Args:
        dataset (List[DataItem]): The original dataset to filter.
        resume_dir (Path): Directory where the record file is stored.
        record_writer (jsonlines.Writer): Writer to append new records.
    Returns:
        The resumed dataset and results.
    """
    # fix the newline issue before opening the file
    for file_name in ["record.jsonl", "tree_step_data.jsonl"]:
        ensure_newline(resume_dir / file_name)

    # Copy the record and tree step data files to the save directory
    copy_file_to_dir(resume_dir / "record.jsonl", save_dir)
    copy_file_to_dir(resume_dir / "tree_step_data.jsonl", save_dir)

    # Read the record file to get answered questions
    answered_questions: Set[str] = set()
    results = []
    with jsonlines.open(resume_dir / "record.jsonl", "r") as reader:
        cnt = 0
        for obj in reader:
            results.append(obj["result"])
            answered_questions.add(obj["question"])
            # record_writer.write(obj)
            cnt += 1

    logger.info(f"Resumed {cnt} questions from {resume_dir}")
    raw_cnt = len(dataset)
    dataset = [item for item in dataset if item.question not in answered_questions]
    new_cnt = len(dataset)
    logger.info(f"After resuming, there are {new_cnt}/{raw_cnt} new questions to answer.")

    return dataset, results


def parallel_evaluate_dataset(
    solver_fn: Callable,
    rm_call: Callable,
    lm_call: Callable,
    dataset: List[DataItem],
    save_dir: Path,
    resume_dir: Optional[Path],
    num_worker: int,
) -> List[Dict[str, Any]]:
    results = []
    if resume_dir is not None:
        dataset, results = resume_from_record(dataset, resume_dir, save_dir)

    actor_pool = ActorPool([MathEvaluator.remote(lm_call, rm_call) for _ in range(num_worker)])
    res_q = actor_pool.map_unordered(lambda p, x: p.evaluate_problem.remote(x, solver_fn), dataset)

    # Distributes tasks from the dataset dataset across the worker pool asynchronously and collects results
    # in any order as they complete. Every worker has a new searching tree as we reset the tree in solver_fn
    record_writer = jsonlines.open(save_dir / f"record.jsonl", mode="a")
    tree_step_writer = jsonlines.open(save_dir / f"tree_step_data.jsonl", mode="a")
    for item in tqdm(res_q, total=len(dataset)):
        input_inst, result, output, tree_step_data = item

        if result is None and output is None:
            continue

        results.append(result)
        obj = {
            "question": input_inst.question,
            "groundtruth": input_inst.gold,
            "result": result,
            "output": output,
        }
        record_writer.write(obj)
        tree_step_writer.write(tree_step_data)

    avg_res = (tree.map_structure(lambda *xs: np.mean(xs), *results),)

    write_json(avg_res, save_dir / "avg_result.json")
    jsonl_to_json(save_dir / "record.jsonl", save_dir / "record.json")
    record_writer.close()
    tree_step_writer.close()
