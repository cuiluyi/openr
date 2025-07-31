import tree
from pathlib import Path
from tqdm import tqdm
from typing import Any, Callable, Dict, List

import jsonlines
import numpy as np
from torch.utils.data import Dataset
from ray.util.actor_pool import ActorPool

from utils import write_json
from reason.eval.evaluator import MathEvaluator


def parallel_evaluate_test_dataset(
    args,
    solver_fn: Callable,
    save_dir: Path,
    record_writer: jsonlines.Writer,
    rm_call: Callable,
    lm_call: Callable,
    dataset: Dataset,
) -> List[Dict[str, Any]]:

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
        total_cnt = len(dataset)
        dataset = [
            item
            for item in dataset
            if (item.get("question") or item.get("problem")) not in answered_questions
        ]
        new_cnt = len(dataset)
        print(f"After resuming, there are {new_cnt}/{total_cnt} new questions to answer.")

    actor_pool = ActorPool([MathEvaluator.remote(lm_call, rm_call) for _ in range(args.num_worker)])
    res_q = actor_pool.map_unordered(
        lambda p, x: p.evaluate_problem.remote(x, solver_fn),
        dataset,
    )
    # Distributes tasks from the dataset dataset across the worker pool asynchronously and collects results
    # in any order as they complete. Every worker has a new searching tree as we reset the tree in solver_fn
    for item in tqdm(res_q, total=len(dataset)):
        input_inst, result, output = item

        # exceptions are handled in the MathEvaluator
        if result is None and output is None:
            continue

        results.append(result)
        if record_writer:
            obj = {
                "question": input_inst["question"],
                "groundtruth": input_inst["gold"],
                "result": result,
                "output": output,
            }
            record_writer.write(obj)
    avg_res = (tree.map_structure(lambda *xs: np.mean(xs), *results),)
    if record_writer:
        write_json(avg_res, save_dir / "avg_result.json")
