import os
import jsonlines
import importlib
import json
import ray
from reason.evaluation.evaluator import judge_ans

task_module = importlib.import_module(f"envs.MATH")


def get_all_files(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename == "record.jsonl":
                files.append(os.path.join(root, filename))
    return files


@ray.remote
def pass_N_accuracy(file_path: str) -> float:
    cnt, num = 0, 0
    try:
        with jsonlines.open(file_path) as f:
            for obj in f:
                question = obj["question"]
                groundtruth = task_module.extract_groundtruth(obj["groundtruth"])
                ans_list = [
                    task_module.extract_answer(ans["text"]) for ans in obj["output"]
                ]
                for answer in ans_list:
                    if task_module.judge_correct(question, groundtruth, answer):
                        cnt = cnt + 1
                        break
                num = num + 1
        return cnt / num
    except Exception as e:
        print(e)
        print(file_path)
        return 0.0


@ray.remote
def majority_vote(file_path: str) -> float:
    cnt, num = 0, 0
    with jsonlines.open(file_path) as f:
        for obj in f:
            question = obj["question"]
            groundtruth = task_module.extract_groundtruth(obj["groundtruth"])
            solution_list = [item["text"] for item in obj["output"]]
            reward_list = [0.0] * len(solution_list)

            flag = judge_ans(
                question,
                groundtruth,
                solution_list,
                reward_list,
                "majority_vote",
                task_module.extract_answer,
                task_module.judge_correct,
            )
            if flag:
                cnt = cnt + 1
            else:
                if obj["result"]["majority_vote"]:
                    print(obj)
            num = num + 1
    return cnt / num


if __name__ == "__main__":
    ray.init()

    path = "/data/cuiluyi/openr/results/MATH/mcts_beam_search/20250311_204912"
    # path = "/data/cuiluyi/openr/results/MATH/cot/20241222_125150"
    # path = "/data/cuiluyi/openr/results/rstar_mcts"
    files = get_all_files(path)

    accuracy_tasks = [pass_N_accuracy.remote(file) for file in files]
    accuracies: list[float] = ray.get(accuracy_tasks)

    for file, accuracy in zip(files, accuracies):
        try:
            avg_file_path = "/".join(file.split("/")[:-1]) + "/" + "avg_result.json"
            with open(avg_file_path, "r") as f:
                results = json.load(f)

            results[0]["pass@N"] = accuracy
            with open(avg_file_path, "w") as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            print(e, file)

    ray.shutdown()