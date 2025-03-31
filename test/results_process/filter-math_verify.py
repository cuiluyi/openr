import jsonlines
import importlib
import json

from math_verify import parse, verify


file_path = (
    "/data/cuiluyi/openr/results/metrics/test/mcts_beam_search/20250311_204912/record.jsonl"
    # "/data/cuiluyi/openr/results/MATH/mcts_beam_search/20250311_204912/record.jsonl"
    # "/data/cuiluyi/openr/results/MATH/mcts_beam_search/20250311_204912/partial_correct_data.json"
)


def write_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def check_fully_correct(a, b, c, d):
    if a + b + c + d == 4:
        return True
    return False


def check_partial_correct(a, b, c, d, ground_truth, ans_list):
    if a + b + c + d != 0:
        return False
    for ans in ans_list:
        if verify(ground_truth, ans):
            return True
    return False


partial_correct_data = []
fully_correct_data = []
fully_wrong_data = []

with jsonlines.open(file_path) as f:
    for obj in f:
    # 后续操作
        ground_answer = parse(obj["groundtruth"])

        ans_list = [parse(ans["text"]) for ans in obj["output"]]

        results = obj["result"]
        # majority_vote, prm_min_max, prm_min_vote, prm_last_max = (
        #     results.get("majority_vote", 1),
        #     results.get("prm_min_max", 1),
        #     results.get("prm_min_vote", 1),
        #     results.get("prm_last_max", 1),
        # )
        majority_vote = results["majority_vote"]
        try:
            prm_min_max, prm_min_vote, prm_last_max = (
                results["prm_min_max"],
                results["prm_min_vote"],
                results["prm_last_max"],
            )
        except Exception as e:
            print(obj)
            prm_min_max, prm_min_vote, prm_last_max = (
                results.get("prm_min_max", majority_vote),
                results.get("prm_min_vote", majority_vote),
                results.get("prm_last_max", majority_vote),
            )

        if check_fully_correct(majority_vote, prm_min_max, prm_min_vote, prm_last_max):
            fully_correct_data.append(obj)
        elif check_partial_correct(
            majority_vote,
            prm_min_max,
            prm_min_vote,
            prm_last_max,
            ground_answer,
            ans_list,
        ):
            partial_correct_data.append(obj)
        else:
            fully_wrong_data.append(obj)

partial_correct_path = (
    "/".join(file_path.split("/")[:-1]) + "/" + "partial_correct_data.json"
)
fully_correct_path = (
    "/".join(file_path.split("/")[:-1]) + "/" + "fully_correct_data.json"
)
fully_wrong_path = "/".join(file_path.split("/")[:-1]) + "/" + "fully_wrong_data.json"

write_json(partial_correct_data, partial_correct_path)
write_json(fully_correct_data, fully_correct_path)
write_json(fully_wrong_data, fully_wrong_path)
