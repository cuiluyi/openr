import json
import jsonlines

with open("results/MATH/vanila_mcts/20250224_122252/record.json", "r") as f:
    data = json.load(f)

incorrect_data = []

for item in data:
    if item["result"]["majority_vote"] == 0:
        incorrect_data.append(item)

error_question_data, incorrect_question_data = [], []
with jsonlines.open("envs/MATH/dataset/test500.jsonl", mode = 'r') as f:
    for item_question in f:
        for item_incorrect_data in incorrect_data:
            if item_question["problem"] == item_incorrect_data["question"]:
                incorrect_question_data.append(item_question)

with jsonlines.open("results/MATH/vanila_mcts/20250224_122252/incorrect_data.jsonl", "w") as f:
    f.write_all(incorrect_question_data)