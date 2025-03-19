import jsonlines
import importlib
from tqdm import tqdm

from results_process.answe_selection_utils import generate_solution

task_module = importlib.import_module(f"envs.MATH")

# file_path = "results/MATH/vanila_mcts/20241228_101217/record.jsonl"
file_path = "results/demo.jsonl"


def func(solution_list: list[str]) -> str:
    unique_solution_list, unique_answer_list = [], []

    for i, solution in enumerate(solution_list):
        answer = task_module.extract_answer(solution)
        if answer in unique_answer_list:
            continue
        unique_solution_list.append(f"Solution{i + 1}: {solution}")
        unique_answer_list.append(answer)

    return "\n".join(unique_solution_list)


def answer_selection():

    correct_num, total_num = 0, 0
    data_list: list[dict] = []

    with jsonlines.open(file_path) as f:
        for obj in tqdm(f, desc="Processing lines"):
            question = "Question: " + obj["question"] + "\n"

            ground_solution = obj["groundtruth"]
            ground_answer = task_module.extract_answer(ground_solution)

            solution_list = [ans["text"] for ans in obj["output"]]
            unique_solutions: str = func(solution_list)

            finial_response = generate_solution(question + unique_solutions)
            finial_answer = task_module.extract_answer(finial_response)

            if finial_answer == ground_answer:
                correct_num += 1

            total_num += 1

            dict_item = {
                "question": question,
                "ground_solution": ground_solution,
                "ground_answer": ground_answer,
                "solution_candidates": unique_solutions,
                "finial_solution": finial_response,
                "finial_answer": finial_answer,
            }
            data_list.append(dict_item)

    save_path = "/".join(file_path.split("/")[:-1]) + "/" + "post_process.jsonl"

    with jsonlines.open(save_path, "w") as f:
        f.write_all(data_list)

    print(f"correct_num: {correct_num}, accuracy: {(correct_num / total_num):.3f}")


if __name__ == "__main__":
    answer_selection()
