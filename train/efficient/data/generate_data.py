from datasets import load_dataset
from math_verify import parse, verify

def func(example):
    problem = example["problem"]
    ground_truth = parse(example["solution"])
    generation = example["generation"]

    splits = generation.split("**Final Answer**")
    think = splits[0]

    efficient_think = []
    for line in think.split("\n\n"):
        answer = parse(line)
        efficient_think.append(line)
        if verify(ground_truth, answer):
            break
    efficient_think = "\n\n".join(efficient_think) + "**Final Answer**" + splits[-1]

    example["messages"] = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": efficient_think},
    ]



dataset = load_dataset("HuggingFaceH4/numina-deepseek-r1-qwen-7b", split="train")

dataset.map(func)

dataset.push_to_hub("LuyiCui/efficient_cot")