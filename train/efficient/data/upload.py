import jsonlines
from datasets import Dataset
from transformers import AutoTokenizer

model_path = "/data/cuiluyi/resources/models/Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

jsonl_file = "/data/cuiluyi/openr/train/efficient/data/sft_data.jsonl"
data_dict = {"text": list()}
with jsonlines.open(jsonl_file) as f:
    for item in f:
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["completion"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        data_dict["text"].append(text)
dataset = Dataset.from_dict(data_dict)

dataset.push_to_hub("LuyiCui/efficient-math")