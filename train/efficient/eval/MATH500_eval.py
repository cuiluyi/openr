import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tqdm import tqdm

import jsonlines
import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from math_verify import parse, verify

model_name = "/data/cuiluyi/openr/train/efficient/tmp/checkpoint-2"
device = "cuda" # the device to load the model onto

model = AutoPeftModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

acc_num, total_num = 0, 0

dataset = load_dataset("LuyiCui/MATH-openai-split", split='test')

writer = jsonlines.open("/data/cuiluyi/openr/train/efficient/eval/result.jsonl", mode='w')

for item in tqdm(dataset):
    prompt = item["problem"]

    # CoT
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=20480,
        use_cache=True,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    temp = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = parse(temp)
    ground_truth = parse(item["solution"])
    total_num += 1
    flag = verify(ground_truth, response)
    acc_num += flag

    item["response"] = temp
    item["flag"] = flag
    writer.write(item)

print(f"{acc_num=}, {total_num=}, {acc_num/total_num=}")