import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

from train.solution_selector.utils import make_step_rewards, read_json, read_jsonl
from math_verify import verify, parse

# 初始化模型和分词器
model_name = "/data/cuiluyi/resources/models/Qwen/Qwen2.5-Math-PRM-7B"
device = "auto"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    device_map=device, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

# 示例批次数据（包含两个样本）
json_data = read_jsonl("/data/cuiluyi/openr/train/solution_selector/test_data.jsonl")

# 生成批次对话字符串
correct_num, total_num = 0, 0
for item in tqdm(json_data):
    texts = []
    for data in item["output"]:
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": item['question']},
            {"role": "assistant", "content": data['text'] + "<extra_0>"},
        ]
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        texts.append(text)

    # 批量编码
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)
    input_ids = inputs.input_ids

    # 模型推理
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    # 生成token masks
    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)

    # 计算步骤奖励
    step_rewards = make_step_rewards(outputs.logits, token_masks)

    index = step_rewards.index(max(step_rewards))

    ans = parse(item["output"][index]["text"])
    ground_truth = parse(item["groundtruth"])
    correct_num = correct_num + int(verify(ans, ground_truth))
    total_num = total_num + 1

print(f"Accuracy: {correct_num / total_num}")