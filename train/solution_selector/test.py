import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

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
batch_data = [
    {
        "system": "Please reason step by step...",
        "query": "Sue lives in a fun neighborhood...",
        "response": [
            "To find out how many more pink...",
            "On Saturday, they take back one third...",
            "On Sunday, the neighbors add...",
            "To find the difference..."
        ]
    },
    {
        "system": "Another system prompt...",
        "query": "Different math problem...",
        "response": [
            "First step analysis...",
            "Second step calculation...",
            "Final conclusion..."
        ]
    }
]

# 生成批次对话字符串
texts = []
for data in batch_data:
    messages = [
        {"role": "system", "content": data['system']},
        {"role": "user", "content": data['query']},
        {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
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

print("Batch step rewards:")
for i, reward in enumerate(step_rewards):
    print(f"Sample {i+1}: {reward}")