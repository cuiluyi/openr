import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from tqdm import tqdm
import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from math_verify import parse, verify

# model_name = "/data/cuiluyi/openr/train/efficient/tmp/checkpoint-1270"
model_name = "/data/cuiluyi/resources/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# model_name = "/data/cuiluyi/resources/models/Qwen/Qwen2.5-Math-1.5B-Instruct"
device = "cuda"

# 模型加载优化：半精度 + Flash Attention（如可用）
try:
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
except Exception as e:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
if tokenizer.pad_token is None:     # 确保pad_token存在
    tokenizer.pad_token = tokenizer.eos_token

acc_num, total_num = 0, 0

dataset = load_dataset("LuyiCui/MATH-openai-split", split='test').to_list()  # 转为列表方便分批

# 设置批处理大小（根据GPU内存调整）
batch_size = 128

for i in tqdm(range(0, len(dataset), batch_size)):
    batch_items = dataset[i:i+batch_size]
    batch_texts = []
    for item in batch_items:
        messages = [
            {"role": "system", "content": "Please reason step by step..."},
            {"role": "user", "content": item["problem"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch_texts.append(text)
    
    # 批量编码并填充
    model_inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    # 生成时禁用梯度 + 优化参数
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,           # 根据实际调整最大生成长度
            num_beams=1,                  # 贪心解码加速
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True                 # 启用缓存加速
        )
    
    # 批量解码
    input_lengths = model_inputs.input_ids.shape[1]
    outputs = generated_ids[:, input_lengths:]
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # 验证结果
    for j, resp in enumerate(responses):
        ground_truth = parse(batch_items[j]["solution"])
        acc_num += verify(ground_truth, parse(resp))
        total_num += 1

print(f"Accuracy: {acc_num}/{total_num} = {acc_num/total_num:.4f}")