import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from math_verify import parse, verify
from transformers import AutoModel, AutoTokenizer, TrainingArguments

from train.solution_selector.trainer import CustomTrainer
from train.solution_selector.loss import pairwise_rank_loss
from train.solution_selector.utils import make_step_rewards, load_and_process

device = "cuda"

# 初始化 tokenizer
model_name = "/data/cuiluyi/resources/models/Qwen/Qwen2.5-Math-PRM-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def preprocess(example):
    return tokenizer(
        example["texts"],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)


# 准备数据集

# data_file_path = "/data/cuiluyi/openr/train/solution_selector/train_data.json"
data_file_path = "/data/cuiluyi/openr/train/solution_selector/train_data_5k.jsonl"
data_dict = load_and_process(data_file_path, tokenizer)
dataset = Dataset.from_dict(data_dict)
# DatasetDict({
#     train: Dataset({
#         features: ['question', 'groundtruth', 'result', 'output'],
#         num_rows: 500
#     })
# })


# dataset = dataset.map(preprocess, batched=True)
dataset = dataset.map(
    preprocess,
    batched=True,
    batch_size=4,
)


# DatasetDict({
#     train: Dataset({
#         features: ['question', 'groundtruth', 'result', 'output', 'input_ids', 'attention_mask', 'labels'],
#         num_rows: 500
#     })
# })


model = AutoModel.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# step_sep_id = tokenizer.encode("<extra_0>")[0]
# 151651


def compute_loss_func(inputs, outputs, labels, num_items_in_batch=None):
    input_ids = inputs["input_ids"]
    token_masks = (input_ids == 151651)
    step_rewards = [item[-1] for item in make_step_rewards(outputs[0], token_masks)]
    return pairwise_rank_loss(labels, step_rewards)


# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results/solution_selector",  # 输出目录
    evaluation_strategy="no",
    learning_rate=0.0001,
    num_train_epochs=3,  # 训练周期数
    per_device_train_batch_size=4,  # 批量大小
    per_device_eval_batch_size=4,  # 验证批量大小
    gradient_accumulation_steps=5,
    weight_decay=0.01,  # 权重衰减
    logging_dir="./logs",  # 日志目录
    logging_steps=10,
    eval_steps=50,
    save_strategy="epoch",
)

# 创建 Trainer 对象
trainer = CustomTrainer(
    model=model,
    args=training_args,
    # train_dataset=dataset["train"],
    # eval_dataset=dataset["test"] if "test" in dataset else dataset["train"],
    train_dataset=dataset,
    eval_dataset=dataset,
    compute_loss_func=compute_loss_func,
)

# 训练模型
trainer.train()

# # 评估模型
# trainer.evaluate()

# Save the fine-tuned model and tokenizer
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"./ckpts/solution_selector_{timestamp}"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)