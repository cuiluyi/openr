# from datasets import load_dataset

# data_files = {
#     "train": "/data/cuiluyi/openr/envs/MATH/dataset/train12k.jsonl",
#     "test": "/data/cuiluyi/openr/envs/MATH/dataset/test500.jsonl",
# }

# dataset = load_dataset("json", data_files=data_files)

# dataset.push_to_hub("LuyiCui/MATH-openai-split")


from datasets import load_dataset, DatasetDict

# 1. 从 Hub 加载整个数据集（包含所有 split）
dataset = load_dataset("LuyiCui/MATH")

# 2. 定义转换函数，将 "level" 字段转换为整数
def process_sample(sample):
    try:
        # 假设 "level" 字段格式为 "level X"，提取 X 并转换为整数
        new_level = int(sample["level"].split()[1])
    except Exception as e:
        # 如果格式不符合预期，则保持原值或处理错误
        new_level = 3
    return {"level": new_level}

# 3. 对每个 split 应用转换
modified_splits = {}
for split_name, split_dataset in dataset.items():
    modified_splits[split_name] = split_dataset.map(process_sample)

# 使用 DatasetDict 将各个 split 组合成一个整体数据集
modified_dataset = DatasetDict(modified_splits)

# 4. 将修改后的数据集推送到 Hugging Face Hub
# 请将 "your_username/your_dataset_name_modified" 修改为你希望上传的新数据集名称
modified_dataset.push_to_hub("LuyiCui/MATH")



# import json
# import re

# file_path = "/data/cuiluyi/openr/envs/MATH/dataset/test500.jsonl"
# # 读取原始文件
# with open(file_path, "r") as f:
#     lines = f.readlines()

# # 处理并写入新文件
# with open(file_path, "w") as f:
#     for line in lines:
#         data = json.loads(line)
#         data["level"] = data["level"].split(" ")[-1]
#         f.write(json.dumps(data) + "\n")