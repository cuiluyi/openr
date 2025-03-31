import json

import torch.nn.functional as F
from math_verify import parse, verify
from transformers import AutoTokenizer


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[
            :, 1
        ]  # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def load_and_process(file_path: str, tokenizer):
    dataset_data = {"texts": [], "labels": []}
    data_list = read_json(file_path) if file_path.endswith("json") else read_jsonl(file_path)
    for data in data_list:
        # 处理每个输出路径
        for item in data["output"]:
            messages = [
                {
                    "role": "system",
                    "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                },
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": item["text"] + "<extra_0>"},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            dataset_data["texts"].append(text)

            ground_truth = parse(data["groundtruth"])
            ans = parse(item["text"])
            label = int(verify(ground_truth, ans))
            dataset_data["labels"].append(label)
    return dataset_data


if __name__ == "__main__":
    model_name = "/data/cuiluyi/resources/models/Qwen/Qwen2.5-Math-PRM-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 初始化模型和分词器

    # model = AutoModel.from_pretrained(
    #     model_name,
    #     device_map=device,
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    # ).eval()

    # test_data_file = Path(
    #     "/data/cuiluyi/s1/train/Qwen2.5-Math-PRM-7B/model/test_data.json"
    # )
    # with open(test_data_file, "r", encoding="utf-8") as f:
    #     raw_data = json.load(f)

    # batch_data = []
    # for i in range(len(raw_data["output"])):
    #     batch_data.append(
    #         {
    #             "system": "Please reason step by step, and put your final answer within \\boxed{}.",
    #             "query": raw_data["question"],
    #             "response": [
    #                 raw_data["output"][i]["text"],
    #             ],
    #         }
    #     )

    # # 生成批次对话字符串
    # texts = []
    # for data in batch_data:
    #     messages = [
    #         {"role": "system", "content": data["system"]},
    #         {"role": "user", "content": data["query"]},
    #         {
    #             "role": "assistant",
    #             "content": "<extra_0>".join(data["response"]) + "<extra_0>",
    #         },
    #     ]
    #     text = tokenizer.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=False
    #     )
    #     texts.append(text)

    # # 批量编码
    # inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
    #     model.device
    # )
    # input_ids = inputs.input_ids

    # # 模型推理
    # with torch.no_grad():
    #     outputs = model(input_ids=input_ids)

    # # 生成token masks
    # step_sep_id = tokenizer.encode("<extra_0>")[0]
    # token_masks = input_ids == step_sep_id

    # # 计算步骤奖励
    # step_rewards = make_step_rewards(outputs[0], token_masks)

    # print("Batch step rewards:")
    # for i, reward in enumerate(step_rewards):
    #     print(f"Sample {i+1}: {reward}")
