import torch
import torch.nn.functional as F


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[
            :, 1
        ]  # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return torch.tensor(all_scores_res)


@torch.inference_mode()
def _qwen_math_prm_infer_fn(input_str: str, model, tokenizer, device):
    STEP_TAG = "<extra_0>"
    question, answer = input_str.split("<this is qwen2.5-math-prm seperation &&&&& >")

    data = {
        "system": "Please reason step by step, and put your final answer within \\boxed{}.",
        "query": question,
        "response": answer,
    }

    messages = [
        {"role": "system", "content": data["system"]},
        {"role": "user", "content": data["query"]},
        {"role": "assistant", "content": data["response"]},
    ]

    conversation_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    input_ids = tokenizer.encode(
        conversation_str,
        return_tensors="pt",
    ).to(model.device)

    outputs = model(input_ids=input_ids)

    step_sep_id = tokenizer.encode(STEP_TAG)[0]
    token_masks = input_ids == step_sep_id
    step_reward = make_step_rewards(outputs[0], token_masks)[0]
    return step_reward


@torch.inference_mode()
def _qwen_math_infer_fn(input_str: str, model, tokenizer, device):
    GOOD_TOKEN = "+"
    BAD_TOKEN = "-"
    STEP_TAG = "\n\n\n\n\n"

    candidate_tokens = tokenizer.encode(f" {GOOD_TOKEN} {BAD_TOKEN}")  # [488, 481]
    step_tag_id = torch.tensor(
        [tokenizer.encode(f" {STEP_TAG}")], device=device
    )  # 76325
    input_id = torch.tensor([tokenizer.encode(input_str)], device=device)
    logits = model(input_id).logits[:, :, candidate_tokens]

    scores = logits.softmax(dim=-1)[:, :, 0]
    mask = input_id == step_tag_id
    step_scores = scores[mask]
    return step_scores


@torch.inference_mode()
def _math_shepherd_infer_fn(input_str: str, model, tokenizer, device):
    GOOD_TOKEN = "+"
    BAD_TOKEN = "-"
    STEP_TAG = "ки"
    candidate_tokens = tokenizer.encode(f"{GOOD_TOKEN} {BAD_TOKEN}")[1:]  # [648, 387]
    step_tag_id = tokenizer.encode(f"{STEP_TAG}")[-1]  # 12902

    input_id = torch.tensor([tokenizer.encode(input_str)], device=device)
    logits = model(input_id).logits[:, :, candidate_tokens]
    scores = logits.softmax(dim=-1)[:, :, 0]
    step_scores = scores[input_id == step_tag_id]
    return step_scores
