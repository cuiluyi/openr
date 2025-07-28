from transformers import AutoTokenizer

def build_query_str(
    problem_input: str,
    model_name: str = "/data/cuiluyi/resources/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
):
    # XXX: this is a hack to make the tokenizer work with the new vllm
    """
    Build the query string for the problem input.
    """
    messages = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {"role": "user", "content": problem_input},
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ret = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return ret