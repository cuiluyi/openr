from transformers import AutoTokenizer
from constants import DEFAULT_TOKENIZER_PATH


def build_query_str(problem_input: str):
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

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_PATH, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removesuffix("<think>\n")
    return prompt
