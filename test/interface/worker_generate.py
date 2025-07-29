import requests

controller_addr = "http://0.0.0.0:28777"
# model_name = "RotatingHeadModel"
# model_name = "s1-20250312_213742"
# model_name = "Qwen2.5-Math-PRM-7B"
# model_name = "s1-20250314_003214"
# model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
# model_name = "Qwen2.5-Math-1.5B-Instruct"
model_name = "checkpoint-200"

prompt = "<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within \\boxed{}.<｜User｜>The expression $2\\cdot 3 \\cdot 4\\cdot 5+1$ is equal to 121, since multiplication is carried out before addition. However, we can obtain values other than 121 for this expression if we are allowed to change it by inserting parentheses. For example, we can obtain 144 by writing \\[\n(2\\cdot (3\\cdot 4)) \\cdot (5+1) = 144.\n\\]In total, how many values can be obtained from the expression $2\\cdot 3\\cdot 4 \\cdot 5 + 1$ by inserting parentheses? (Note that rearranging terms is not allowed, only inserting parentheses).<｜Assistant｜>"

gen_params = {
    "model": model_name,
    "prompt": prompt,
    "temperature": 0.6,
    "n": 4,
    "top_p": 1.0,
    "top_k": -1,
    "stop_token_ids": None,
    "max_new_tokens": 5120,
    "stop": ["<｜begin▁of▁system1｜>", "<｜begin▁of▁system2｜>", "<｜end▁of▁sentence｜>"],
    "echo": False,
    "include_stop_str_in_output": False,
    "repetition_penalty": 1.0,
}

worker_addr = requests.post(
    controller_addr + "/get_worker_address",
    json={"model": model_name},
).json()["address"]

headers = {"User-Agent": "FastChat Client"}

response = requests.post(
    worker_addr + "/worker_generate",
    headers=headers,
    json=gen_params,
    stream=True,
)
print(response)
results = response.json()

print(results)
