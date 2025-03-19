import requests

controller_addr = "http://0.0.0.0:28777"
# model_name = "RotatingHeadModel"
# model_name = "s1-20250312_213742"
# model_name = "Qwen2.5-Math-PRM-7B"
model_name = "s1-20250314_003214"

worker_addr = requests.post(controller_addr + "/get_worker_address", json={"model": model_name}).json()["address"]
headers = {"User-Agent": "FastChat Client"}
gen_params = {
    "model": model_name,
    "prompt": "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\nConvert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$<|im_end|>\n<|im_start|>assistant\n",
    "temperature": 0.7,
    "n": 1,
    "top_p": 1.0,
    "top_k": 1,
    "stop_token_ids": None,
    "max_new_tokens": 2048,
    "stop": None,
    "echo": False,
    "include_stop_str_in_output": False,
    "seed": 0
}

response = requests.post(
    worker_addr + "/worker_generate",
    headers=headers,
    json=gen_params,
    stream=True,
)
print(response)
results = response.json()

print(results)
