import requests

controller_addr = "http://0.0.0.0:28777"
# model_name = "RotatingHeadModel"
# model_name = "s1-20250312_213742"
# model_name = "Qwen2.5-Math-PRM-7B"
# model_name = "s1-20250314_003214"
model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
prompt = '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\nBelow is the graph of $y = a \\sin (bx + c) + d$ for some positive constants $a,$ $b,$ $c,$ and $d.$  Find the smallest possible value of $c.$\n\n[asy]import TrigMacros;\n\nsize(400);\n\nreal f(real x)\n{\n\treturn 2*sin(3*x + pi) + 1;\n}\n\ndraw(graph(f,-3*pi,3*pi,n=700,join=operator ..),red);\ntrig_axes(-3*pi,3*pi,-4,4,pi/2,1);\nlayer();\nrm_trig_labels(-5,5, 2);\n\nlabel("$1$", (0,1), E);\nlabel("$2$", (0,2), E);\nlabel("$3$", (0,3), E);\nlabel("$-1$", (0,-1), E);\nlabel("$-2$", (0,-2), E);\nlabel("$-3$", (0,-3), E);\n[/asy]<|im_end|>\n<|im_start|>assistant\n'

gen_params = {
    "model": model_name,
    "prompt": prompt,
    "temperature": 0.7,  # 0.7
    "n": 1,
    "top_p": 1.0,
    "top_k": 1,
    "stop_token_ids": None,
    "max_new_tokens": 10240,
    "stop": None,
    "echo": False,
    "include_stop_str_in_output": False,
    "seed": 0,
    "repetition_penalty": 3,
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
