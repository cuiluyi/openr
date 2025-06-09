import requests

controller_addr = "http://0.0.0.0:28777"
# model_name = "RotatingHeadModel"
# model_name = "s1-20250312_213742"
# model_name = "Qwen2.5-Math-PRM-7B"
# model_name = "math-shepherd-mistral-7b-prm"
model_name = "Math-psa-7B"
# model_name = "s1-20250314_003214"

response = requests.post(controller_addr + "/get_worker_address", json={"model": model_name})
# response = requests.post(controller_addr + "/get_worker_address")

results = response.json()

print(results)
