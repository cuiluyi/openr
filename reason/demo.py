import requests

controller_addr = "http://0.0.0.0:28777"
model_name = "peiyi9979/mistral-7b-sft"

ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )

print(ret.json())  # 如果响应是 JSON 格式