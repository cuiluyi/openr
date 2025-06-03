from vllm import LLM, SamplingParams

llm = LLM(
    model="/data/cuiluyi/resources/models/Qwen/Qwen2.5-Math-1.5B-Instruct",
    trust_remote_code=True,
    gpu_memory_utilization=0.5,
)
print("LLM initialized successfully.")