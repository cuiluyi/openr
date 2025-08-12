# add reward model worker
export LOGDIR=logs/fastchat
CUDA_VISIBLE_DEVICES=7 /data/cuiluyi/anaconda3/envs/open_reasoner/bin/python -m reason.serve.reward_model_worker --model-path /data/cuiluyi/resources/models/Qwen/Qwen2.5-Math-PRM-7B --controller-address http://0.0.0.0:28777 --host 0.0.0.0 --port 30018 --worker-address http://0.0.0.0:30018

# add language model worker
export LOGDIR=logs/fastchat
CUDA_VISIBLE_DEVICES=7 /data/cuiluyi/anaconda3/envs/open_reasoner/bin/python -m reason.serve.vllm_worker --model-path /data/cuiluyi/openr/ckpts/DeepSeek-R1-Distill-Qwen-1.5B/slow_fast_reason-sft-s1k-1.1_full/checkpoint-400 --controller-address http://0.0.0.0:28777 --host 0.0.0.0 --port 30025 --worker-address http://0.0.0.0:30025  --gpu_memory_utilization 0.45