# add reward model worker
export LOGDIR=logs/fastchat
CUDA_VISIBLE_DEVICES=6 /data/cuiluyi/anaconda3/envs/open_reasoner/bin/python -m reason.serve.reward_model_worker --model-path /data/cuiluyi/resources/models/Qwen/Qwen2.5-Math-PRM-7B --controller-address http://0.0.0.0:28777 --host 0.0.0.0 --port 30018 --worker-address http://0.0.0.0:30018

# add language model worker
export LOGDIR=logs/fastchat
CUDA_VISIBLE_DEVICES=5 /data/cuiluyi/anaconda3/envs/open_reasoner/bin/python -m reason.serve.vllm_worker --model-path /data/cuiluyi/open-r1/ckpts/DeepSeek-R1-Distill-Qwen-1.5B/slow_fast_reason-sft-s1k-1.1_full/checkpoint-400 --controller-address http://0.0.0.0:28777 --host 0.0.0.0 --port 30015 --worker-address http://0.0.0.0:30015