CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /data/cuiluyi/resources/models/Qwen/Qwen2.5-Math-1.5B

CUDA_VISIBLE_DEVICES=1,2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes 3 \
    src/open_r1/cepo.py \
    --config recipes/Qwen2.5-1.5B/cepo/config_MATH.yaml
