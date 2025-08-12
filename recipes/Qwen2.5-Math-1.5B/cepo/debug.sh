CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /data/cuiluyi/resources/models/Qwen/Qwen2.5-Math-1.5B



CUDA_VISIBLE_DEVICES=1,2,3 python -m debugpy \
    --listen 59910 \
    --wait-for-client -m accelerate.commands.launch \
    --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes 3 \
    --main_process_port 29501 \
    src/open_r1/cepo.py \
    --config recipes/Qwen2.5-1.5B/cepo/config_MATH.yaml
