python -m debugpy \
    --listen 59910 \
    --wait-for-client -m accelerate.commands.launch \
    --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes 3 \
    --main_process_port 29501 \
    src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B/grpo/config_MATH.yaml
