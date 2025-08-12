python -m debugpy \
    --listen 59910 \
    --wait-for-client -m accelerate.commands.launch \
    --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes 3 \
    --main_process_port 29501 \
    train/cepo.py \
    --config recipes/Qwen2.5-1.5B/cepo/config_MATH.yaml
