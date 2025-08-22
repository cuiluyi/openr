CUDA_VISIBLE_DEVICES=4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    python -m debugpy --listen 63655 --wait-for-client \
    -m accelerate.commands.launch \
    --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes 4 \
    train/grpo.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
    --vllm_mode colocate \
    > ckpts/DeepSeek-R1-Distill-Qwen-1.5B-GRPO/output.log 2>&1