python reason/main.py \
    --LM checkpoint-400 \
    --RM Skywork-Reward \
    --dataset Idavidrein/gpqa \
    --split train \
    --subset gpqa_main \
    --temperature 0.8 \
    --top_p 1 \
    --top_k -1 \
    --max_new_tokens 5120 \
    --num_sequence 3 \
    --max_width 3 \
    --max_steps 20 \
    --save_dir results \
    --method vanilla_mcts \
    --num_worker 64 \
    --controller_addr http://0.0.0.0:28777



# python -m debugpy --listen 63655 --wait-for-client \
#     reason/main.py \
#     --LM checkpoint-400 \
#     --RM Skywork-Reward \
#     --dataset Idavidrein/gpqa \
#     --split train \
#     --subset gpqa_main \
#     --temperature 0.8 \
#     --top_p 1 \
#     --top_k -1 \
#     --max_new_tokens 5120 \
#     --num_sequence 3 \
#     --max_width 3 \
#     --max_steps 20 \
#     --save_dir results \
#     --method vanilla_mcts \
#     --num_worker 64 \
#     --controller_addr http://0.0.0.0:28777 \
#     --local
