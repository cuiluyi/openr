python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM checkpoint-2127 \
    --task_name MATH \
    --temperature 0.7 \
    --max_new_tokens 2048 \
    --num_sequence 8 \
    --tree_max_width 16 \
    --tree_max_depth 50 \
    --save_dir results \
    --method vanila_mcts \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    --local

# math-shepherd-mistral-7b-prm