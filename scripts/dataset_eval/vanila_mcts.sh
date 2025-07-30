python reason/main.py \
    --LM checkpoint-200 \
    --RM Qwen2.5-Math-PRM-7B \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.8 \
    --top_p 1 \
    --top_k -1 \
    --max_new_tokens 5120 \
    --num_sequence 6 \
    --tree_max_width 6 \
    --tree_max_depth 30 \
    --save_dir results \
    --method vanila_mcts \
    --num_worker 1 \
    --controller_addr http://0.0.0.0:28777

# --LM "Qwen2.5-Math-1.5B-Instruct&s1-20250312_213742&s1-20250314_003214" \