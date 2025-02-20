python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM Qwen2.5-Math-PRM-7B \
    --task_name rstar \
    --max_new_tokens 2048 \
    --num_sequence 8 \
    --tree_max_width 4 \
    --tree_max_depth 50 \
    --save_dir results \
    --method rstar_mcts \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    --top_k 40 \
    --top_p 0.95 \
    --temperature 0.8 \
    # --local

# Tips: Make sure the input (--LM, --RM) in the script aligns with the command output (basename $LANGUAGE_MODEL_NAME, basename $REWARD_MODEL_NAME) in the pending worker!