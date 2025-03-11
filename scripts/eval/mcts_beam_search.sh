python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM Qwen2.5-Math-PRM-7B \
    --task_name MATH \
    --temperature 0.8 \
    --top_p 0.95 \
    --top_k -1 \
    --max_new_tokens 2048 \
    --num_sequence 4 \
    --simulate_num 8 \
    --tree_max_width 4 \
    --tree_max_depth 50 \
    --save_dir results \
    --method mcts_beam_search \
    --num_worker 32 \
    --controller_addr http://0.0.0.0:28777 \
    # --local


# Tips: Make sure the input (--LM, --RM) in the script aligns with the command output (basename $LANGUAGE_MODEL_NAME, basename $REWARD_MODEL_NAME) in the pending worker!

# if you debug into the ray code, you must set the option '--local'