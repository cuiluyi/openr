python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM checkpoint-2127 \
    --task_name MATH \
    --temperature 0.7 \
    --max_new_tokens 2048 \
    --num_sequence 8 \
    --tree_max_width 4 \
    --tree_max_depth 50 \
    --save_dir results \
    --method beam_search \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777

# Tips: Make sure the input (--LM, --RM) in the script aligns with variable ($POLICY_MODEL_NAME, $VALUE_MODEL_NAME) in the pending worker!