python reason/evaluation/evaluate.py \
    --LM mistral-7b-sft \
    --RM math-shepherd-mistral-7b-prm \
    --task_name MATH \
    --temperature 0.7 \
    --max_new_tokens 2048 \
    --num_sequence 1 \
    --tree_max_width 4 \
    --tree_max_depth 50 \
    --save_dir results \
    --method beam_search \
    --num_worker 32 \
    --controller_addr http://0.0.0.0:28777

# Tips: Make sure the input (--LM, --RM) in the script aligns with variable ($POLICY_MODEL_NAME, $VALUE_MODEL_NAME) in the pending worker!