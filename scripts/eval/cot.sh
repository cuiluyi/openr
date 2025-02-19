python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --task_name MATH \
    --temperature 0.0 \
    --max_new_tokens 2048 \
    --save_dir results \
    --method cot \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    # --local

# Tips: Make sure the input (--LM, --RM) in the script aligns with variable ($POLICY_MODEL_NAME, $VALUE_MODEL_NAME) in the pending worker!