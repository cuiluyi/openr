python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM Qwen2.5-Math-PRM-7B \
    --task_name MATH \
    --temperature 0.0 \
    --top_p 0.95 \
    --top_k -1 \
    --max_new_tokens 2048 \
    --save_dir results \
    --method cot \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    # --local

# Tips: Make sure the input (--LM, --RM) in the script aligns with the command output (basename $LANGUAGE_MODEL_NAME, basename $REWARD_MODEL_NAME) in the pending worker!