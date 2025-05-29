python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM Qwen2.5-Math-PRM-7B \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.7 \
    --top_p 0.95 \
    --top_k -1 \
    --max_new_tokens 2048 \
    --save_dir results/w_correct_filter \
    --method cot \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    # --local