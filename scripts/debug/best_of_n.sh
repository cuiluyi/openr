python -m debugpy --listen 63655 --wait-for-client \
    reason/evaluation/evaluate.py \\
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM Qwen2.5-Math-PRM-7B \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.7 \
    --top_p 1 \
    --top_k -1 \
    --num_sequence 4 \
    --max_new_tokens 2048 \
    --save_dir results \
    --method best_of_n \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    --local

# --LM "Qwen2.5-Math-1.5B-Instruct&s1-20250312_213742&s1-20250314_003214" \
