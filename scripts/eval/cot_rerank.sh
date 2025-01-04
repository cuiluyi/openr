python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM checkpoint-2127 \
    --task_name MATH \
    --temperature 0.7 \
    --num_sequence 8 \
    --max_new_tokens 2048 \
    --save_dir results \
    --method best_of_n \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777


# math-shepherd-mistral-7b-prm
