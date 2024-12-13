python reason/evaluation/evaluate.py \
    --LM mistral-7b-sft \
    --RM math-shepherd-mistral-7b-prm \
    --task_name MATH \
    --temperature 0.7 \
    --num_sequence 8 \
    --max_new_tokens 2048 \
    --save_dir results \
    --method best_of_n \
    --num_worker 32 \
    --controller_addr http://0.0.0.0:28777


# math-shepherd-mistral-7b-prm
