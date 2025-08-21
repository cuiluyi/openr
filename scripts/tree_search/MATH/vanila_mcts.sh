python reason/main.py \
    --LM checkpoint-640 \
    --RM Qwen2.5-Math-PRM-7B \
    --dataset /data/cuiluyi/openr/data/difficulty_data/Qwen2.5-1.5B-Instruct/wrong_data_MATH-openai-split.jsonl \
    --split train \
    --temperature 0.8 \
    --top_p 1 \
    --top_k -1 \
    --max_new_tokens 2048 \
    --num_sequence 16 \
    --max_width 4 \
    --max_steps 20 \
    --save_dir results \
    --method vanilla_mcts \
    --num_worker 100 \
    --controller_addr http://0.0.0.0:28777 \
    --resume_dir /data/cuiluyi/openr/results/vanilla_mcts/wrong_data_MATH-openai-split/20250821_091240
# --LM "Qwen2.5-Math-1.5B-Instruct&s1-20250312_213742&s1-20250314_003214" \