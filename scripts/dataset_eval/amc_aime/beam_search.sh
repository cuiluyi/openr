python reason/main.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM Qwen2.5-Math-PRM-7B \
    --dataset /data/cuiluyi/resources/datasets/open-r1/OpenR1-Math-220k \
    --subset all \
    --split train \
    --temperature 0.7 \
    --top_p 0.95 \
    --top_k -1 \
    --max_new_tokens 2048 \
    --num_sequence 1 \
    --max_width 4 \
    --max_steps 20 \
    --save_dir results \
    --method beam_search \
    --num_worker 32 \
    --controller_addr http://0.0.0.0:28777

# --LM "Qwen2.5-Math-1.5B-Instruct&s1-20250312_213742&s1-20250314_003214" \