python -m debugpy --listen 63655 --wait-for-client \
    reason/main.py \
    --LM checkpoint-400 \
    --RM Qwen2.5-Math-PRM-7B \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.8 \
    --top_p 1 \
    --top_k -1 \
    --max_new_tokens 5120 \
    --num_sequence 4 \
    --max_width 4 \
    --max_steps 20 \
    --save_dir results \
    --method vanilla_mcts \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    --local

# --LM "Qwen2.5-Math-1.5B-Instruct&s1-20250312_213742&s1-20250314_003214" \
# Tips: Make sure the input (--LM, --RM) in the script aligns with the command output (basename $LANGUAGE_MODEL_NAME, basename $REWARD_MODEL_NAME) in the pending worker!
# if you debug into the ray code, you must set the option '--local'