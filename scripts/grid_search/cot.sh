python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM Qwen2.5-Math-PRM-7B \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0 \
    --top_p 1 \
    --top_k -1 \
    --num_sequence 3 \
    --max_new_tokens 2048 \
    --save_dir results/grid_search \
    --method best_of_n \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    # --local

python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM Qwen2.5-Math-PRM-7B \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.2 \
    --top_p 1 \
    --top_k -1 \
    --num_sequence 3 \
    --max_new_tokens 2048 \
    --save_dir results/grid_search \
    --method best_of_n \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    # --local

python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM Qwen2.5-Math-PRM-7B \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.4 \
    --top_p 1 \
    --top_k -1 \
    --num_sequence 3 \
    --max_new_tokens 2048 \
    --save_dir results/grid_search \
    --method best_of_n \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    # --local

python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM Qwen2.5-Math-PRM-7B \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.6 \
    --top_p 1 \
    --top_k -1 \
    --num_sequence 3 \
    --max_new_tokens 2048 \
    --save_dir results/grid_search \
    --method best_of_n \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    # --local

python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM Qwen2.5-Math-PRM-7B \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.8 \
    --top_p 1 \
    --top_k -1 \
    --num_sequence 3 \
    --max_new_tokens 2048 \
    --save_dir results/grid_search \
    --method best_of_n \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    # --local

python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM Qwen2.5-Math-PRM-7B \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 1.0 \
    --top_p 1 \
    --top_k -1 \
    --num_sequence 3 \
    --max_new_tokens 2048 \
    --save_dir results/grid_search \
    --method best_of_n \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    # --local

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.0 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 20480 \
#     --save_dir results/grid_search \
#     --method cot \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.2 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 20480 \
#     --save_dir results/grid_search \
#     --method cot \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.4 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 20480 \
#     --save_dir results/grid_search \
#     --method cot \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.6 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 20480 \
#     --save_dir results/grid_search \
#     --method cot \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.8 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 20480 \
#     --save_dir results/grid_search \
#     --method cot \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 1.0 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 20480 \
#     --save_dir results/grid_search \
#     --method cot \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local

# # --LM "Qwen2.5-Math-1.5B-Instruct&s1-20250312_213742&s1-20250314_003214" \
# # Tips: Make sure the input (--LM, --RM) in the script aligns with the command output (basename $LANGUAGE_MODEL_NAME, basename $REWARD_MODEL_NAME) in the pending worker!
# # if you debug into the ray code, you must set the option '--local'