# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 2 \
#     --tree_max_width 9 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     --resume_dir /data/cuiluyi/openr/results/w_correct_filter/MATH/beam_search/20250508_122030

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 4 \
#     --tree_max_width 3 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 6 \
#     --tree_max_width 3 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 8 \
#     --tree_max_width 3 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local


# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 10 \
#     --tree_max_width 3 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local

python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM Qwen2.5-Math-PRM-7B \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.7 \
    --top_p 0.95 \
    --top_k -1 \
    --max_new_tokens 2048 \
    --num_sequence 12 \
    --tree_max_width 3 \
    --tree_max_depth 50 \
    --save_dir results/w_correct_filter \
    --method beam_search \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    --resume_dir /data/cuiluyi/openr/results/w_correct_filter/MATH/beam_search/20250512_224840
    # --local

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 2 \
#     --tree_max_width 3 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 2 \
#     --tree_max_width 6 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 2 \
#     --tree_max_width 9 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local


# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 2 \
#     --tree_max_width 12 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 2 \
#     --tree_max_width 15 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 4 \
#     --tree_max_width 3 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 4 \
#     --tree_max_width 6 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 4 \
#     --tree_max_width 9 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local


# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 4 \
#     --tree_max_width 12 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local

# python reason/evaluation/evaluate.py \
#     --LM Qwen2.5-Math-1.5B-Instruct \
#     --RM Qwen2.5-Math-PRM-7B \
#     --task_name MATH \
#     --dataset HuggingFaceH4/MATH-500 \
#     --temperature 0.7 \
#     --top_p 0.95 \
#     --top_k -1 \
#     --max_new_tokens 2048 \
#     --num_sequence 4 \
#     --tree_max_width 15 \
#     --tree_max_depth 50 \
#     --save_dir results/w_correct_filter \
#     --method beam_search \
#     --num_worker 16 \
#     --controller_addr http://0.0.0.0:28777 \
#     # --local
# # --LM "Qwen2.5-Math-1.5B-Instruct&s1-20250312_213742&s1-20250314_003214" \
# # Tips: Make sure the input (--LM, --RM) in the script aligns with the command output (basename $LANGUAGE_MODEL_NAME, basename $REWARD_MODEL_NAME) in the pending worker!
# # if you debug into the ray code, you must set the option '--local'