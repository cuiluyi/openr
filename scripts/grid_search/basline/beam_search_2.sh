


python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM math-shepherd-mistral-7b-prm \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.7 \
    --top_p 0.95 \
    --top_k -1 \
    --max_new_tokens 2048 \
    --num_sequence 1 \
    --tree_max_width 3 \
    --tree_max_depth 50 \
    --save_dir results/grid_search \
    --method beam_search \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \

python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM math-shepherd-mistral-7b-prm \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.7 \
    --top_p 0.95 \
    --top_k -1 \
    --max_new_tokens 2048 \
    --num_sequence 1 \
    --tree_max_width 6 \
    --tree_max_depth 50 \
    --save_dir results/grid_search \
    --method beam_search \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    # --local

python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM math-shepherd-mistral-7b-prm \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.7 \
    --top_p 0.95 \
    --top_k -1 \
    --max_new_tokens 2048 \
    --num_sequence 1 \
    --tree_max_width 9 \
    --tree_max_depth 50 \
    --save_dir results/grid_search \
    --method beam_search \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    # --local


python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM math-shepherd-mistral-7b-prm \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.7 \
    --top_p 0.95 \
    --top_k -1 \
    --max_new_tokens 2048 \
    --num_sequence 1 \
    --tree_max_width 12 \
    --tree_max_depth 50 \
    --save_dir results/grid_search \
    --method beam_search \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    # --local

python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --RM math-shepherd-mistral-7b-prm \
    --task_name MATH \
    --dataset HuggingFaceH4/MATH-500 \
    --temperature 0.7 \
    --top_p 0.95 \
    --top_k -1 \
    --max_new_tokens 2048 \
    --num_sequence 1 \
    --tree_max_width 15 \
    --tree_max_depth 50 \
    --save_dir results/grid_search \
    --method beam_search \
    --num_worker 16 \
    --controller_addr http://0.0.0.0:28777 \
    # --local