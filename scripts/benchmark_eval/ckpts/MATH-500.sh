DATASET_NAME="HuggingFaceH4/MATH-500"
DATASET_TAG=$(basename "$DATASET_NAME")

MODEL_BASE="/data/cuiluyi/openr/ckpts"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/format-sft-s1k-1.1"

# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-filtered_amc_aime_batchsize32/checkpoint-47"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-wrong_data_amc_aime_batchsize32/checkpoint-119"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-wrong_data_amc_aime_batchsize32/checkpoint-297"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-wrong_data_amc_aime_batchsize32/checkpoint-594"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-wrong_data_amc_aime_batchsize32/checkpoint-891"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/dpo1-sft2-wrong_data_amc_aime_batchsize32/checkpoint-839"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/dpo1-sft2-wrong_data_amc_aime_batchsize32/checkpoint-1678"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/dpo1-sft2-wrong_data_amc_aime_batchsize32/checkpoint-2517"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-optim_paths-wrong_data_amc_aime_batchsize32/checkpoint-119"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-optim_paths-wrong_data_amc_aime_batchsize32/checkpoint-238"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-optim_paths-wrong_data_amc_aime_batchsize32/checkpoint-357"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-optim_paths-wrong_data_amc_aime_batchsize32/checkpoint-476"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-optim_paths-wrong_data_amc_aime_batchsize32/checkpoint-595"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-optim_paths-wrong_data_amc_aime_batchsize32/checkpoint-714"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-optim_paths-wrong_data_amc_aime_batchsize32/checkpoint-833"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-optim_paths-wrong_data_amc_aime_batchsize32/checkpoint-952"
# MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-wrong_data_amc_aime_batchsize32/checkpoint-1188"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-wrong_data_amc_aime_batchsize32/checkpoint-1485"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-wrong_data_amc_aime_batchsize32/checkpoint-1782"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-wrong_data_amc_aime_batchsize32/checkpoint-2079"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/sft2-wrong_data_amc_aime_batchsize32/checkpoint-2376"

MODEL_PATH="${MODEL_BASE}/${MODEL_NAME}"

OUTPUT_BASE="./data/evals"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_NAME="${OUTPUT_BASE}/${MODEL_NAME}/result_benchmark_${DATASET_TAG}_${TIMESTAMP}.json"
# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_NAME")"

# SYSTEM_PROMPT="You are a helpful AI Assistant that solves problems using a step-by-step reasoning process. For each problem, you alternate between two stages:\n\n1. <think>: You explain your internal thoughts and planning, like you're thinking to yourself.\n2. <step>: You write down the actual result of the reasoning step, as if solving the problem step by step.\n\nRepeat this <think>/<step> pair until the final answer is reached.\n\nRespond in the following format:\n<think>\n...\n</think>\n<step>\n...\n</step>\n\n<think>\n...\n</think>\n<step>\n...\n</step>\n...\n\nHere is an example of response:\n\n---\n\n<think>\nThe regular hexagon is made up of six equilateral triangles. Each triangle has 3 equal sides.\n</think>\n<step>\nSo one triangle has 3 equal sides that sum to 21 inches, meaning each side is 21 ÷ 3 = 7 inches.\n</step>\n\n<think>\nAll sides of the hexagon are equal to the side of the equilateral triangle.\n</think>\n<step>\nSo the hexagon has 6 sides, each 7 inches long. The perimeter is 6 × 7 = 42 inches.\n</step>\n\n<think>\nFinal answer.\n</think>\n<step>\nThe perimeter of the regular hexagon is \\boxed{42} inches.\n</step>"
SYSTEM_PROMPT="Please reason step by step, and put your final answer within \\boxed{}."

# Run benchmark
CUDA_VISIBLE_DEVICES="7" python ./scripts/benchmark.py \
    --model_name="$MODEL_PATH" \
    --dtype='bfloat16' \
    --num_gpus=1 \
    --gpu_memory_utilization=0.8 \
    --dataset_name="$DATASET_NAME" \
    --dataset_split='test' \
    --question_column='problem' \
    --solution_column='solution' \
    --system_prompt="$SYSTEM_PROMPT" \
    --output_name="$OUTPUT_NAME" \
    --temperature=0.6 \
    --top_p=0.95 \
    --max_output_tokens=32768 \
    --include_stop_str_in_output
