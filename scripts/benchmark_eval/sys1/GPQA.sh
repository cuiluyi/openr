DATASET_NAME="Idavidrein/gpqa"
DATASET_TAG=$(basename "$DATASET_NAME")

MODEL_BASE="/data/cuiluyi/openr/ckpts"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B/format-sft-s1k-1.1"
MODEL_PATH="${MODEL_BASE}/${MODEL_NAME}"

OUTPUT_BASE="./data/evals"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_NAME="${OUTPUT_BASE}/${MODEL_NAME}/result_benchmark_${DATASET_TAG}_${TIMESTAMP}.json"
# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_NAME")"

SYSTEM_PROMPT="You are a helpful AI Assistant that solves problems using a step-by-step reasoning process. For each problem, you alternate between two stages:\n\n1. <think>: You explain your internal thoughts and planning, like you're thinking to yourself.\n2. <step>: You write down the actual result of the reasoning step, as if solving the problem step by step.\n\nRepeat this <think>/<step> pair until the final answer is reached.\n\nRespond in the following format:\n<think>\n...\n</think>\n<step>\n...\n</step>\n\n<think>\n...\n</think>\n<step>\n...\n</step>\n...\n\nHere is an example of response:\n\n---\n\n<think>\nThe regular hexagon is made up of six equilateral triangles. Each triangle has 3 equal sides.\n</think>\n<step>\nSo one triangle has 3 equal sides that sum to 21 inches, meaning each side is 21 ÷ 3 = 7 inches.\n</step>\n\n<think>\nAll sides of the hexagon are equal to the side of the equilateral triangle.\n</think>\n<step>\nSo the hexagon has 6 sides, each 7 inches long. The perimeter is 6 × 7 = 42 inches.\n</step>\n\n<think>\nFinal answer.\n</think>\n<step>\nThe perimeter of the regular hexagon is \\boxed{42} inches.\n</step>"

# Run benchmark
python ./train/benchmark.py \
    --model_name="$MODEL_PATH" \
    --dtype='bfloat16' \
    --num_gpus=4 \
    --gpu_memory_utilization=0.8 \
    --dataset_name="$DATASET_NAME" \
    --dataset_subset='gpqa_diamond' \
    --dataset_split='train' \
    --question_column='Question' \
    --solution_column='Correct Answer' \
    --system_prompt="$SYSTEM_PROMPT" \
    --output_name="$OUTPUT_NAME" \
    --temperature=0.6 \
    --top_p=0.95 \
    --max_output_tokens=32768 \
    --include_stop_str_in_output