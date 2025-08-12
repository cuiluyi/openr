DATASET_NAME="/data/cuiluyi/resources/datasets/HuggingFaceH4/aime_2024"
DATASET_TAG=$(basename "$DATASET_NAME")

MODEL_BASE="/data/cuiluyi/resources/models"
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_PATH="${MODEL_BASE}/${MODEL_NAME}"

OUTPUT_BASE="./data/evals"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_NAME="${OUTPUT_BASE}/${MODEL_NAME}/result_benchmark_${DATASET_TAG}_${TIMESTAMP}.json"
# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_NAME")"

# Run benchmark
python ./train/benchmark.py \
    --model_name="$MODEL_PATH" \
    --dtype='bfloat16' \
    --num_gpus=4 \
    --gpu_memory_utilization=0.8 \
    --dataset_name="$DATASET_NAME" \
    --dataset_split='train' \
    --question_column='problem' \
    --solution_column='answer' \
    --output_name="$OUTPUT_NAME" \
    --temperature=0.6 \
    --top_p=0.95 \
    --max_output_tokens=32768 \
    --include_stop_str_in_output
