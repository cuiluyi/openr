export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

NUM_GPUS=4
MODEL_BASE=/data/cuiluyi/resources/models
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_ARGS="model_name=$MODEL_BASE/$MODEL_NAME,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=data/evals/$MODEL_NAME
mkdir -p "$OUTPUT_DIR" # Create output directory if it doesn't exist

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/$TASK.log 2>&1


# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/$TASK.log 2>&1

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/$TASK.log 2>&1

# LiveCodeBench
lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/$TASK.log 2>&1