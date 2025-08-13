export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

NUM_GPUS=4
MODEL_BASE=/data/cuiluyi/openr/ckpts
MODEL_NAME=DeepSeek-R1-Distill-Qwen-1.5B/format-sft-s1k-1.1
MODEL_ARGS="model_name=$MODEL_BASE/$MODEL_NAME,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=data/evals/$MODEL_NAME
mkdir -p "$OUTPUT_DIR" # Create output directory if it doesn't exist

SYSTEM_PROMPT="You are a helpful AI Assistant that solves problems using a step-by-step reasoning process. For each problem, you alternate between two stages:\n\n1. <think>: You explain your internal thoughts and planning, like you're thinking to yourself.\n2. <step>: You write down the actual result of the reasoning step, as if solving the problem step by step.\n\nRepeat this <think>/<step> pair until the final answer is reached.\n\nRespond in the following format:\n<think>\n...\n</think>\n<step>\n...\n</step>\n\n<think>\n...\n</think>\n<step>\n...\n</step>\n...\n\nHere is an example of response:\n\n---\n\n<think>\nThe regular hexagon is made up of six equilateral triangles. Each triangle has 3 equal sides.\n</think>\n<step>\nSo one triangle has 3 equal sides that sum to 21 inches, meaning each side is 21 ÷ 3 = 7 inches.\n</step>\n\n<think>\nAll sides of the hexagon are equal to the side of the equilateral triangle.\n</think>\n<step>\nSo the hexagon has 6 sides, each 7 inches long. The perimeter is 6 × 7 = 42 inches.\n</step>\n\n<think>\nFinal answer.\n</think>\n<step>\nThe perimeter of the regular hexagon is \\boxed{42} inches.\n</step>"

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --system-prompt "$SYSTEM_PROMPT" \
    > $OUTPUT_DIR/$TASK.log 2>&1

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --system-prompt "$SYSTEM_PROMPT" \
    > $OUTPUT_DIR/$TASK.log 2>&1

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --system-prompt "$SYSTEM_PROMPT" \
    > $OUTPUT_DIR/$TASK.log 2>&1

# LiveCodeBench
lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --system-prompt "$SYSTEM_PROMPT" \
    > $OUTPUT_DIR/$TASK.log 2>&1