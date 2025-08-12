NUM_GPUS=4
MODEL=/data/cuiluyi/openr/data/DeepSeek-R1-Distill-Qwen-1.5B-DPO-1-wrong-dataset
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=data/evals/$(basename $MODEL)

# AIME 2024
TASK=aime24
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR



NUM_GPUS=4
MODEL=/data/cuiluyi/openr/data/DeepSeek-R1-Distill-Qwen-1.5B-DPO-123-save_steps/checkpoint-100
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$(basename $MODEL)

# AIME 2024
TASK=aime24
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR



NUM_GPUS=4
MODEL=/data/cuiluyi/openr/data/DeepSeek-R1-Distill-Qwen-1.5B-DPO-123-save_steps/checkpoint-150
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$(basename $MODEL)

# AIME 2024
TASK=aime24
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR




NUM_GPUS=4
MODEL=/data/cuiluyi/openr/data/DeepSeek-R1-Distill-Qwen-1.5B-DPO-123-save_steps/checkpoint-200
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$(basename $MODEL)

# AIME 2024
TASK=aime24
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR





NUM_GPUS=4
MODEL=/data/cuiluyi/openr/data/DeepSeek-R1-Distill-Qwen-1.5B-DPO-123-save_steps/checkpoint-250
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$(basename $MODEL)

# AIME 2024
TASK=aime24
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR





NUM_GPUS=4
MODEL=/data/cuiluyi/openr/data/DeepSeek-R1-Distill-Qwen-1.5B-DPO-123-save_steps/checkpoint-283
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$(basename $MODEL)

# AIME 2024
TASK=aime24
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
VLLM_WORKER_MULTIPROC_METHOD=spawn lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR