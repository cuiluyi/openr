HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30010

MODEL_BASE=/data/cuiluyi/resources/models

# LANGUAGE_MODEL_NAME=peiyi9979/mistral-7b-sft
LANGUAGE_MODEL_NAME=Qwen/Qwen2.5-Math-1.5B-Instruct

LANGUAGE_MODEL_PATH=$MODEL_BASE/$LANGUAGE_MODEL_NAME
export LOGDIR=logs/fastchat

WORKER_PORT=$((WORKER_BASE_PORT))
export CUDA_VISIBLE_DEVICES=0
# You can also add --dtype bfloat16, --swap-space 32, --gpu-memory-utilization, etc. For all options, see the help message of `python reason.llm_service.workers.vllm_worker -h`
python -m debugpy --listen 63655 --wait-for-client -m reason.llm_service.workers.vllm_worker --model-path $LANGUAGE_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT
