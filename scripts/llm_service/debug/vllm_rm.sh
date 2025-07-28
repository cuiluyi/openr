HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30015

MODEL_BASE=/data/cuiluyi/resources/models

# REWARD_MODEL_NAME=peiyi9979/math-shepherd-mistral-7b-prm
# REWARD_MODEL_NAME=Qwen/Math-psa-7B
REWARD_MODEL_NAME=Qwen/Qwen2.5-Math-PRM-7B

REWARD_MODEL_PATH=$MODEL_BASE/$REWARD_MODEL_NAME

export LOGDIR=logs/fastchat

echo "Starting reward-model workers"
WORKER_PORT=$((WORKER_BASE_PORT + 1))
python -m debugpy --listen 63655 --wait-for-client -m reason.llm_service.workers.reward_model_worker --model-path $REWARD_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT
