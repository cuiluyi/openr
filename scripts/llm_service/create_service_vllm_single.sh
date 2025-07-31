set -e

HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30010

echo PYTHON_EXECUTABLE=$(which python)
PYTHON_EXECUTABLE=$(which python)

# MODEL_BASE=/data/cuiluyi/resources/models
MODEL_BASE=/data/cuiluyi/resources

# LANGUAGE_MODEL_NAME=peiyi9979/mistral-7b-sft
# LANGUAGE_MODEL_NAME=Qwen/Qwen2.5-Math-1.5B-Instruct
# LANGUAGE_MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
LANGUAGE_MODEL_NAME=ckpts/DeepSeek-R1-Distill-Qwen-1.5B/slow_fast_reason-sft-s1k-1.1_full/checkpoint-400

# REWARD_MODEL_NAME=peiyi9979/math-shepherd-mistral-7b-prm
# REWARD_MODEL_NAME=Qwen/Math-psa-7B
REWARD_MODEL_NAME=models/Qwen/Qwen2.5-Math-PRM-7B

LANGUAGE_MODEL_PATH=$MODEL_BASE/$LANGUAGE_MODEL_NAME
REWARD_MODEL_PATH=$MODEL_BASE/$REWARD_MODEL_NAME
LOGDIR=logs/fastchat

tmux start-server
tmux new-session -s FastChat -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m reason.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter


echo "Wait 5 seconds ..."
sleep 5


echo "Starting policy-model workers"
WORKER_PORT=$((WORKER_BASE_PORT))
tmux new-window -n policy_worker_0
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "CUDA_VISIBLE_DEVICES=5 $PYTHON_EXECUTABLE -m reason.serve.vllm_worker --model-path $LANGUAGE_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT" Enter


echo "Starting reward-model workers"
WORKER_PORT=$((WORKER_BASE_PORT+1))
tmux new-window -n value_worker_0
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "CUDA_VISIBLE_DEVICES=7 $PYTHON_EXECUTABLE -m reason.serve.reward_model_worker --model-path $REWARD_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT" Enter