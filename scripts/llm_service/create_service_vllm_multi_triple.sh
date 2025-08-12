set -e

HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30010

echo PYTHON_EXECUTABLE=$(which python)
PYTHON_EXECUTABLE=$(which python)

# MODEL_BASE=/data/cuiluyi/resources/models
# MODEL_BASE=/data/cuiluyi/resources
MODEL_BASE=/data/cuiluyi

# LANGUAGE_MODEL_NAME=peiyi9979/mistral-7b-sft
# LANGUAGE_MODEL_NAME=Qwen/Qwen2.5-Math-1.5B-Instruct
# LANGUAGE_MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
LANGUAGE_MODEL_NAME=openr/ckpts/DeepSeek-R1-Distill-Qwen-1.5B/slow_fast_reason-sft-s1k-1.1_full/checkpoint-400

# REWARD_MODEL_NAME=peiyi9979/math-shepherd-mistral-7b-prm
# REWARD_MODEL_NAME=Qwen/Math-psa-7B
# REWARD_MODEL_NAME=resources/models/Qwen/Qwen2.5-Math-PRM-7B
REWARD_MODEL_NAME=resources/models/Qwen/Qwen2.5-Math-PRM-7B

LANGUAGE_MODEL_PATH=$MODEL_BASE/$LANGUAGE_MODEL_NAME
REWARD_MODEL_PATH=$MODEL_BASE/$REWARD_MODEL_NAME

CUDA_DEVICE_BASE=4
LOGDIR=logs/fastchat

tmux start-server
tmux new-session -s FastChat -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m reason.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter

NUM_LM_WORKER=12
NUM_RM_WORKER=3

echo "Wait 5 seconds ..."
sleep 5

echo "Starting policy-model workers"
for i in $(seq 0 $((NUM_LM_WORKER-1)))
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  tmux new-window -n policy_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  # Two workers share the same GPU: i=0,1 -> GPU0; i=2,3 -> GPU1; etc.
  GPU_ID=$((i / 3 + CUDA_DEVICE_BASE))
  tmux send-keys "CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_EXECUTABLE -m reason.serve.vllm_worker --model-path $LANGUAGE_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --gpu_memory_utilization 0.2" Enter
  echo "start policy_worker_$i"
  sleep 5
done



# echo "Starting reward-model workers"
# for i in $(seq 0 $((NUM_RM_WORKER-1)))
# do
#   WORKER_PORT=$((WORKER_BASE_PORT+NUM_LM_WORKER+i))
#   tmux new-window -n value_worker_$i
#   tmux send-keys "export LOGDIR=${LOGDIR}" Enter
#   # Same sharing logic for reward workers if needed
#   GPU_ID=$(( (i / 2) + NUM_LM_WORKER/3 + CUDA_DEVICE_BASE ))
#   tmux send-keys "CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_EXECUTABLE -m reason.serve.reward_model_worker --model-path $REWARD_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT" Enter
# done
