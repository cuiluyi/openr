set -e

HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30010

echo PYTHON_EXECUTABLE=$(which python)
PYTHON_EXECUTABLE=$(which python)

POLICY_MODEL_BASE=/data/cuiluyi/resources/models/peiyi9979
VALUE_MODEL_BASE=/data/cuiluyi/resources/models/openreasoner/Math-psa

CUDA_DEVICE_BASE=2
POLICY_MODEL_NAME=mistral-7b-sft
VALUE_MODEL_NAME=checkpoint-2127
MODEL_PATH=$POLICY_MODEL_BASE/$POLICY_MODEL_NAME
VALUE_MODEL_PATH=$VALUE_MODEL_BASE/$VALUE_MODEL_NAME

LOGDIR=logs_fastchat

tmux start-server
tmux new-session -s FastChat -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter

NUM_LM_WORKER=1
NUM_RM_WORKER=1

echo "Wait 10 seconds ..."
sleep 5

echo "Starting policy-model workers"
for i in $(seq 0 $((NUM_LM_WORKER-1)))
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  tmux new-window -n policy_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  # tmux send-keys "CUDA_VISIBLE_DEVICES=$((i+CUDA_DEVICE_BASE)) $PYTHON_EXECUTABLE -m reason.llm_service.workers.vllm_worker --model-path $MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --dtype bfloat16 --swap-space 32" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=0 $PYTHON_EXECUTABLE -m reason.llm_service.workers.vllm_worker --model-path $MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --dtype bfloat16 --swap-space 32" Enter
done


echo "Starting reward-model workers"
for i in $(seq 0 $((NUM_RM_WORKER-1)))
do
  WORKER_PORT=$((i+WORKER_BASE_PORT+NUM_LM_WORKER))
  tmux new-window -n value_worker
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  # tmux send-keys "CUDA_VISIBLE_DEVICES=$((i+NUM_LM_WORKER+CUDA_DEVICE_BASE)) $PYTHON_EXECUTABLE -m reason.llm_service.workers.reward_model_worker --model-path $VALUE_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=5 $PYTHON_EXECUTABLE -m reason.llm_service.workers.reward_model_worker --model-path $VALUE_MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT" Enter
done