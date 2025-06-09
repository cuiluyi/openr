HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777

export LOGDIR=logs/fastchat

python -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR
