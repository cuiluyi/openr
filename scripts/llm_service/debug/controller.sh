HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777

export LOGDIR=logs/fastchat

# python -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR

python -m debugpy --listen 63655 --wait-for-client /data/cuiluyi/openr/reason/serve/controller.py --port ${CONTROLER_PORT} --host $HOST_ADDR