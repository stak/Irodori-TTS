#!/bin/bash
# Start the gradio server in WSL for verification and wait until it is ready.
# Usage: start_wsl_server.sh <port> [extra env as KEY=VALUE ...]
set -u
PORT="${1:?port required}"
shift
cd /mnt/c/Users/stak/irodori/Irodori-TTS

if curl -s -o /dev/null "http://127.0.0.1:${PORT}/config"; then
    echo "ALREADY_UP"
    exit 0
fi

for kv in "$@"; do
    export "$kv"
done

LOG="/tmp/gradio_${PORT}.log"
nohup /home/stak/venvs/irodori-tts/bin/python -u gradio_app.py \
    --server-name 127.0.0.1 --server-port "${PORT}" > "${LOG}" 2>&1 &
PID=$!
echo "spawned pid=${PID} log=${LOG}"

for i in $(seq 1 120); do
    sleep 5
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "SERVER_EXITED_EARLY"
        tail -20 "${LOG}"
        exit 1
    fi
    if curl -s -o /dev/null "http://127.0.0.1:${PORT}/config"; then
        echo "UP after $((i * 5))s"
        exit 0
    fi
done
echo "NEVER_UP"
tail -20 "${LOG}"
exit 1
