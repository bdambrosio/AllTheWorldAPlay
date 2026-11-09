#!/bin/bash
# Run ATWAP locally against an already-running vLLM server on :5000.
#   image server : FLUX.2-klein on the spare GPU, :5008  (src/utils/klein_serve.py)
#   backend      : uvicorn main:app, :$BACKEND_PORT (8010; :8000 is taken on this box)
#   UI           : React dev server, :3000
# Logs go to ./logs/. Ctrl+C stops everything.
#
# Env overrides: BACKEND_PORT, IMAGE_GPU (nvidia-smi index for images + embeddings, default 0),
#                NO_IMAGES=1 (grey placeholders instead of Klein), NO_TTS=1 (no voices)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8010}"
LOGS="$ROOT/logs"; mkdir -p "$LOGS"

# API keys come from ~/.config/secrets.env, loaded whole (a value there replaces one already
# in the environment), as the other local projects do: a shell that is not interactive does
# not load ~/.bashrc, which is what loads that file for a terminal. None are required for the
# local vLLM setup; ELEVENLABS_API_KEY enables voices. NO_TTS=1 runs without voices.
[ -f "$HOME/.config/secrets.env" ] && { set -a; . "$HOME/.config/secrets.env"; set +a; }
[ "${NO_TTS:-0}" = 1 ] && unset ELEVENLABS_API_KEY

export PATH="$ROOT/.venv/bin:$PATH"
# Keep everything except vLLM off the RTX PRO 6000 (nvidia-smi 1): vLLM runs at
# --gpu-memory-utilization 0.96, and the engine's embedding model defaulting to torch's
# cuda:0 (which is the 6000 in torch's default FASTEST_FIRST order) took ~850 MB of that
# headroom; vLLM's EngineCore crashed twice. All our CUDA users get the spare GPU.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${IMAGE_GPU:-0}"
# Unbuffered so logs/*.log are written in real time (also used by tools/smoke_run.py).
export PYTHONUNBUFFERED=1

if ! curl -sf -m 3 http://localhost:5000/v1/models >/dev/null; then
    echo "WARNING: no vLLM server answering on :5000 — plays will fail until it is up" >&2
fi

pids=()
cleanup() { echo "stopping..."; kill "${pids[@]}" 2>/dev/null || true; pkill -f "sim/simulation.py" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

cd "$ROOT/src/utils"
if [ "${NO_IMAGES:-0}" = 1 ]; then
    python blank_image_serve.py > "$LOGS/image.log" 2>&1 &
else
    python klein_serve.py > "$LOGS/image.log" 2>&1 &
fi
pids+=($!)

cd "$ROOT/src/sim"
uvicorn main:app --port "$BACKEND_PORT" > "$LOGS/backend.log" 2>&1 &
pids+=($!)

cd "$ROOT/src/sim/webworld"
REACT_APP_BACKEND_PORT="$BACKEND_PORT" BROWSER=none npm start > "$LOGS/ui.log" 2>&1 &
pids+=($!)

echo "image :5008, backend :$BACKEND_PORT, UI http://localhost:3000   (logs in $LOGS)"
wait -n
echo "a service exited — see $LOGS" >&2
