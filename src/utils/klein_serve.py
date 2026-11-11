"""Shared local text-to-image server: FLUX.2-klein-4B on the spare GPU.

Speaks two APIs so one process can serve both apps:
  - AllTheWorldAPlay:     GET  /generate_image?prompt=...&size=WxH   -> image/png
  - Cognitive_workbench:  POST /generate {prompt, steps, height, width, seed}
                          GET  /healthz
    (point the workbench at it with GENERATE_IMAGE_URL=http://localhost:5008)

Run from src/utils:   python klein_serve.py            (port 5008)
Env:
  IMAGE_GPU      nvidia-smi index of the card to use (default 0, the 5060 Ti;
                 the RTX PRO 6000 holds the live vLLM)
  IMAGE_PORT     listen port (default 5008, the port ATWAP calls)
  KLEIN_MODEL    HF model id (default black-forest-labs/FLUX.2-klein-4B)
  KLEIN_OFFLOAD  "textcpu" (default: transformer+VAE resident on GPU, ~8GB; text
                 encoder on CPU; ~2.5s/image), "model" (diffusers CPU offload,
                 least VRAM, ~8s/image) or "none" (all on GPU, needs ~17GB+)
"""
import os

# Pin to the spare card BEFORE torch initialises CUDA. torch's default
# FASTEST_FIRST order reverses nvidia-smi on this box, so cuda:0 would be the
# vLLM card; PCI_BUS_ID makes the index match nvidia-smi. Both must be set.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("IMAGE_GPU", "0"))

import io
import random
import threading
import time

import torch
import uvicorn
from diffusers import Flux2KleinPipeline
from fastapi import FastAPI, HTTPException, Response
from PIL import Image
from pydantic import BaseModel

MODEL_ID = os.environ.get("KLEIN_MODEL", "black-forest-labs/FLUX.2-klein-4B")
OFFLOAD = os.environ.get("KLEIN_OFFLOAD", "textcpu")
PORT = int(os.environ.get("IMAGE_PORT", "5008"))
DEFAULT_STEPS = 4      # klein is distilled for 4 steps
MIN_SIDE = 256         # generate at least this big, then downscale small requests
MAX_SIDE = 1024

print(f"klein_serve: loading {MODEL_ID} on {torch.cuda.get_device_name(0)} (offload={OFFLOAD})")
pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
_text_encoder = None
if OFFLOAD == "none":
    pipe.to("cuda")
elif OFFLOAD == "textcpu":
    # Klein's Qwen3-4B text encoder (~8GB) and transformer (~8GB) don't both fit
    # on 16GB, and per-call CPU offload shuttles ~16GB each image. Keep the
    # transformer+VAE resident and run the one-shot text encode on CPU instead.
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    _text_encoder = pipe.text_encoder
    pipe.text_encoder = None   # so the pipeline's execution device resolves to cuda
else:
    pipe.enable_model_cpu_offload()


def _encode(prompt: str):
    """Text-encode on CPU (textcpu mode); returns embeddings on the GPU."""
    pipe.text_encoder = _text_encoder
    try:
        out = pipe.encode_prompt(prompt, device="cpu")
    finally:
        pipe.text_encoder = None
    embeds = out[0] if isinstance(out, tuple) else out
    return embeds.to("cuda")

_lock = threading.Lock()   # one pipeline, one generation at a time
app = FastAPI()


def _gen_side(n: int) -> int:
    """Clamp to [MIN_SIDE, MAX_SIDE] and round up to a multiple of 16 (FLUX latent grid)."""
    n = max(MIN_SIDE, min(MAX_SIDE, n))
    return (n + 15) // 16 * 16


def _render(prompt: str, width: int, height: int, steps: int = DEFAULT_STEPS, seed=None):
    gw, gh = _gen_side(width), _gen_side(height)
    seed = random.randint(0, 2**32 - 1) if seed is None else int(seed)
    t0 = time.time()
    with _lock:
        text = ({"prompt_embeds": _encode(prompt)} if _text_encoder is not None
                else {"prompt": prompt})
        img = pipe(**text, width=gw, height=gh, guidance_scale=1.0,
                   num_inference_steps=steps,
                   generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]
    if (gw, gh) != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), time.time() - t0


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_ID}


@app.get("/generate_image")   # AllTheWorldAPlay
def generate_image(prompt: str, size: str = "512x512", seed: int | None = None):
    try:
        w, h = (int(v) for v in size.lower().split("x"))
    except ValueError:
        raise HTTPException(400, f"bad size {size!r}, expected WxH")
    png, secs = _render(prompt, w, h, seed=seed)
    return Response(png, media_type="image/png", headers={"x-wall-seconds": f"{secs:.2f}"})


class GenerateReq(BaseModel):   # Cognitive_workbench (Bonsai-compatible subset)
    prompt: str
    steps: int = DEFAULT_STEPS
    height: int = 512
    width: int = 512
    seed: int | None = None
    backend: str | None = None   # accepted and ignored


@app.post("/generate")
def generate(req: GenerateReq):
    png, secs = _render(req.prompt, req.width, req.height, req.steps, req.seed)
    return Response(png, media_type="image/png", headers={"x-wall-seconds": f"{secs:.2f}"})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
