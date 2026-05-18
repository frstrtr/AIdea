"""Minimal SDXL inference HTTP server for AIdea Brew illustration.

Runs on the GPU host (default: 192.168.86.22) — loads Juggernaut XL v9
once at startup, exposes POST /render that takes a prompt and returns
PNG bytes. AIdea's brew_image.request_image() is the client.

Deployment on the GPU host:

    # one-time:
    /home/user0/aidea-gpu-venv/bin/pip install \
        diffusers accelerate transformers safetensors \
        fastapi 'uvicorn[standard]' Pillow

    # download Juggernaut XL on first run (resolves automatically):
    /home/user0/aidea-gpu-venv/bin/python -c "from diffusers import \
        StableDiffusionXLPipeline as P; P.from_pretrained( \
        'RunDiffusion/Juggernaut-XL-v9', use_safetensors=True)"

    # systemd user unit at ~/.config/systemd/user/aidea-gpu.service:
    [Unit]
    Description=AIdea SDXL inference server
    After=network-online.target
    [Service]
    Type=simple
    Environment=AIDEA_GPU_MODEL_ID=RunDiffusion/Juggernaut-XL-v9
    Environment=AIDEA_GPU_PORT=8765
    ExecStart=/home/user0/aidea-gpu-venv/bin/python \
        /home/user0/aidea-gpu-server.py
    Restart=on-failure
    [Install]
    WantedBy=default.target

    systemctl --user enable --now aidea-gpu
    loginctl enable-linger user0   # keep service alive without login

VRAM math for the GTX 1070 (8 GB):
  - SDXL UNet fp16:       ~5.0 GB
  - VAE + text encoders:  ~1.8 GB
  - Inference activations: ~1.0 GB
  Total:                  ~7.8 GB  ← tight but fits without offload

If we ever load a 1.5×-larger checkpoint we enable
``enable_model_cpu_offload()`` to swap UNet → CPU between inferences
(adds ~3 s/call but unblocks bigger models). For now, plain CUDA is
the simplest fast path.
"""
from __future__ import annotations

import io
import logging
import os
import time
from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("aidea.gpu")

# Bumpable via env so we can swap models without code changes.
MODEL_ID = os.environ.get("AIDEA_GPU_MODEL_ID", "RunDiffusion/Juggernaut-XL-v9")
PORT = int(os.environ.get("AIDEA_GPU_PORT", "8765"))
HOST = os.environ.get("AIDEA_GPU_HOST", "0.0.0.0")
DEFAULT_NEG = (
    "text, letters, watermark, signature, words, numbers, "
    "chart, graph, low quality, blurry, jpeg artifacts, deformed, "
    "ugly, bad anatomy, extra fingers"
)


class RenderRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: Optional[str] = None
    width: int = Field(1024, ge=512, le=1536)
    height: int = Field(1024, ge=512, le=1536)
    steps: int = Field(28, ge=8, le=60)
    guidance: float = Field(6.5, ge=1.0, le=15.0)
    seed: Optional[int] = None


def _load_pipeline() -> StableDiffusionXLPipeline:
    """Load SDXL once. Picks fp16 on CUDA, falls back to fp32 on CPU
    (which would be unusable in practice, but it doesn't crash)."""
    log.info("loading pipeline: %s", MODEL_ID)
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        variant="fp16" if torch.cuda.is_available() else None,
        add_watermarker=False,
    )
    if torch.cuda.is_available():
        pipe.to("cuda")
        # Memory-efficient attention if xformers is available; otherwise
        # PyTorch's SDPA is good enough.
        try:
            pipe.enable_xformers_memory_efficient_attention()
            log.info("xformers attention enabled")
        except Exception:
            log.info("xformers unavailable — using torch SDPA")
    pipe.set_progress_bar_config(disable=True)
    log.info("pipeline loaded in %.1fs", time.time() - t0)
    return pipe


# Build the FastAPI app and load the pipeline at startup.
app = FastAPI(title="AIdea SDXL Inference", version="0.1")
PIPE: StableDiffusionXLPipeline | None = None


@app.on_event("startup")
async def _on_startup() -> None:
    global PIPE
    PIPE = _load_pipeline()


@app.get("/health")
def health() -> dict:
    return {
        "ok": PIPE is not None,
        "model_id": MODEL_ID,
        "cuda": torch.cuda.is_available(),
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu"
        ),
        "vram_free_mib": (
            int(torch.cuda.mem_get_info()[0] / (1024 * 1024))
            if torch.cuda.is_available() else None
        ),
    }


@app.post("/render")
def render(req: RenderRequest) -> Response:
    if PIPE is None:
        raise HTTPException(503, "pipeline not loaded yet")
    seed = req.seed if req.seed is not None else int(time.time() * 1000) & 0x7fffffff
    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).manual_seed(seed)
    t0 = time.time()
    out = PIPE(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt or DEFAULT_NEG,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance,
        width=req.width,
        height=req.height,
        generator=generator,
    )
    dt = time.time() - t0
    img = out.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    bytes_ = buf.getvalue()
    log.info(
        "render: %dx%d, %d steps, seed=%d, %.1fs, %d KB",
        req.width, req.height, req.steps, seed, dt, len(bytes_) // 1024,
    )
    return Response(content=bytes_, media_type="image/png", headers={
        "X-Inference-Sec": f"{dt:.2f}",
        "X-Seed": str(seed),
    })


if __name__ == "__main__":
    import uvicorn
    log.info("serving on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
