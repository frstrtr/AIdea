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
  Total:                  ~7.8 GB  ← would fit, but cross-attn allocs
  push 7.91 GB card over the edge → CUDA OOM at first /render

Mitigation we apply (turn the knobs that have lowest quality cost):
  - ``enable_attention_slicing("auto")`` chunks the [HW × HW] attention
    matrix that OOMed our first run — peak memory drops ~5× without
    the per-step CPU↔GPU transfer overhead (cpu_offload took 14s/step
    on a 1070, 5+ min/image).
  - ``vae.enable_slicing()`` + ``vae.enable_tiling()`` split the
    final latent → 1024² pixel decode the same way.
  - ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` reduces
    allocator fragmentation across inferences.
"""
from __future__ import annotations

import io
import logging
import os
import time
from typing import Optional

# Must be set BEFORE the torch import so the CUDA caching allocator
# picks it up. expandable_segments lets the allocator grow segments
# in place, which avoids fragmentation when activation tensors of
# varying sizes are allocated/freed across diffusion steps.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    (which would be unusable in practice, but it doesn't crash).

    On the GTX 1070 (8 GB) we MUST enable model_cpu_offload() — the
    plain ``.to("cuda")`` path loads the full UNet on-device and then
    OOMs at the first cross-attention activation. cpu_offload swaps
    sub-modules in/out as needed, costing ~3 s/render for the savings.
    """
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
        # The 8 GB GTX 1070 cannot hold the full SDXL model on GPU at
        # once — pipe.to("cuda") takes 7.61 GiB leaving only 300 MiB
        # for activations, which OOMs even at 768². cpu_offload moves
        # sub-modules in/out per pipeline stage, freeing the headroom.
        # Pair with attention slicing + VAE slicing/tiling to keep the
        # per-step working-set within the freed VRAM.
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing("auto")
        try:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            log.info("attention + VAE slicing/tiling enabled")
        except Exception:
            log.warning("VAE slicing/tiling unavailable — large outputs may OOM")
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
