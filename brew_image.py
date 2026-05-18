"""Image-illustration helper for Brew cards.

Two roles in the Brew pipeline:

  1. ``generate_image_prompt(idea_fields, model)`` — one cheap Claude
     call that translates the synthesised idea (title + pitch +
     mechanism) into a single 2-3 sentence VISUAL SCENE description
     suitable for an image-gen model. No text in the image, single
     focal subject, atmospheric, photo-realistic — see PROMPT_SYSTEM
     for the constraints we enforce.

  2. ``request_image(scene_prompt, host_url, output_path)`` — HTTP
     POST to a remote inference server (currently the SDXL +
     Juggernaut XL host at ``AIDEA_BREW_IMAGE_HOST``, default
     ``http://192.168.86.22:8765``). Body shape is ``{"prompt": str,
     "negative_prompt": str | None, "seed": int | None,
     "width": int, "height": int}`` → response is the raw PNG bytes.

The image is then composited with the existing Pillow text panel via
``brew_render.render_card_with_image`` for the final 1080×1920 card.

Env switches (read at use site, not import time):

  AIDEA_BREW_IMAGE_ENABLE   = 1 to turn on illustrated Brew
  AIDEA_BREW_IMAGE_HOST     = http://192.168.86.22:8765 (default)
  AIDEA_BREW_IMAGE_TIMEOUT  = seconds, default 90 — generous because
                              SDXL inference on a 1070 is 5-10s but
                              first-call cold-start is 30-60s
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx


_DEFAULT_HOST = "http://192.168.86.22:8765"
_DEFAULT_TIMEOUT = 90.0


# ---------------------------------------------------------------------------
# Stage 1 — translate idea to image prompt via Claude.
# ---------------------------------------------------------------------------


PROMPT_SYSTEM = (
    "You translate a thinking-tool's idea into a single visual scene "
    "description for an image-generation model. Output ONE 2-3 sentence "
    "scene paragraph. No commentary, no preamble, no markdown.\n\n"
    "HARD CONSTRAINTS:\n"
    "  • No text or letters in the scene. No charts, no signs, no logos.\n"
    "  • Single focal subject. Concrete physical object or environment "
    "    that metaphorically embodies the idea's mechanism — not a "
    "    literal illustration of the idea's words.\n"
    "  • Cinematic / photo-realistic style. Specific lighting "
    "    (golden hour, overcast, candle, fluorescent, etc.).\n"
    "  • Avoid people unless the mechanism specifically requires a "
    "    human action — anatomy is unreliable in image models.\n"
    "  • Keep it tangible: brass clock, frayed rope, dewdrop, "
    "    abandoned greenhouse — NOT abstract concepts like 'innovation' "
    "    or 'flow.'\n"
)


PROMPT_USER_TEMPLATE = """\
The thinking tool produced this idea:

  Title:     {title}
  Pitch:     {pitch}
  Mechanism: {mechanism}

Describe ONE concrete visual scene that metaphorically embodies the
mechanism. 2-3 sentences. No text in the scene. Cinematic, specific.
Begin:"""


async def generate_image_prompt(
    idea_fields: dict,
    model: str = "claude-opus-4-7",
) -> str:
    """Single LLM call. Returns a scene description usable as an image-gen
    prompt. Falls back to a stringified title if the call fails — the
    image still renders something, just less aligned to the mechanism."""
    title = (idea_fields.get("title") or "").strip()
    pitch = (idea_fields.get("pitch") or "").strip()
    mechanism = (idea_fields.get("mechanism") or "").strip()
    if not (title or pitch or mechanism):
        return "Empty scene — soft grey gradient, no objects."

    prompt = PROMPT_USER_TEMPLATE.format(
        title=title or "(no title)",
        pitch=pitch or "(no pitch)",
        mechanism=mechanism or "(no mechanism)",
    )
    try:
        # Reuse AIdea's wired Claude path. _query_text is private-ish but
        # stable enough for tooling — same convention as distill/.
        from aidea import _query_text
        raw = await _query_text(
            prompt, PROMPT_SYSTEM, model, kind="brew_image_prompt",
        )
        scene = (raw or "").strip()
        # Strip any stray markdown / quoting if the model added it.
        scene = scene.strip("`").strip('"').strip("'").strip()
        if scene:
            return scene
    except Exception:
        pass
    # Fallback — works but loses metaphor.
    return f"A still-life scene representing the concept: {title}. Cinematic light, photo-realistic."


# ---------------------------------------------------------------------------
# Stage 2 — request the image from the GPU host's HTTP inference server.
# ---------------------------------------------------------------------------


def _host_url() -> str:
    return os.environ.get("AIDEA_BREW_IMAGE_HOST", _DEFAULT_HOST).rstrip("/")


def _timeout() -> float:
    try:
        return float(os.environ.get("AIDEA_BREW_IMAGE_TIMEOUT", _DEFAULT_TIMEOUT))
    except ValueError:
        return _DEFAULT_TIMEOUT


def is_enabled() -> bool:
    """``AIDEA_BREW_IMAGE_ENABLE`` is read at use site so flipping the env
    var on the LXC restarts cleanly without code changes."""
    v = os.environ.get("AIDEA_BREW_IMAGE_ENABLE", "").strip().lower()
    return v not in ("", "0", "false", "no", "off")


async def request_image(
    scene_prompt: str,
    output_path: Path,
    *,
    width: int = 1024,
    height: int = 1024,
    negative_prompt: str | None = (
        "text, letters, watermark, signature, words, numbers, "
        "chart, graph, low quality, blurry, jpeg artifacts, deformed"
    ),
    seed: int | None = None,
    host: str | None = None,
    timeout: float | None = None,
) -> Path:
    """POST to ``{host}/render`` and write the returned PNG to ``output_path``.

    Width/height default to 1024×1024 (SDXL native) — we crop to the
    target 1080×1200 in the composite step rather than asking the model
    to render a non-square aspect, which produces better results on SDXL.
    """
    host_url = (host or _host_url())
    body: dict[str, Any] = {
        "prompt": scene_prompt,
        "width": int(width),
        "height": int(height),
    }
    if negative_prompt:
        body["negative_prompt"] = negative_prompt
    if seed is not None:
        body["seed"] = int(seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    to = timeout if timeout is not None else _timeout()
    async with httpx.AsyncClient(timeout=to) as client:
        resp = await client.post(f"{host_url}/render", json=body)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
    return output_path


# ---------------------------------------------------------------------------
# Combined helper — used by the Brew pipeline.
# ---------------------------------------------------------------------------


async def illustrate_idea(
    idea_fields: dict,
    output_path: Path,
    *,
    model: str = "claude-opus-4-7",
) -> tuple[Path, str]:
    """Generate the visual-scene prompt + the image. Returns (png_path,
    scene_prompt). Caller composites the PNG with the text panel via
    brew_render.render_card_with_image."""
    scene = await generate_image_prompt(idea_fields, model=model)
    png = await request_image(scene, output_path)
    return png, scene


# ---------------------------------------------------------------------------
# Tiny CLI for smoke-testing — call directly with `python -m brew_image
# --title ... --pitch ... --mechanism ... --out /tmp/x.png`.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse
    import asyncio
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--title", required=True)
    ap.add_argument("--pitch", default="")
    ap.add_argument("--mechanism", default="")
    ap.add_argument("--out", default="/tmp/brew_smoke.png")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    async def _run():
        fields = {
            "title": args.title,
            "pitch": args.pitch,
            "mechanism": args.mechanism,
        }
        png, scene = await illustrate_idea(fields, Path(args.out))
        print(f"scene prompt:\n  {scene}")
        print(f"image saved: {png} ({png.stat().st_size:,} bytes)")

    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
