"""Pillow renderer for Brew cards — 1080×1920 PNG suitable for sharing
to Instagram / TikTok Stories.

The card composition is intentionally minimal: one bold title, a single
pitch line, the donor-mechanism block, the "first step" block, and an
AIdea wordmark. No image generation, no AI graphics — just typeset
text on a flat dark canvas, so each card is reproducible and renders
in <100 ms.

Phase-2 work that lives elsewhere: scheduled reveal (SQLite + JobQueue),
optional image-gen via an external model (Ideogram / Flux), Dispatcher
micro-question cards. For now this just shapes the output of an existing
AIdea synthesis into a shareable artifact.
"""
from __future__ import annotations

import re
import textwrap
import time
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


# Stories ratio — same as Instagram / Snapchat / TikTok vertical posts.
WIDTH = 1080
HEIGHT = 1920

# Flat-dark palette. High contrast on grey/white app backgrounds, plays
# well with both Light and Dark mode previews.
BG = (18, 22, 30)
FG = (235, 235, 240)
ACCENT = (165, 215, 180)   # muted green — the "brewed" cue
SUBTLE = (140, 145, 158)

# Font candidates by role. DejaVuSans is always present on Debian; if the
# bold variant is missing we fall back to regular. PIL's default font is
# the last resort (renders, but tiny).
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _font(size: int, prefer_bold: bool = False) -> ImageFont.ImageFont:
    """Best-effort TrueType lookup, falls back to PIL default."""
    paths = _FONT_CANDIDATES if prefer_bold else _FONT_CANDIDATES[1:]
    for p in paths:
        try:
            return ImageFont.truetype(p, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _wrap_to_pixel_width(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Word-wrap text into lines that each fit within max_width pixels.
    Falls back to character-level breaks for words that exceed the box."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    draw_ctx = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    for w in words:
        trial = " ".join(current + [w])
        w_px = draw_ctx.textlength(trial, font=font)
        if w_px <= max_width or not current:
            current.append(w)
        else:
            lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return lines


def _draw_block(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_spacing: int = 8,
) -> int:
    """Draw word-wrapped text starting at (x, y). Returns the y just below
    the last drawn line, ready for the next block."""
    if not text:
        return y
    lines = _wrap_to_pixel_width(text, font, max_width)
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_spacing
    return y


def _strip_markdown(s: str) -> str:
    """Crude markdown stripping so a title like '*Brew Dispatcher*' renders
    as 'Brew Dispatcher' without literal asterisks."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"_(.+?)_", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    return s.strip()


def render_card(
    *,
    title: str,
    pitch: str,
    mechanism: str = "",
    first_step: str = "",
    output_path: Path,
    footer: str = "AIdea · brewed for you",
) -> Path:
    """Compose and save a 1080×1920 Brew card. Returns the saved path."""
    title = _strip_markdown(title or "Untitled brew")
    pitch = _strip_markdown(pitch or "")
    mechanism = _strip_markdown(mechanism or "")
    first_step = _strip_markdown(first_step or "")

    img = Image.new("RGB", (WIDTH, HEIGHT), color=BG)
    draw = ImageDraw.Draw(img)

    # Layout grid: 80px outer padding both sides.
    PAD = 80
    inner_w = WIDTH - 2 * PAD

    # Top wordmark.
    mark_font = _font(36, prefer_bold=True)
    draw.text((PAD, 80), "☕ AIdea", fill=ACCENT, font=mark_font)

    # Divider stripe under wordmark.
    draw.line([(PAD, 145), (WIDTH - PAD, 145)], fill=SUBTLE, width=2)

    y = 200

    # Title — bold, large, wraps generously.
    title_font = _font(60, prefer_bold=True)
    y = _draw_block(
        draw, PAD, y, title, title_font, FG,
        max_width=inner_w, line_spacing=14,
    )
    y += 30

    # Pitch — regular, mid-size.
    if pitch:
        pitch_font = _font(30)
        y = _draw_block(
            draw, PAD, y, pitch, pitch_font, FG,
            max_width=inner_w, line_spacing=10,
        )
        y += 40

    # Mechanism label + body.
    if mechanism:
        label_font = _font(20, prefer_bold=True)
        draw.text((PAD, y), "MECHANISM", fill=ACCENT, font=label_font)
        y += 38
        body_font = _font(24)
        y = _draw_block(
            draw, PAD, y, mechanism, body_font, SUBTLE,
            max_width=inner_w, line_spacing=8,
        )
        y += 36

    # First step label + body.
    if first_step:
        label_font = _font(20, prefer_bold=True)
        draw.text((PAD, y), "FIRST STEP", fill=ACCENT, font=label_font)
        y += 38
        body_font = _font(26, prefer_bold=True)
        y = _draw_block(
            draw, PAD, y, first_step, body_font, FG,
            max_width=inner_w, line_spacing=8,
        )
        y += 24

    # Footer — anchored to bottom regardless of content height.
    foot_font = _font(22)
    draw.line([(PAD, HEIGHT - 130), (WIDTH - PAD, HEIGHT - 130)], fill=SUBTLE, width=2)
    draw.text((PAD, HEIGHT - 100), footer, fill=SUBTLE, font=foot_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG", optimize=True)
    return output_path


# ---------------------------------------------------------------------------
# Idea-text parser — pulls the labelled blocks out of the synthesis output
# so render_card can be called with structured fields.
# ---------------------------------------------------------------------------


_FIELD_PATTERNS: list[tuple[str, list[str]]] = [
    # (target_field, list of candidate prefix labels — case-insensitive)
    ("title",      ["title", "название", "загаловок"]),
    ("pitch",      ["one-line pitch", "one line pitch", "однострочный питч",
                    "одностраничный питч", "питч в одну строку",
                    "одной строкой", "идея в одной фразе"]),
    ("mechanism",  ["mechanism", "механизм"]),
    ("first_step", ["first step", "первый шаг"]),
]


def parse_idea_fields(text: str) -> dict[str, str]:
    """Extract title/pitch/mechanism/first_step from a synthesis result.
    Tolerant to localized labels (RU + EN) since the synth output language
    follows the topic. Anything we can't identify is left as empty string —
    render_card handles empties gracefully."""
    out: dict[str, str] = {k: "" for k, _ in _FIELD_PATTERNS}
    # Normalise line endings, strip stray asterisks around the labels.
    lines = [_strip_markdown(line).strip() for line in (text or "").splitlines()]

    current_field: str | None = None
    buf: list[str] = []

    def _flush():
        nonlocal buf, current_field
        if current_field and buf:
            existing = out.get(current_field, "")
            joined = " ".join(buf).strip()
            out[current_field] = (existing + " " + joined).strip() if existing else joined
        buf = []

    for raw in lines:
        if not raw:
            _flush()
            continue
        matched_field = None
        body_after_label = raw
        # See if this line starts with one of our labels. Tolerant of
        # suffixes — e.g. "First step the user could take this week: ..." —
        # by consuming everything between the label and the first colon.
        for field, labels in _FIELD_PATTERNS:
            for lab in labels:
                pattern = re.compile(
                    rf"^{re.escape(lab)}\b[^:\-—]*?\s*[:\-—]\s*",
                    re.IGNORECASE,
                )
                m = pattern.match(raw)
                if m:
                    matched_field = field
                    body_after_label = raw[m.end():].strip()
                    break
            if matched_field:
                break
        if matched_field:
            _flush()
            current_field = matched_field
            if body_after_label:
                buf.append(body_after_label)
        else:
            # Stop ingesting once we hit a section we don't track (e.g.
            # "Why it's unexpected", "Risks") — but only after we've started
            # collecting something. Treating these as end-of-block keeps the
            # mechanism/first_step blocks tight.
            if current_field and re.match(
                r"^(where|why|risks?|риски|где|почему|чем)\b",
                raw,
                re.IGNORECASE,
            ):
                _flush()
                current_field = None
            elif current_field:
                buf.append(raw)
    _flush()
    return out


def default_output_path(brew_id: int | str | None = None) -> Path:
    """Standard /opt/aidea/brews/<id|ts>.png location, writable by aidea user."""
    base = Path(__file__).parent / "brews"
    base.mkdir(parents=True, exist_ok=True)
    name = str(brew_id) if brew_id is not None else f"brew-{int(time.time())}"
    return base / f"{name}.png"
