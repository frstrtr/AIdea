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
import time
from pathlib import Path

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
    ("title",      ["title", "название", "заголовок", "загаловок"]),
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
        nonlocal buf  # current_field is only read here, not reassigned
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


# ---------------------------------------------------------------------------
# Composite layout — illustrated Brew cards
#
# 1080×1920 canvas:
#   - top    1080×1200  generated image (cropped/scaled center-fit)
#   - bottom 1080× 720  text panel (dark BG, title + pitch + first step + wordmark)
#
# The text panel uses a darker variant of the existing palette so it
# anchors the image visually rather than competing with it.
# ---------------------------------------------------------------------------


_IMAGE_PANEL_H = 1200
_TEXT_PANEL_H = 1920 - _IMAGE_PANEL_H   # 720
_TEXT_PANEL_BG = (12, 16, 22)            # slightly deeper than text-only card
_TEXT_PAD = 64


def _fit_into(src: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale + center-crop ``src`` into a (target_w × target_h) canvas without
    leaving black bars. Preserves the source aspect by scaling to fill
    then center-cropping the overflow axis."""
    src_w, src_h = src.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    resampled = src.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resampled.crop((left, top, left + target_w, top + target_h))


def render_card_with_image(
    *,
    image_path: Path,
    title: str,
    pitch: str = "",
    first_step: str = "",
    output_path: Path,
    footer: str = "AIdea · brewed for you",
) -> Path:
    """Compose the illustrated Brew card — generated image on top, text
    panel on bottom.

    The mechanism block is intentionally omitted from the composite (it
    fits in the text-only card but crowds the text panel under the
    image). The illustration itself carries the mechanism metaphor;
    the text band is for legibility of what the idea IS.
    """
    title = _strip_markdown(title or "Untitled brew")
    pitch = _strip_markdown(pitch or "")
    first_step = _strip_markdown(first_step or "")

    canvas = Image.new("RGB", (WIDTH, HEIGHT), color=BG)

    # ---- Image panel (top) -------------------------------------------------
    try:
        src = Image.open(image_path).convert("RGB")
        cropped = _fit_into(src, WIDTH, _IMAGE_PANEL_H)
        canvas.paste(cropped, (0, 0))
    except Exception:
        # Failed to load the image — fall back to a flat coloured panel
        # so the card still renders (better than crashing the Brew run).
        placeholder = Image.new(
            "RGB", (WIDTH, _IMAGE_PANEL_H), color=(40, 50, 65),
        )
        canvas.paste(placeholder, (0, 0))

    # ---- Text panel (bottom) -----------------------------------------------
    panel = Image.new("RGB", (WIDTH, _TEXT_PANEL_H), color=_TEXT_PANEL_BG)
    draw = ImageDraw.Draw(panel)

    inner_w = WIDTH - 2 * _TEXT_PAD
    y = 56

    # Top accent line so the panel reads as a deliberate section, not a
    # blank tray.
    draw.line(
        [(_TEXT_PAD, 26), (WIDTH - _TEXT_PAD, 26)],
        fill=ACCENT, width=3,
    )
    # ☕ AIdea wordmark in the upper-left of the panel.
    mark_font = _font(26, prefer_bold=True)
    draw.text((_TEXT_PAD, 36), "☕ AIdea", fill=ACCENT, font=mark_font)
    y = 90

    # Title.
    title_font = _font(48, prefer_bold=True)
    y = _draw_block(
        draw, _TEXT_PAD, y, title, title_font, FG,
        max_width=inner_w, line_spacing=8,
    )
    y += 18

    # Pitch (one line ideally).
    if pitch:
        pitch_font = _font(24)
        y = _draw_block(
            draw, _TEXT_PAD, y, pitch, pitch_font, (210, 215, 222),
            max_width=inner_w, line_spacing=6,
        )
        y += 18

    # First step — most actionable line, bottom-aligned to footer.
    if first_step:
        label_font = _font(18, prefer_bold=True)
        draw.text(
            (_TEXT_PAD, y), "FIRST STEP",
            fill=ACCENT, font=label_font,
        )
        y += 30
        body_font = _font(22, prefer_bold=True)
        _draw_block(
            draw, _TEXT_PAD, y, first_step, body_font, FG,
            max_width=inner_w, line_spacing=6,
        )

    # Footer — always at the bottom of the panel.
    foot_font = _font(18)
    draw.text(
        (_TEXT_PAD, _TEXT_PANEL_H - 40),
        footer, fill=SUBTLE, font=foot_font,
    )

    canvas.paste(panel, (0, _IMAGE_PANEL_H))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return output_path
