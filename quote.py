"""Картка цитати за шаблоном з Фігми (node 699:1033).

Фон #0d0d0d, текст #cfcfcf, Helvetica Neue 39/43, блок 710 по центру,
легке розмиття 2.5. Формат 1080×1350.
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import render

W = int(os.getenv("QUOTE_W", "1080"))
H = int(os.getenv("QUOTE_H", "1350"))
BG = os.getenv("QUOTE_BG", "#0d0d0d")
FG = os.getenv("QUOTE_FG", "#cfcfcf")
SIZE = int(os.getenv("QUOTE_SIZE", "39"))
LINE = int(os.getenv("QUOTE_LINE", "43"))
BOX_W = int(os.getenv("QUOTE_BOX_W", "710"))
TRACKING = float(os.getenv("QUOTE_TRACKING", "-0.01"))    # -0.39px на 39px
BLUR = float(os.getenv("QUOTE_BLUR", "2.5"))
OFFSET_Y = int(os.getenv("QUOTE_OFFSET_Y", "-107"))       # центр блоку відносно центру
PARA_GAP = float(os.getenv("QUOTE_PARA_GAP", "1.0"))      # порожній рядок між абзацами


def _hex(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _wrap(draw, words, font, max_w):
    """Розбиває слова на рядки за шириною блоку."""
    lines, cur = [], ""
    for word in words:
        probe = (cur + " " + word).strip()
        if draw.textlength(probe, font=font) <= max_w or not cur:
            cur = probe
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _draw_line(draw, text, font, cx, y, fill, tracking):
    """Малює рядок по центру з міжлітерним інтервалом."""
    extra = SIZE * tracking
    width = draw.textlength(text, font=font) + extra * max(0, len(text) - 1)
    x = cx - width / 2
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + extra


def compose(text: str):
    """Повертає картку цитати як BGR-масив."""
    canvas = Image.new("RGB", (W, H), _hex(BG))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    font = render._truetype(render.FONT_REGULAR, SIZE, render.FALLBACKS)

    # абзаци розділяємо порожнім рядком, як просив Артур
    paragraphs = [p.strip() for p in str(text or "").split("\n") if p.strip()]
    blocks = []
    for para in paragraphs:
        blocks.append(_wrap(draw, para.split(), font, BOX_W))

    total = 0
    for i, lines in enumerate(blocks):
        total += len(lines) * LINE
        if i < len(blocks) - 1:
            total += LINE * PARA_GAP

    y = (H / 2 + OFFSET_Y) - total / 2
    fill = _hex(FG) + (255,)
    for i, lines in enumerate(blocks):
        for line in lines:
            _draw_line(draw, line, font, W / 2, y, fill, TRACKING)
            y += LINE
        if i < len(blocks) - 1:
            y += LINE * PARA_GAP

    if BLUR > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(BLUR))

    canvas = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
