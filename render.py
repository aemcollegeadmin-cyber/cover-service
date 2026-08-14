"""Рендер обкладинки: кроп, шум, затемнення, текст."""

import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920

# темплейт
TEXT_TOP = 1125
TEXT_LEFT = 150
TEXT_WIDTH = W - TEXT_LEFT * 2          # 780
FONT_SIZE = 98
LINE_HEIGHT = 106
TRACKING = -0.04                        # -4%
TEXT_COLOR = (0xF0, 0xF0, 0xF0)
ACCENT_OPACITY = 0.65
MAX_LINES = 4
SIZE_STEPS = [98, 88, 80]

# шум
NOISE_CELL = 3.9
NOISE_ALPHA = 0.25
# затемнення
DIM_ALPHA = 0.15

FONT_REGULAR = os.getenv("FONT_REGULAR", "/app/fonts/HelveticaNeue-Regular.ttf")
FONT_ITALIC = os.getenv("FONT_ITALIC", "/app/fonts/HelveticaNeue-MediumItalic.ttf")


def crop(img):
    """Вписує кадр у 1080x1920 по короткій стороні, центрований."""
    h, w = img.shape[:2]
    scale = max(W / w, H / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x, y = (nw - W) // 2, (nh - H) // 2
    return resized[y:y + H, x:x + W]


def noise(img):
    """Mono-шум як у Figma: комірка 3.9, чорний 25%."""
    ch, cw = int(H / NOISE_CELL) + 1, int(W / NOISE_CELL) + 1
    rng = np.random.default_rng()
    small = rng.random((ch, cw), dtype=np.float32)
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    factor = (1.0 - NOISE_ALPHA * big)[:, :, None]
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def dim(img):
    """Плоске чорне 15% на весь кадр."""
    return np.clip(img.astype(np.float32) * (1.0 - DIM_ALPHA), 0, 255).astype(np.uint8)


def _load(size):
    reg = ImageFont.truetype(FONT_REGULAR, size)
    try:
        ital = ImageFont.truetype(FONT_ITALIC, size)
    except Exception:
        ital = reg
    return reg, ital


def _chars(text):
    """Розмічає кожен символ прапорцем акценту. '*слово*' -> акцент."""
    out = []
    for part in re.split(r"(\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("*") and part.endswith("*") and len(part) > 2:
            out += [(c, True) for c in part[1:-1]]
        else:
            out += [(c, False) for c in part]
    return out


def _words(text):
    """Список слів, кожне як список (символ, акцент). Пунктуація лишається зі словом."""
    words, cur = [], []
    for ch, accent in _chars(text):
        if ch.isspace():
            if cur:
                words.append(cur)
                cur = []
        else:
            cur.append((ch, accent))
    if cur:
        words.append(cur)
    return words


def _word_width(draw, word, reg, ital, size):
    w = 0.0
    for ch, accent in word:
        w += draw.textlength(ch, font=ital if accent else reg)
    return w + TRACKING * size * max(len(word) - 1, 0)


def _wrap(draw, text, size, reg, ital):
    """Розкладає текст у рядки з урахуванням трекінгу."""
    lines, cur, cur_w = [], [], 0.0
    space = draw.textlength(" ", font=reg) + TRACKING * size
    for word in _words(text):
        fw = _word_width(draw, word, reg, ital, size)
        add = fw if not cur else cur_w + space + fw
        if cur and add > TEXT_WIDTH:
            lines.append(cur)
            cur, cur_w = [word], fw
        else:
            cur.append(word)
            cur_w = add
    if cur:
        lines.append(cur)
    return lines


def draw_text(img_bgr, text):
    """Малює заголовок за темплейтом. Повертає BGR."""
    if not text or not text.strip():
        return img_bgr

    base = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    probe = ImageDraw.Draw(base)

    size = SIZE_STEPS[-1]
    lines = None
    for s in SIZE_STEPS:
        reg, ital = _load(s)
        candidate = _wrap(probe, text, s, reg, ital)
        if len(candidate) <= MAX_LINES:
            size, lines = s, candidate
            break
    if lines is None:
        reg, ital = _load(SIZE_STEPS[-1])
        lines = _wrap(probe, text, SIZE_STEPS[-1], reg, ital)[:MAX_LINES]
        size = SIZE_STEPS[-1]

    reg, ital = _load(size)
    lh = int(round(LINE_HEIGHT * size / FONT_SIZE))
    space = probe.textlength(" ", font=reg) + TRACKING * size

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    y = TEXT_TOP
    for line in lines:
        x = float(TEXT_LEFT)
        for word in line:
            for ch, accent in word:
                font = ital if accent else reg
                alpha = int(255 * (ACCENT_OPACITY if accent else 1.0))
                d.text((x, y), ch, font=font, fill=TEXT_COLOR + (alpha,))
                x += d.textlength(ch, font=font) + TRACKING * size
            x += space
        y += lh

    out = Image.alpha_composite(base, layer).convert("RGB")
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


def compose(frame_bgr, text=None):
    img = crop(frame_bgr)
    img = noise(img)
    img = dim(img)
    if text:
        img = draw_text(img, text)
    return img
