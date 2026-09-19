"""Картка цитати за шаблоном з Фігми (node 699:1033).

Ефект знято вимірюванням реальної картки (k2vp.jpg), а не на око:

  фон       рівно #0d0d0d
  текст     рівно #cfcfcf
  краї      НЕ гаусове розмиття, а dissolve: поріг(blur(маска) + шум).
            Саме тому літери поїдені зерном, а не м'яко розмиті.
  геометрія шрифт 39.4, інтерліньяж 43, блок 725 по центру,
            центр блоку y=673.8 на полотні 1080x1350
  фейд      затемнення+розчинення в правий нижній кут (варіант B)

Звірка з оригіналом на тому самому тексті: переноси рядків збігаються
всі 5 з 5, bbox тексту в межах 3-11 px на полотні 3240, щільність
чорнила 13.6% проти 13.1%.

Вихід: 3240x4050 (scale 3), як віддавав експорт із Фігми.
"""

import hashlib
import os
import pathlib

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --- геометрія (в одиницях дизайну 1080x1350) --------------------------------
W = int(os.getenv("QUOTE_W", "1080"))
H = int(os.getenv("QUOTE_H", "1350"))
SCALE = int(os.getenv("QUOTE_SCALE", "3"))          # підсумковий експорт 3x
SS = int(os.getenv("QUOTE_SS", "2"))                # суперсемплінг під зерно

BG = os.getenv("QUOTE_BG", "#0d0d0d")
FG = os.getenv("QUOTE_FG", "#cfcfcf")

SIZE = float(os.getenv("QUOTE_SIZE", "39.4"))
LINE = float(os.getenv("QUOTE_LINE", "43"))
BOX_W = float(os.getenv("QUOTE_BOX_W", "725"))
CENTER_Y = float(os.getenv("QUOTE_CENTER_Y", "673.8"))
PARA_GAP = float(os.getenv("QUOTE_PARA_GAP", "0"))  # 0 = абзаци йдуть підряд

# --- dissolve ----------------------------------------------------------------
BLUR = float(os.getenv("QUOTE_BLUR", "3.5"))        # радіус до порогу
NOISE_AMP = float(os.getenv("QUOTE_NOISE_AMP", "0.99"))   # <1, щоб фон лишався чистим
NOISE_CELL = int(os.getenv("QUOTE_NOISE_CELL", "3"))      # крупність зерна
THRESH = float(os.getenv("QUOTE_THRESH", "0.55"))         # >0.5 підтоншує літери
SEED = os.getenv("QUOTE_SEED", "")                  # порожньо = випадкове зерно

# --- фейд у правий нижній кут (варіант B) ------------------------------------
FADE = float(os.getenv("QUOTE_FADE", "0.55"))       # 0 = вимкнути
FADE_START = float(os.getenv("QUOTE_FADE_START", "0.45"))
FADE_DIM = float(os.getenv("QUOTE_FADE_DIM", "0.5"))

# Шрифт лежить у тому ж Bunny, що й самі картки, і кешується на диск при
# першому запиті. Окремої точки відмови це не додає: якщо Bunny лежить —
# пайплайн і так не працює. Локальний файл, якщо він є, завжди в пріоритеті.
FONT_PATH = os.getenv("QUOTE_FONT", "fonts/Onest-Medium.ttf")
FONT_URL = os.getenv(
    "QUOTE_FONT_URL", "https://aemcollege.b-cdn.net/fonts/onest-medium.ttf"
)
FONT_MD5 = os.getenv("QUOTE_FONT_MD5", "5d6641f6926791b19a140d41b19b5d5c")
FONT_CACHE = os.getenv("QUOTE_FONT_CACHE", "/tmp/onest-medium.ttf")

_HERE = pathlib.Path(__file__).resolve().parent


def _hex(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _valid(path: pathlib.Path) -> bool:
    if not path.exists() or path.stat().st_size < 4096:
        return False
    if not FONT_MD5:
        return True
    return hashlib.md5(path.read_bytes()).hexdigest() == FONT_MD5


def _font_file() -> str:
    """Локальний шрифт, інакше — качаємо з Bunny і кешуємо."""
    local = _HERE / FONT_PATH
    if local.exists():
        return str(local)

    cache = pathlib.Path(FONT_CACHE)
    if _valid(cache):
        return str(cache)

    data = requests.get(FONT_URL, timeout=30)
    data.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data.content)
    if not _valid(cache):
        got = hashlib.md5(cache.read_bytes()).hexdigest()
        cache.unlink(missing_ok=True)
        raise RuntimeError(f"шрифт з {FONT_URL} не збігся: md5 {got}, чекали {FONT_MD5}")
    return str(cache)


def _wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        probe = (cur + " " + word).strip()
        if draw.textlength(probe, font=font) <= max_w or not cur:
            cur = probe
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _text_mask(text: str, px_w: int, px_h: int, k: float):
    """Маска тексту (L) у робочій роздільності. k — множник одиниць дизайну."""
    img = Image.new("L", (px_w, px_h), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(_font_file(), int(round(SIZE * k)))

    paragraphs = [p.strip() for p in str(text or "").split("\n") if p.strip()]
    blocks = [_wrap(draw, p, font, BOX_W * k) for p in paragraphs]

    lh = LINE * k
    total = sum(len(b) * lh for b in blocks) + lh * PARA_GAP * max(0, len(blocks) - 1)

    y = CENTER_Y * k - total / 2
    for i, lines in enumerate(blocks):
        for line in lines:
            w = draw.textlength(line, font=font)
            draw.text(((px_w - w) / 2, y), line, font=font, fill=255, anchor="la")
            y += lh
        if i < len(blocks) - 1:
            y += lh * PARA_GAP
    return img, [ln for b in blocks for ln in b]


def _fade_field(shape, bbox):
    """1 -> 1-FADE у напрямку правого нижнього кута текстового блоку."""
    h, w = shape
    x0, y0, x1, y1 = bbox
    gx = (np.arange(w, dtype=np.float32) - x0) / max(x1 - x0, 1)
    gy = (np.arange(h, dtype=np.float32) - y0) / max(y1 - y0, 1)
    g = (np.clip(gx, 0, 1)[None, :] + np.clip(gy, 0, 1)[:, None]) / 2.0
    t = np.clip((g - FADE_START) / (1.0 - FADE_START), 0, 1)
    t = t * t * (3 - 2 * t)                      # smoothstep
    return 1.0 - FADE * t


def _dissolve(mask, blur_px, fade, rng):
    m = np.asarray(mask.filter(ImageFilter.GaussianBlur(blur_px))).astype(np.float32) / 255.0
    if fade is not None:
        m = m * fade
    if NOISE_CELL > 1:
        h, w = m.shape
        small = rng.random((h // NOISE_CELL + 1, w // NOISE_CELL + 1)).astype(np.float32)
        noise = np.asarray(
            Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
        ).astype(np.float32) / 255.0
    else:
        noise = rng.random(m.shape).astype(np.float32)
    return (m + (noise - 0.5) * NOISE_AMP) > THRESH


def compose(text: str):
    """Повертає картку цитати як BGR-масив SCALE-кратного розміру."""
    k = SCALE * SS
    px_w, px_h = W * k, H * k

    mask, lines = _text_mask(text, px_w, px_h, k)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("порожній текст цитати")

    fade = _fade_field((px_h, px_w), bbox) if FADE > 0 else None
    rng = np.random.default_rng(int(SEED) if SEED.strip() else None)
    keep = _dissolve(mask, BLUR * k / 3.0, fade, rng)

    bg, fg = _hex(BG), _hex(FG)
    rgb = np.empty(keep.shape + (3,), dtype=np.uint8)
    rgb[...] = bg
    if FADE_DIM > 0 and fade is not None:
        for c in range(3):
            lvl = np.clip(bg[c] + (fg[c] - bg[c]) * (1 - FADE_DIM * (1 - fade)), 0, 255)
            rgb[..., c] = np.where(keep, lvl.astype(np.uint8), bg[c])
    else:
        rgb[keep] = fg

    out = Image.fromarray(rgb).resize((W * SCALE, H * SCALE), Image.LANCZOS)
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


def layout(text: str):
    """Тільки переноси рядків — для діагностики, без рендеру."""
    k = SCALE * SS
    _mask, lines = _text_mask(text, W * k, H * k, k)
    return lines
