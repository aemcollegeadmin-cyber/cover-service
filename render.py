"""Рендер обкладинки: кроп, шум, затемнення, плашка з текстом."""

import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920

# --- плашка ---
PLAQUE_W = int(os.getenv("PLAQUE_W", "800"))
PAD_T = int(os.getenv("PAD_T", "40"))
PAD_R = int(os.getenv("PAD_R", "40"))
PAD_B = int(os.getenv("PAD_B", "50"))
PAD_L = int(os.getenv("PAD_L", "40"))
RADIUS_MAIN = int(os.getenv("RADIUS_MAIN", "160"))   # три великі кути
RADIUS_SMALL = int(os.getenv("RADIUS_SMALL", "12"))  # гострий нижній правий
RADIUS = (RADIUS_MAIN, RADIUS_MAIN, RADIUS_SMALL, RADIUS_MAIN)   # TL, TR, BR, BL
SMOOTHING = float(os.getenv("SMOOTHING", "1.4"))  # corner smoothing; 0.6 як в iOS, вище м'якше
PLAQUE_FILL = (0x0B, 0x06, 0x07)
PLAQUE_ALPHA = float(os.getenv("PLAQUE_ALPHA", "0.50"))
PLAQUE_BLUR = float(os.getenv("PLAQUE_BLUR", "104"))
ROTATE = float(os.getenv("ROTATE", "4"))        # градуси, як у Figma
MARGIN_BOTTOM = int(os.getenv("MARGIN_BOTTOM", "430"))

# --- пресет без плашки: текст лягає прямо на кадр ---
PLAIN_TEXT = os.getenv("PLAIN_TEXT", "1") not in ("0", "false", "False")
TEXT_X = int(os.getenv("TEXT_X", "148"))            # лівий відступ тексту
TEXT_BOTTOM = int(os.getenv("TEXT_BOTTOM", "469"))  # від низу кадру до низу тексту
TEXT_BOX_W = int(os.getenv("TEXT_BOX_W", "784"))    # ширина блоку тексту
DESCENDER_PAD = float(os.getenv("DESCENDER_PAD", "0.32"))  # запас під хвости р, у, д

# --- текст ---
FONT_SIZE = int(os.getenv("FONT_SIZE", "112"))
LINE_HEIGHT = int(os.getenv("LINE_HEIGHT", "86"))
TRACKING = float(os.getenv("TRACKING", "-0.04"))
TEXT_COLOR = (0xF0, 0xF0, 0xF0)
TEXT_ALPHA = float(os.getenv("TEXT_ALPHA", "1.0"))
ACCENT_OPACITY = float(os.getenv("ACCENT_OPACITY", "1.0"))
MAX_LINES = int(os.getenv("MAX_LINES", "4"))
SIZE_STEPS = [FONT_SIZE, int(FONT_SIZE * 0.9), int(FONT_SIZE * 0.8)]
TEXT_WIDTH = (TEXT_BOX_W if PLAIN_TEXT else PLAQUE_W - PAD_L - PAD_R)

SHADOW_ALPHA = float(os.getenv("SHADOW_ALPHA", "0.25"))
SHADOW_BLUR = float(os.getenv("SHADOW_BLUR", "74"))   # css blur 149 ≈ sigma 74
SHADOW_OFFSET = int(os.getenv("SHADOW_OFFSET", "3"))

# --- кадр ---
NOISE_CELL = float(os.getenv("NOISE_CELL", "3.9"))
NOISE_ALPHA = float(os.getenv("NOISE_ALPHA", "0.13"))
DIM_ALPHA = float(os.getenv("DIM_ALPHA", "0.20"))

# --- автоматичне наближення обличчя ---
AUTO_ZOOM = os.getenv("AUTO_ZOOM", "1") not in ("0", "false", "False")
FACE_TARGET = float(os.getenv("FACE_TARGET", "0.26"))   # бажана висота обличчя від кадру
MAX_ZOOM = float(os.getenv("MAX_ZOOM", "1.6"))
UPSCALE_LIMIT = float(os.getenv("UPSCALE_LIMIT", "1.25"))  # наскільки можна розтягнути понад рідну роздільність
SHARPEN = float(os.getenv("SHARPEN", "0.35"))              # підняття різкості після збільшення
EYE_LINE = float(os.getenv("EYE_LINE", "0.40"))         # де сидить лінія очей
FACE_GAP = int(os.getenv("FACE_GAP", "24"))             # просвіт між підборіддям і плашкою
HEAD_TOP = float(os.getenv("HEAD_TOP", "0.07"))         # мінімум повітря над головою
GRAY_MIX = float(os.getenv("GRAY_MIX", "1.0"))   # 1.0 = повне ЧБ

FONT_REGULAR = os.getenv("FONT_REGULAR", "/app/fonts/HelveticaNeue-Regular.ttf")
FONT_ITALIC = os.getenv(
    "FONT_ITALIC", "/app/fonts/PlayfairDisplay-Italic.ttf"
)

FALLBACKS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


# ------------------------------------------------------------------ кадр

def crop(img, face=None, safe_bottom=None):
    """Вписує кадр у 1080x1920. З face наближає й центрує обличчя."""
    h, w = img.shape[:2]
    scale = max(W / w, H / h)
    zoom = 1.0

    if AUTO_ZOOM and face:
        x0, y0, x1, y1, eye_y = face
        face_h = max(1.0, y1 - y0)
        current = (face_h * scale) / H
        if current > 0:
            zoom = min(max(FACE_TARGET / current, 1.0), MAX_ZOOM)

        # якщо обличчя не влазить над плашкою, наближаємо сильніше:
        # при зумі під підборіддям зʼявляється запас для зсуву
        if safe_bottom is not None:
            below = (h - y1) * scale          # скільки картинки під підборіддям
            need = H - safe_bottom            # скільки треба, щоб підборіддя стало на межу
            if below > 1 and need > 0:
                zoom = max(zoom, need / below)
            zoom = min(zoom, MAX_ZOOM)

        # не розтягуємо кадр сильніше, ніж дозволяє рідна роздільність
        zoom = min(zoom, (1.0 / scale) * UPSCALE_LIMIT)
        zoom = max(zoom, 1.0)

    scale *= zoom
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if scale < 1:
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        if SHARPEN > 0 and scale > 1.01:
            blur = cv2.GaussianBlur(resized, (0, 0), 1.2)
            resized = cv2.addWeighted(resized, 1 + SHARPEN, blur, -SHARPEN, 0)

    if face:
        x0, y0, x1, y1, eye_y = face
        cx = ((x0 + x1) / 2) * scale
        ey = eye_y * scale
        x = int(round(cx - W / 2))
        y = int(round(ey - H * EYE_LINE))

        if safe_bottom is not None:
            # підборіддя має лишитись вище плашки
            y_needed = y1 * scale - safe_bottom
            y = max(y, int(round(y_needed)))
            # маківку бережемо, але не ціною того, що плашка накриє обличчя
            y_head_cap = int(round(y0 * scale - H * HEAD_TOP))
            if y_head_cap >= y_needed:
                y = min(y, y_head_cap)
    else:
        x, y = (nw - W) // 2, (nh - H) // 2

    x = max(0, min(x, nw - W))
    y = max(0, min(y, nh - H))
    return resized[y:y + H, x:x + W]


def noise(img):
    if NOISE_ALPHA <= 0:
        return img
    ch, cw = int(H / NOISE_CELL) + 1, int(W / NOISE_CELL) + 1
    rng = np.random.default_rng()
    small = rng.random((ch, cw), dtype=np.float32)
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    factor = (1.0 - NOISE_ALPHA * big)[:, :, None]
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def desaturate(img, mix=None):
    """Знебарвлення. mix=1.0 повне ЧБ, 0.5 приглушені кольори."""
    m = GRAY_MIX if mix is None else mix
    if m <= 0:
        return img
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if m >= 1:
        return gray
    return cv2.addWeighted(gray, m, img, 1.0 - m, 0)


def dim(img):
    return np.clip(img.astype(np.float32) * (1.0 - DIM_ALPHA), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- шрифт

ITALIC_WEIGHT = float(os.getenv("ITALIC_WEIGHT", "500"))


def _truetype(path, size, fallbacks):
    for candidate in [path] + fallbacks:
        if not candidate:
            continue
        try:
            font = ImageFont.truetype(candidate, size)
            # Playfair Display варіативний: виставляємо потрібну вагу
            if ITALIC_WEIGHT and "Playfair" in str(candidate):
                try:
                    font.set_variation_by_axes([ITALIC_WEIGHT])
                except Exception:
                    pass
            return font
        except Exception:
            continue
    return ImageFont.load_default()


def font_status():
    out = {}
    for label, path in (("regular", FONT_REGULAR), ("italic", FONT_ITALIC)):
        try:
            ImageFont.truetype(path, 24)
            out[label] = path
        except Exception as e:
            out[label] = f"MISSING ({type(e).__name__})"
    return out


def _load(size):
    reg = _truetype(FONT_REGULAR, size, FALLBACKS)
    ital = _truetype(FONT_ITALIC, size, [FONT_REGULAR] + FALLBACKS)
    return reg, ital


# ------------------------------------------------------------------ текст

def _chars(text):
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


def _layout(text):
    """Підбирає кегль. Повертає (рядки, кегль, висота блоку тексту)."""
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    for s in SIZE_STEPS:
        reg, ital = _load(s)
        candidate = _wrap(probe, text, s, reg, ital)
        if len(candidate) <= MAX_LINES:
            lh = int(round(LINE_HEIGHT * s / FONT_SIZE))
            return candidate, s, len(candidate) * lh

    s = SIZE_STEPS[-1]
    reg, ital = _load(s)
    lines = _wrap(probe, text, s, reg, ital)
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES]
        lines[-1] = lines[-1] + [[(".", False), (".", False), (".", False)]]
    lh = int(round(LINE_HEIGHT * s / FONT_SIZE))
    return lines, s, len(lines) * lh


# ---------------------------------------------------------------- плашка

def _corner_points(cx, cy, r, sx, sy, smoothing, steps=48):
    """Чверть супереліпса. smoothing=0 дає коло, більше значення робить кут м'якшим."""
    if r <= 0:
        return [(cx + sx * 0, cy + sy * 0)]
    n = 2.0 + 3.0 * max(0.0, min(2.5, smoothing))
    e = 2.0 / n
    pts = []
    for i in range(steps + 1):
        t = (np.pi / 2) * i / steps
        x = cx + sx * r * (abs(np.cos(t)) ** e)
        y = cy + sy * r * (abs(np.sin(t)) ** e)
        pts.append((float(x), float(y)))
    return pts


def _rounded_mask(w, h, radii, smoothing=None):
    """Маска зі своїм радіусом на кожен кут: TL, TR, BR, BL."""
    sm = SMOOTHING if smoothing is None else smoothing
    tl, tr, br, bl = [int(max(0, min(r, min(w, h) // 2))) for r in radii]

    ss = 4  # надсемплінг заради чистих країв
    W4, H4 = w * ss, h * ss
    tl4, tr4, br4, bl4 = tl * ss, tr * ss, br * ss, bl * ss

    pts = []
    # правий верхній: від верхньої грані до правої
    pts += _corner_points(W4 - tr4, tr4, tr4, 1, -1, sm)[::-1]
    # правий нижній
    pts += _corner_points(W4 - br4, H4 - br4, br4, 1, 1, sm)
    # лівий нижній
    pts += _corner_points(bl4, H4 - bl4, bl4, -1, 1, sm)[::-1]
    # лівий верхній
    pts += _corner_points(tl4, tl4, tl4, -1, -1, sm)

    big = Image.new("L", (W4, H4), 0)
    ImageDraw.Draw(big).polygon(pts, fill=255)
    return big.resize((w, h), Image.LANCZOS)


def _text_layer(size_px, lines, size, offset):
    """RGBA-шар з текстом і тінню, розміром плашки."""
    reg, ital = _load(size)
    lh = int(round(LINE_HEIGHT * size / FONT_SIZE))
    layer = Image.new("RGBA", size_px, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    space = d.textlength(" ", font=reg) + TRACKING * size

    # Figma центрує гліфи всередині рядка висотою lh, тому рахуємо базову лінію
    asc, desc = reg.getmetrics()
    baseline = (lh - (asc + desc)) / 2.0 + asc

    for i, line in enumerate(lines):
        x = float(offset[0])
        y = offset[1] + i * lh + baseline
        for word in line:
            for ch, accent in word:
                font = ital if accent else reg
                a = TEXT_ALPHA * (ACCENT_OPACITY if accent else 1.0)
                d.text((x, y), ch, font=font, anchor="ls",
                       fill=TEXT_COLOR + (int(255 * a),))
                x += d.textlength(ch, font=font) + TRACKING * size
            x += space

    if SHADOW_ALPHA <= 0:
        return layer

    alpha = layer.split()[3].point(lambda v: int(v * SHADOW_ALPHA))
    shadow = Image.new("RGBA", size_px, (0, 0, 0, 255))
    shadow.putalpha(alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))
    shifted = Image.new("RGBA", size_px, (0, 0, 0, 0))
    shifted.paste(shadow, (0, SHADOW_OFFSET), shadow)
    return Image.alpha_composite(shifted, layer)


def plaque(img_bgr, text):
    """Плашка з текстом, прикріплена до низу кадру."""
    if not text or not text.strip():
        return img_bgr

    lines, size, text_h = _layout(text)
    pw = PLAQUE_W
    ph = text_h + PAD_T + PAD_B

    base = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

    mask_rot = _rounded_mask(pw, ph, RADIUS).rotate(
        ROTATE, resample=Image.BICUBIC, expand=True
    )
    rw, rh = mask_rot.size
    px = (W - rw) // 2
    py = H - MARGIN_BOTTOM - rh

    full_mask = Image.new("L", (W, H), 0)
    full_mask.paste(mask_rot, (px, py))

    if PLAQUE_BLUR > 0:
        blurred = base.filter(ImageFilter.GaussianBlur(PLAQUE_BLUR / 3.0))
        base = Image.composite(blurred, base, full_mask)

    fill = Image.new("RGBA", (W, H), PLAQUE_FILL + (255,))
    fill.putalpha(full_mask.point(lambda v: int(v * PLAQUE_ALPHA)))
    base = Image.alpha_composite(base, fill)

    text_rot = _text_layer((pw, ph), lines, size, (PAD_L, PAD_T)).rotate(
        ROTATE, resample=Image.BICUBIC, expand=True
    )
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(
        text_rot,
        ((W - text_rot.size[0]) // 2, H - MARGIN_BOTTOM - text_rot.size[1]),
        text_rot,
    )
    base = Image.alpha_composite(base, canvas)

    return cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)


def plain_text(img_bgr, text):
    """Текст прямо на кадрі, без плашки. Притиснутий влів і донизу."""
    if not text or not text.strip():
        return img_bgr

    lines, size, text_h = _layout(text)
    # запас під хвости літер, що звисають нижче базової лінії
    tail = int(size * DESCENDER_PAD)
    layer = _text_layer((TEXT_WIDTH, text_h + tail), lines, size, (0, 0))

    base = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(layer, (TEXT_X, H - TEXT_BOTTOM - text_h), layer)
    base = Image.alpha_composite(base, canvas)
    return cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)


def text_top(text):
    """Верхня межа текстового блоку, або None якщо тексту немає."""
    if not text or not text.strip():
        return None
    _lines, _size, text_h = _layout(text)
    return H - TEXT_BOTTOM - text_h


def plaque_top(text):
    """Верхня межа плашки з урахуванням повороту, або None якщо тексту немає."""
    if not text or not text.strip():
        return None
    _lines, _size, text_h = _layout(text)
    ph = text_h + PAD_T + PAD_B
    rad = abs(np.deg2rad(ROTATE))
    rh = PLAQUE_W * np.sin(rad) + ph * np.cos(rad)
    return H - MARGIN_BOTTOM - rh


def compose(frame_bgr, text=None, bw=False, face=None):
    top = text_top(text) if PLAIN_TEXT else plaque_top(text)
    safe = None if top is None else top - FACE_GAP
    img = crop(frame_bgr, face, safe)
    if bw:
        img = desaturate(img)
    img = noise(img)
    img = dim(img)
    if text:
        img = plain_text(img, text) if PLAIN_TEXT else plaque(img, text)
    return img
