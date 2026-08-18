"""Прибирання випалених субтитрів з кадру."""

import os

import cv2
import numpy as np

BAND_TOP = float(os.getenv("CLEAN_BAND_TOP", "0.40"))
BAND_BOT = float(os.getenv("CLEAN_BAND_BOT", "0.88"))
MIN_BRIGHT = int(os.getenv("CLEAN_MIN_BRIGHT", "150"))
MAX_SAT = int(os.getenv("CLEAN_MAX_SAT", "95"))
DILATE = int(os.getenv("CLEAN_DILATE", "5"))
MIN_RATIO = float(os.getenv("CLEAN_MIN_RATIO", "0.9"))   # наскільки витягнутим має бути блок
MAX_RATIO = float(os.getenv("CLEAN_MAX_RATIO", "30"))
MIN_W = int(os.getenv("CLEAN_MIN_W", "40"))
MAX_H = float(os.getenv("CLEAN_MAX_H", "0.18"))
MAX_W_FRAC = float(os.getenv("CLEAN_MAX_W_FRAC", "1.0"))  # блоки на всю ширину теж бувають текстом
MASK_MAX_H = float(os.getenv("CLEAN_MASK_MAX_H", "0.45"))  # межа висоти всередині смуги пошуку
# нижня частина кадру, куди сяде плашка з заголовком: там прибирати нема сенсу
SKIP_BOTTOM = float(os.getenv("CLEAN_SKIP_BOTTOM", "0.62"))
RADIUS = int(os.getenv("CLEAN_RADIUS", "6"))


def text_mask(img):
    """Маска випаленого світлого тексту в нижній частині кадру."""
    h, w = img.shape[:2]
    y0, y1 = int(h * BAND_TOP), int(h * BAND_BOT)
    band = img[y0:y1, :]

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]

    # текст світліший за локальний фон, тому порівнюємо з розмитою версією
    local = cv2.GaussianBlur(val, (0, 0), 15)
    lifted = cv2.subtract(val, local)
    bright = (
        (val >= MIN_BRIGHT) & (sat <= MAX_SAT) & (lifted >= 12)
    ).astype(np.uint8) * 255

    # текст має різкі краї, тому лишаємо тільки те, що збігається з градієнтом
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(
        gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    _, edges = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    cand = cv2.bitwise_and(bright, edges)

    # склеюємо букви в слова
    glued = cv2.morphologyEx(
        cand, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)), iterations=1
    )

    mask = np.zeros(band.shape[:2], np.uint8)
    cnts, _ = cv2.findContours(glued, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bh = band.shape[0]
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < bh * 0.012 or ch > bh * MASK_MAX_H:   # не занизьке, не завелике
            continue
        if cw < ch * MIN_RATIO:                       # схоже на слово або рядок
            continue
        if cw > w * MAX_W_FRAC:
            continue
        cv2.drawContours(mask, [c], -1, 255, -1)

    if DILATE:
        mask = cv2.dilate(mask, np.ones((DILATE, DILATE), np.uint8), iterations=1)

    full = np.zeros(img.shape[:2], np.uint8)
    full[y0:y1, :] = mask
    return full


INPAINT_SENS = float(os.getenv("INPAINT_SENS", "9"))     # чутливість до слідів тексту
INPAINT_DILATE = int(os.getenv("INPAINT_DILATE", "4"))
INPAINT_RADIUS = int(os.getenv("INPAINT_RADIUS", "3"))
INPAINT_PASSES = int(os.getenv("INPAINT_PASSES", "1"))


def _glyph_mask(img, box):
    """Повна маска гліфів у рамці рядка: світлі штрихи, згладжені краї і тінь."""
    H, W = img.shape[:2]
    x, y, w, h = box
    ex, ey = int(h * 0.10), int(h * 0.10)
    x0, y0 = max(0, x - ex), max(0, y - ey)
    x1, y1 = min(W, x + w + ex), min(H, y + h + ey)

    roi = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # локальний фон: медіана великим ядром, текст на неї не впливає
    k = max(3, (int(h * 0.9) // 2) * 2 + 1)
    bg = cv2.medianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), min(k, 99)).astype(np.float32)
    diff = gray - bg

    # і світліші за фон штрихи, і темніша обводка під ними
    m = ((diff > INPAINT_SENS) | (diff < -INPAINT_SENS * 1.6)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if INPAINT_DILATE:
        m = cv2.dilate(m, np.ones((INPAINT_DILATE, INPAINT_DILATE), np.uint8))

    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = m
    return full


def clean(img, face=None):
    """Прибирає субтитри інпейнтингом строго в межах рядків тексту."""
    boxes = [b for b in _line_boxes(text_mask(img), img.shape)
             if not (INPAINT_FACE_SAFE and _overlaps(b, face))]
    if not boxes:
        return img, 0.0

    work = np.zeros(img.shape[:2], np.uint8)
    for b in boxes:
        work = np.maximum(work, _glyph_mask(img, b))

    covered = float(work.mean() / 255.0)
    if covered < 0.0002:
        return img, 0.0

    out = img
    for i in range(max(1, INPAINT_PASSES)):
        method = cv2.INPAINT_TELEA if i % 2 == 0 else cv2.INPAINT_NS
        out = cv2.inpaint(out, work, INPAINT_RADIUS, method)

    # мʼякий стик із рештою кадру, щоб не було рамки
    feather = cv2.GaussianBlur(work, (0, 0), max(2.0, INPAINT_RADIUS * 0.6))
    a = (feather.astype(np.float32) / 255.0)[:, :, None]
    blended = img.astype(np.float32) * (1 - a) + out.astype(np.float32) * a
    return np.clip(blended, 0, 255).astype(np.uint8), round(covered, 4)


MARKER_ALPHA = float(os.getenv("MARKER_ALPHA", "0.92"))
MARKER_PAD = float(os.getenv("MARKER_PAD", "1.9"))
MARKER_FACE_PAD = float(os.getenv("MARKER_FACE_PAD", "0.03"))   # товщина відносно висоти рядка
MARKER_COLOR = (26, 22, 22)   # темно-сірий, не чистий чорний
MARKER_SLANT = float(os.getenv("MARKER_SLANT", "1.5"))   # нахил мазків
MARKER_GRAIN = float(os.getenv("MARKER_GRAIN", "0.28"))  # текстурність
MARKER_DENSITY = float(os.getenv("MARKER_DENSITY", "0.30"))  # щільність мазків
MARKER_EXPAND = float(os.getenv("MARKER_EXPAND", "0.10"))    # наскільки вилазити за рядок
MARKER_FACE_SAFE = os.getenv("MARKER_FACE_SAFE", "1") not in ("0", "false", "False")
# для підстановки пікселів обличчя оминати не треба, там нічого не домальовується
PATCH_FACE_SAFE = os.getenv("PATCH_FACE_SAFE", "0") not in ("0", "false", "False")
INPAINT_FACE_SAFE = os.getenv("INPAINT_FACE_SAFE", "1") not in ("0", "false", "False")


def _line_boxes(mask, frame_shape, min_w=None):
    """Рамки рядків тексту з маски. Все, що не схоже на рядок, відсіюється."""
    H, W = frame_shape[:2]
    min_w = MIN_W if min_w is None else min_w
    band = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (61, 3)), iterations=1
    )
    cnts, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w:
            continue
        if not (H * 0.012 <= h <= H * MAX_H):     # висота рядка субтитрів
            continue
        if y > H * SKIP_BOTTOM:                    # зона майбутньої плашки
            continue
        ratio = w / float(h)
        if not (MIN_RATIO <= ratio <= MAX_RATIO):  # схоже на слово або рядок
            continue
        if w > W * MAX_W_FRAC:
            continue
        boxes.append((x, y, w, h))
    return sorted(boxes, key=lambda b: b[1])


def _stroke(layer, box, rng):
    """Діагональне штрихування поверх рядка, як олівцем."""
    x, y, w, h = box
    pad_x, pad_y = int(h * MARKER_EXPAND), int(h * MARKER_EXPAND * 0.6)
    x0, x1 = x - pad_x, x + w + pad_x
    y0, y1 = y - pad_y, y + h + pad_y
    bh = y1 - y0

    thick = max(5, int(bh * 0.42))
    slant = bh * MARKER_SLANT
    step = max(4, int(thick * MARKER_DENSITY))

    # мазки йдуть зліва направо з нахилом
    sx = x0 - int(slant)
    while sx < x1 + slant:
        jitter = rng.normal(0, bh * 0.06)
        p0 = (int(sx), int(y1 + jitter))
        p2 = (int(sx + slant), int(y0 + jitter))
        pm = (int((p0[0] + p2[0]) / 2 + rng.normal(0, bh * 0.09)),
              int((p0[1] + p2[1]) / 2))
        pts = np.array([p0, pm, p2], np.int32)
        val = int(rng.integers(210, 256))
        cv2.polylines(layer, [pts], False, val,
                      max(3, int(thick * rng.uniform(0.6, 1.15))),
                      lineType=cv2.LINE_AA)
        sx += step * rng.uniform(0.8, 1.15)

    # кілька довших мазків поверх, щоб покриття було щільнішим
    for _ in range(4):
        yy = int(rng.uniform(y0 + bh * 0.15, y1 - bh * 0.15))
        pts = np.array([[x0 - 4, yy + int(rng.normal(0, 4))],
                        [int((x0 + x1) / 2), yy + int(rng.normal(0, 6))],
                        [x1 + 4, yy + int(rng.normal(0, 4))]], np.int32)
        cv2.polylines(layer, [pts], False, int(rng.integers(215, 256)),
                      max(5, int(thick * 1.0)), lineType=cv2.LINE_AA)


def _overlaps(box, face, pad=None):
    """Чи заходить рамка в зону обличчя."""
    if not face:
        return False
    x, y, w, h = box
    fx0, fy0, fx1, fy1, _eye = face
    p = MARKER_FACE_PAD if pad is None else pad
    px, py = (fx1 - fx0) * p, (fy1 - fy0) * p
    return not (x + w < fx0 - px or x > fx1 + px or
                y + h < fy0 - py or y > fy1 + py)


def marker(img, face=None):
    """Замальовує субтитри чорним маркером. Повертає (кадр, частка площі)."""
    mask = text_mask(img)
    boxes = [b for b in _line_boxes(mask, img.shape) if not (MARKER_FACE_SAFE and _overlaps(b, face))]
    if not boxes:
        return img, 0.0

    rng = np.random.default_rng()
    layer = np.zeros(img.shape[:2], np.uint8)
    H, W = img.shape[:2]
    for b in boxes:
        tmp = np.zeros((H, W), np.uint8)
        _stroke(tmp, b, rng)
        # мазки не мають вилазити за рядок
        x, y, w, h = b
        ex, ey = int(h * MARKER_EXPAND), int(h * MARKER_EXPAND * 0.6)
        clip = np.zeros((H, W), np.uint8)
        cv2.rectangle(clip,
                      (max(0, x - ex), max(0, y - ey)),
                      (min(W - 1, x + w + ex), min(H - 1, y + h + ey)),
                      255, -1)
        clip = cv2.GaussianBlur(clip, (0, 0), max(1.0, h * 0.05))
        tmp = ((tmp.astype(np.float32) * (clip.astype(np.float32) / 255.0))
               ).astype(np.uint8)
        layer = np.maximum(layer, tmp)

    layer = cv2.GaussianBlur(layer, (0, 0), 0.8)

    # зерно, щоб мазок виглядав як олівець, а не заливка
    if MARKER_GRAIN > 0:
        g = rng.random(layer.shape, dtype=np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.7)
        grain = 1.0 - MARKER_GRAIN * g
        layer = np.clip(layer.astype(np.float32) * grain, 0, 255).astype(np.uint8)

    a = (layer.astype(np.float32) / 255.0 * MARKER_ALPHA)[:, :, None]
    paint = np.zeros_like(img, np.float32)
    paint[:] = MARKER_COLOR
    out = img.astype(np.float32) * (1 - a) + paint * a
    return np.clip(out, 0, 255).astype(np.uint8), round(float(layer.mean() / 255), 4)


# --- підстановка з іншого кадру ---
PATCH_OFFSETS = [float(v) for v in os.getenv(
    "PATCH_OFFSETS", "-1.2,1.2,-2.0,2.0,-0.7,0.7,-3.0,3.0,-4.5,4.5").split(",")]
PATCH_MAX_DIFF = float(os.getenv("PATCH_MAX_DIFF", "7"))      # допустима різниця по кільцю
PATCH_MAX_OVERLAP = float(os.getenv("PATCH_MAX_OVERLAP", "0.12"))
PATCH_DILATE = int(os.getenv("PATCH_DILATE", "17"))           # запас навколо тексту


def clean_temporal(img, grab, ts, dur=None, face=None):
    """Замінює субтитри пікселями з іншого кадру відео.

    grab(t) має повертати кадр того ж розміру або None.
    Повертає (кадр, частка площі, звідки взято).
    """
    boxes = [b for b in _line_boxes(text_mask(img), img.shape)
             if not (PATCH_FACE_SAFE and _overlaps(b, face))]
    if not boxes:
        return img, 0.0, None

    mask = np.zeros(img.shape[:2], np.uint8)
    for b in boxes:
        mask = np.maximum(mask, _glyph_mask(img, b))
    if mask.mean() / 255.0 < 0.0002:
        return img, 0.0, None

    # для підстановки беремо із запасом: пікселі справжні, тож ширша маска не шкодить
    if PATCH_DILATE:
        mask = cv2.dilate(mask, np.ones((PATCH_DILATE, PATCH_DILATE), np.uint8))

    # кільце навколо тексту: по ньому перевіряємо, чи кадр не зрушив
    ring = cv2.subtract(
        cv2.dilate(mask, np.ones((41, 41), np.uint8)),
        cv2.dilate(mask, np.ones((9, 9), np.uint8)),
    )
    ring_idx = ring > 0
    need = cv2.countNonZero(mask)

    best, best_diff, best_off = None, None, None
    for off in PATCH_OFFSETS:
        t = ts + off
        if t < 0.2 or (dur and t > dur - 0.2):
            continue
        try:
            donor = grab(t)
        except Exception:
            donor = None
        if donor is None or donor.shape != img.shape:
            continue

        # у донора не має бути свого тексту на цьому ж місці
        dmask = np.zeros(donor.shape[:2], np.uint8)
        for b in boxes:
            dmask = np.maximum(dmask, _glyph_mask(donor, b))
        overlap = cv2.countNonZero(cv2.bitwise_and(dmask, mask)) / float(need)
        if overlap > PATCH_MAX_OVERLAP:
            continue

        diff = float(np.mean(np.abs(
            img[ring_idx].astype(np.float32) - donor[ring_idx].astype(np.float32)
        )))
        if best_diff is None or diff < best_diff:
            best, best_diff, best_off = donor, diff, off

    if best is None or best_diff > PATCH_MAX_DIFF:
        out, cov = clean(img, face=face)
        return out, cov, "inpaint"

    feather = cv2.GaussianBlur(mask, (0, 0), 3.0)
    a = (feather.astype(np.float32) / 255.0)[:, :, None]
    out = img.astype(np.float32) * (1 - a) + best.astype(np.float32) * a
    return (np.clip(out, 0, 255).astype(np.uint8),
            round(float(mask.mean() / 255.0), 4),
            f"frame{best_off:+.1f}s")
