"""Детекція тексту в кадрі моделлю EAST.

Знаходить будь-який текст: субтитри, декоративні написи, логотипи.
Якщо модель недоступна, повертає None і викликач падає на евристику.
"""

import os

import cv2
import numpy as np

MODEL_PATH = os.getenv("EAST_MODEL", "/app/models/frozen_east_text_detection.pb")
CONF = float(os.getenv("EAST_CONF", "0.5"))          # поріг упевненості
NMS = float(os.getenv("EAST_NMS", "0.3"))
INPUT_W = int(os.getenv("EAST_W", "736"))            # кратне 32
INPUT_H = int(os.getenv("EAST_H", "1312"))
MERGE_GAP = float(os.getenv("EAST_MERGE_GAP", "0.6"))  # злиття слів у рядок

_net = None
_tried = False


def available() -> bool:
    return _load() is not None


def _load():
    global _net, _tried
    if _net is None and not _tried:
        _tried = True
        try:
            if os.path.exists(MODEL_PATH):
                _net = cv2.dnn.readNet(MODEL_PATH)
        except Exception:
            _net = None
    return _net


def _decode(scores, geometry, conf):
    """Розбирає вихід EAST у прямокутники та впевненості."""
    rects, confs = [], []
    rows, cols = scores.shape[2:4]
    for y in range(rows):
        s = scores[0, 0, y]
        x0, x1, x2, x3 = (geometry[0, i, y] for i in range(4))
        angles = geometry[0, 4, y]
        for x in range(cols):
            if s[x] < conf:
                continue
            ox, oy = x * 4.0, y * 4.0
            angle = angles[x]
            cos, sin = np.cos(angle), np.sin(angle)
            h = x0[x] + x2[x]
            w = x1[x] + x3[x]
            ex = ox + cos * x1[x] + sin * x2[x]
            ey = oy - sin * x1[x] + cos * x2[x]
            rects.append(((ex - w / 2, ey - h / 2), (w, h), -angle * 180.0 / np.pi))
            confs.append(float(s[x]))
    return rects, confs


def _merge(boxes, gap):
    """Зливає слова, що стоять поруч, в один рядок."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    out = []
    for b in boxes:
        x, y, w, h = b
        placed = False
        for i, (X, Y, W_, H_) in enumerate(out):
            # по вертикалі перетинаються, по горизонталі поруч
            v = min(y + h, Y + H_) - max(y, Y)
            if v <= 0:
                continue
            if v < 0.4 * min(h, H_):
                continue
            dist = max(X - (x + w), x - (X + W_))
            if dist > gap * max(h, H_):
                continue
            nx, ny = min(x, X), min(y, Y)
            nx2, ny2 = max(x + w, X + W_), max(y + h, Y + H_)
            out[i] = (nx, ny, nx2 - nx, ny2 - ny)
            placed = True
            break
        if not placed:
            out.append(b)
    return out


def boxes(img, conf=None):
    """Рамки тексту (x, y, w, h) у пікселях кадру. None, якщо моделі немає."""
    net = _load()
    if net is None:
        return None

    H, W = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img, 1.0, (INPUT_W, INPUT_H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False,
    )
    net.setInput(blob)
    try:
        scores, geometry = net.forward(
            ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        )
    except Exception:
        return None

    rects, confs = _decode(scores, geometry, CONF if conf is None else conf)
    if not rects:
        return []

    idx = cv2.dnn.NMSBoxesRotated(rects, confs, CONF if conf is None else conf, NMS)
    if idx is None or len(idx) == 0:
        return []

    kx, ky = W / float(INPUT_W), H / float(INPUT_H)
    found = []
    for i in np.array(idx).flatten():
        pts = cv2.boxPoints(rects[int(i)])
        xs = [p[0] * kx for p in pts]
        ys = [p[1] * ky for p in pts]
        x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
        x1, y1 = min(W, int(max(xs))), min(H, int(max(ys)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        found.append((x0, y0, x1 - x0, y1 - y0))

    merged = _merge(found, MERGE_GAP)
    return sorted(merged, key=lambda b: b[1])
