"""Прибирання випалених субтитрів з кадру."""

import os

import cv2
import numpy as np

BAND_TOP = float(os.getenv("CLEAN_BAND_TOP", "0.40"))
BAND_BOT = float(os.getenv("CLEAN_BAND_BOT", "0.97"))
MIN_BRIGHT = int(os.getenv("CLEAN_MIN_BRIGHT", "150"))
MAX_SAT = int(os.getenv("CLEAN_MAX_SAT", "95"))
DILATE = int(os.getenv("CLEAN_DILATE", "5"))
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
        if ch < bh * 0.012 or ch > bh * 0.22:      # не занизьке, не завелике
            continue
        if cw < ch * 1.2:                          # рядок тексту витягнутий
            continue
        if cw > w * 0.98:
            continue
        cv2.drawContours(mask, [c], -1, 255, -1)

    if DILATE:
        mask = cv2.dilate(mask, np.ones((DILATE, DILATE), np.uint8), iterations=1)

    full = np.zeros(img.shape[:2], np.uint8)
    full[y0:y1, :] = mask
    return full


def clean(img):
    """Повертає (кадр без тексту, частка зафарбованої площі)."""
    mask = text_mask(img)
    covered = float(mask.mean() / 255.0)
    if covered < 0.0005:
        return img, 0.0
    out = cv2.inpaint(img, mask, RADIUS, cv2.INPAINT_TELEA)
    return out, round(covered, 4)
