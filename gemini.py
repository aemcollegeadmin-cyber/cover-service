"""Прибирання накладеного тексту через Gemini (Nano Banana).

Кадр іде в модель з інструкцією стерти субтитри й написи, не чіпаючи
решту. Якщо ключа немає або запит не вдався, повертає None, і викликач
падає на стару логіку з донором і інпейнтингом.
"""

import base64
import os

import cv2
import numpy as np
import requests

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-image-preview")
TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "120"))
MAX_SIDE = int(os.getenv("GEMINI_MAX_SIDE", "1920"))
JPEG_Q = int(os.getenv("GEMINI_JPEG_Q", "92"))

PROMPT = (
    "You are an image retoucher. This frame has text burned into it: "
    "subtitles, captions, decorative lettering or stickers.\n\n"
    "TASK: erase ALL of that text completely, including partial words cut "
    "off by the frame edges, their drop shadows and outlines. Rebuild the "
    "background that was hidden behind them so the result looks like the "
    "text was never there.\n\n"
    "RULES:\n"
    "- Change nothing except the text areas.\n"
    "- The person's face, expression, eyes, skin, hair and clothing must "
    "stay exactly as they are. Do not redraw the face.\n"
    "- Keep the same framing, aspect ratio, resolution, colours, grain "
    "and lighting.\n"
    "- Do not add any new text, logos or objects.\n"
    "- Do not stylise, sharpen or beautify the image.\n\n"
    "Output the edited image."
)

_last = {"error": None, "calls": 0, "ok": 0, "model": MODEL}


def status():
    return {"enabled": bool(API_KEY), **_last}


def available() -> bool:
    return bool(API_KEY)


def _encode(img):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def clean(img, regions=None):
    """Повертає кадр без накладеного тексту або None, якщо не вдалося.

    regions: список (x, y, w, h) — де саме лишився текст. Якщо переданий,
    модель отримує явні координати й працює прицільно.
    """
    if not API_KEY:
        _last["error"] = "GEMINI_API_KEY не заданий"
        return None

    _last["calls"] += 1
    h, w = img.shape[:2]

    prompt = PROMPT
    if regions:
        spots = "; ".join(
            f"({int(x)},{int(y)}) to ({int(x + bw)},{int(y + bh)})"
            for x, y, bw, bh in regions[:8]
        )
        prompt += (
            f"\n\nText is still visible in these regions of the "
            f"{w}x{h} image (top-left origin): {spots}. "
            "Erase it completely and rebuild the background there."
        )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent"
    )
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": _encode(img)}},
            ],
        }],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }

    try:
        r = requests.post(
            url,
            headers={"x-goog-api-key": API_KEY, "Content-Type": "application/json"},
            json=body,
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            _last["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
            return None

        data = r.json()
        parts = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [])
        )
        raw = None
        for p in parts:
            blob = p.get("inline_data") or p.get("inlineData")
            if blob and blob.get("data"):
                raw = blob["data"]
                break
        if raw is None:
            _last["error"] = "у відповіді немає зображення"
            return None

        arr = np.frombuffer(base64.b64decode(raw), np.uint8)
        out = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if out is None:
            _last["error"] = "не вдалося декодувати відповідь"
            return None

        # модель може віддати інший розмір, повертаємо до оригінального
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)

        _last["ok"] += 1
        _last["error"] = None
        return out

    except Exception as e:
        _last["error"] = f"{type(e).__name__}: {e}"[:300]
        return None
