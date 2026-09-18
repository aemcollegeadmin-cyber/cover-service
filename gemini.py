"""Прибирання накладеного тексту через Gemini (Nano Banana).

Кадр іде в модель з інструкцією стерти субтитри й написи, не чіпаючи
решту. Якщо ключа немає або запит не вдався, повертає None, і викликач
падає на стару логіку з донором і інпейнтингом.
"""

import base64
import os
import random
import time

import cv2
import numpy as np
import requests

API_KEY = os.getenv("GEMINI_API_KEY", "")
# Flash іде першою: окрема інфраструктура, 3-5 с на кадр, майже не падає.
# Pro лишається другою — вона якісніша, але це preview з дефіцитом ємності.
MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-2.5-flash-image,gemini-3-pro-image-preview"
).split(",") if m.strip()]
MODEL = MODELS[0]
ROUNDS = int(os.getenv("GEMINI_ROUNDS", "2"))        # заходів по всіх моделях
BASE_DELAY = float(os.getenv("GEMINI_BASE_DELAY", "8"))
MAX_TOTAL = float(os.getenv("GEMINI_MAX_TOTAL", "420"))  # стеля на весь кадр
TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "240"))
MAX_SIDE = int(os.getenv("GEMINI_MAX_SIDE", "1280"))
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
    "- Do not stylise, sharpen or beautify the image.\n"
    "- NEVER cover the text with a flat rectangle, blur patch or solid "
    "colour block. Reconstruct the real background texture instead.\n\n"
    "Output the edited image."
)

_quota_hit = False
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
    """Кілька заходів по всіх моделях із наростаючою паузою.

    503 у Gemini означає перевантаження, а не відмову, і минає само.
    Обкладинки робляться заздалегідь, тому чекати ми можемо.
    """
    global _quota_hit
    if _quota_hit:
        print("[gemini] квота вибита, пропускаю без запитів", flush=True)
        return None

    started = time.time()
    for attempt in range(ROUNDS):
        for model in MODELS:
            if time.time() - started > MAX_TOTAL:
                print("[gemini] вичерпано час очікування", flush=True)
                return None
            out = _call(model, img, regions)
            if out is not None:
                return out
            if "HTTP 429" in str(_last.get("error") or ""):
                _quota_hit = True
                print("[gemini] 429: ліміт запитів, далі не пробую", flush=True)
                return None

        if attempt < ROUNDS - 1:
            delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 4)
            print(f"[gemini] всі моделі зайняті, пауза {delay:.0f} с "
                  f"(захід {attempt + 1}/{ROUNDS})", flush=True)
            time.sleep(delay)

    print("[gemini] здаюсь після всіх спроб", flush=True)
    return None


def _call(model, img, regions=None):
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
        f"{model}:generateContent"
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
            print(f"[gemini] HTTP {r.status_code}: {r.text[:180]}", flush=True)
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
            print("[gemini] у відповіді немає зображення", flush=True)
            return None

        arr = np.frombuffer(base64.b64decode(raw), np.uint8)
        out = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if out is None:
            _last["error"] = "не вдалося декодувати відповідь"
            return None

        # модель може віддати інший розмір, повертаємо до оригінального
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)

        out = _keep_colour(img, out)

        if _flat_patch(img, out):
            _last["error"] = "модель замалювала ділянку рівною плямою"
            print("[gemini] відхилено: рівна пляма", flush=True)
            return None

        _last["ok"] += 1
        _last["error"] = None
        print("[gemini] ok" + (" (з координатами)" if regions else ""), flush=True)
        return out

    except Exception as e:
        _last["error"] = f"{type(e).__name__}: {e}"[:300]
        return None


FLAT_MIN_AREA = float(os.getenv("GEMINI_FLAT_AREA", "0.01"))   # 1% кадру
FLAT_STD = float(os.getenv("GEMINI_FLAT_STD", "5.0"))


def _flat_patch(src, out) -> bool:
    """Чи з'явилась велика рівна пляма там, де раніше була текстура."""
    try:
        h, w = src.shape[:2]
        d = cv2.absdiff(src, out).mean(axis=2)
        changed = (d > 12).astype(np.uint8) * 255
        if cv2.countNonZero(changed) < h * w * FLAT_MIN_AREA:
            return False

        changed = cv2.morphologyEx(
            changed, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
        )
        cnts, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh < h * w * FLAT_MIN_AREA:
                continue
            patch = cv2.cvtColor(out[y:y + bh, x:x + bw], cv2.COLOR_BGR2GRAY)
            if float(patch.std()) < FLAT_STD:      # майже однотонна ділянка
                return True
        return False
    except Exception:
        return False


KEEP_COLOUR = os.getenv("GEMINI_KEEP_COLOUR", "1") not in ("0", "false", "False")


def _keep_colour(src, out):
    """Модель іноді знебарвлює кадр. Беремо від неї лише яскравість,
    а кольоровість повертаємо з оригіналу."""
    if not KEEP_COLOUR:
        return out
    try:
        s_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        o_hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        # якщо модель майже прибрала насиченість, відновлюємо з оригіналу
        if float(o_hsv[:, :, 1].mean()) < float(s_hsv[:, :, 1].mean()) * 0.6:
            o_lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            s_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
            merged = cv2.merge([o_lab[:, :, 0], s_lab[:, :, 1], s_lab[:, :, 2]])
            print("[gemini] відновлено колір з оригіналу", flush=True)
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        pass
    return out
