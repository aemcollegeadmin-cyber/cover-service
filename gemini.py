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
ASPECT_TOL = float(os.getenv("GEMINI_ASPECT_TOL", "0.03"))   # допуск 3%
_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]


def _aspect(w, h):
    """Найближча пропорція з тих, які Gemini вміє віддавати."""
    r = w / float(h)
    def val(s):
        a, b = s.split(":")
        return int(a) / int(b)
    return min(_RATIOS, key=lambda s: abs(val(s) - r))
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

    Кадр спершу мінімально обрізається рівно до пропорції, яку Gemini віддає,
    щоб відповідь не довелось розтягувати. Після чистки вставляється назад.
    """
    h, w = img.shape[:2]
    ratio = _ratio_value(_aspect(w, h))
    cw, ch = w, int(round(w / ratio))
    if ch > h:
        ch, cw = h, int(round(h * ratio))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    sub = img[y0:y0 + ch, x0:x0 + cw]
    sub_regions = None
    if regions:
        sub_regions = [(x - x0, y - y0, bw, bh) for x, y, bw, bh in regions]

    out = _clean_rounds(sub, sub_regions)
    if out is None:
        return None
    full = img.copy()
    full[y0:y0 + ch, x0:x0 + cw] = out
    return full


def _ratio_value(s):
    a, b = s.split(":")
    return int(a) / int(b)


def _clean_rounds(img, regions=None):
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
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": _aspect(w, h)},
        },
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
            body = " ".join(str(r.text or "").split())[:400]
            print(f"[gemini] {model} HTTP {r.status_code}: {body}", flush=True)
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

        # розтягувати відповідь не можна: якщо пропорції не збіглись,
        # обличчя сплющиться. Тоді правку просто не беремо.
        oh, ow = out.shape[:2]
        if abs((ow / oh) - (w / h)) / (w / h) > ASPECT_TOL:
            _last["error"] = f"інші пропорції: {ow}x{oh} замість {w}x{h}"
            print(f"[gemini] відхилено: {ow}x{oh} замість {w}x{h}", flush=True)
            return None
        if (oh, ow) != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)

        out = _keep_colour(img, out)


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


KEEP_DIFF = int(os.getenv("GEMINI_KEEP_DIFF", "28"))   # що вважаємо «прибраним текстом»


def _only_text_areas(src, out):
    """Gemini перемальовує весь кадр і попутно згладжує шкіру, як бюті-фільтр.
    Беремо від нього лише ті ділянки, де він справді щось стер (текст),
    решту кадру, включно з обличчям, лишаємо оригінальною."""
    try:
        d = cv2.absdiff(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))
        mask = (d > KEEP_DIFF).astype(np.uint8) * 255
        # шматки тексту — це скупчення сильних змін; дрібний шум відкидаємо
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
        if cv2.countNonZero(mask) == 0:
            return src
        a = (cv2.GaussianBlur(mask, (0, 0), 6).astype(np.float32) / 255.0)[:, :, None]
        mixed = src.astype(np.float32) * (1 - a) + out.astype(np.float32) * a
        print("[gemini] взято лише ділянки тексту", flush=True)
        return np.clip(mixed, 0, 255).astype(np.uint8)
    except Exception:
        return out


ALIGN_MAX = float(os.getenv("GEMINI_ALIGN_MAX", "12"))   # px, більше — не зсув, а перемальовка


def _align(src, out):
    """Gemini віддає кадр, зсунутий на кілька пікселів. Без вирівнювання
    склеювання з оригіналом дає подвійні контури по краю обличчя."""
    try:
        a = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
        b = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
        if abs(dx) < 0.3 and abs(dy) < 0.3:
            return out
        if abs(dx) > ALIGN_MAX or abs(dy) > ALIGN_MAX:
            return out
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        print(f"[gemini] вирівняно зсув {dx:.1f},{dy:.1f}", flush=True)
        return cv2.warpAffine(out, M, (out.shape[1], out.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return out
