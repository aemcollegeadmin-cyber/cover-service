"""Фінальний вибір кадру через vision-модель."""

import base64
import json
import os

import cv2
import requests

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
API_URL = "https://api.anthropic.com/v1/messages"

PROMPT = """Ти обираєш обкладинку для вертикального відео в Instagram та TikTok.

Нижче кілька кадрів з одного відео, пронумерованих з 1.
Критерії, у порядку важливості:

1. Людина в кадрі виглядає нормально: не в момент мовлення з викривленим ротом,
   не з дивною мімікою, не з незручною позою, очі відкриті та живі.
2. Видно контекст відео: зрозуміло, де людина знаходиться і що відбувається.
   Кадр з цікавим оточенням кращий за крупний план на порожній стіні.
3. Кадр чистий: без субтитрів та накладеного тексту.

Зверху на обкладинку ляже великий заголовок, тому верхня третина кадру
може бути перекрита. Це не проблема.

Відповідай ТІЛЬКИ JSON без markdown:
{"index": <номер кадру>, "reason": "<коротко українською, одне речення>"}"""


def _b64(img, max_w=768):
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()


def choose(finalists):
    """finalists: список (ts, img, metrics). Повертає (index, reason)."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key or len(finalists) == 1:
        return 0, "без моделі, взято найкращий за скорингом"

    content = []
    for i, (_ts, img, _m) in enumerate(finalists, start=1):
        content.append({"type": "text", "text": f"Кадр {i}:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(img)},
        })
    content.append({"type": "text", "text": PROMPT})

    try:
        r = requests.post(
            API_URL,
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=90,
        )
        r.raise_for_status()
        text = "".join(
            b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text"
        )
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        idx = int(data["index"]) - 1
        if 0 <= idx < len(finalists):
            return idx, data.get("reason", "")
    except Exception as e:  # модель не має права валити весь пайплайн
        return 0, f"фолбек на скоринг ({type(e).__name__})"

    return 0, "фолбек на скоринг"
