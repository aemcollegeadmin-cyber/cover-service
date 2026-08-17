"""Фінальний вибір кадру через vision-модель."""

import base64
import json
import os

import cv2
import requests

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
API_URL = "https://api.anthropic.com/v1/messages"

PROMPT = """Ти обираєш обкладинку для вертикального відео в Instagram та TikTok.
Це особистий бренд, тому головне — як людина виглядає.

Спочатку відкинь кадри, де вираз обличчя грає проти автора:
- зверхність, зневага, примружене одне око, крива посмішка на один бік
- гримаса посеред слова, скривлені губи, надута щока
- сонний або відсутній погляд, підняте підборіддя «згори вниз»
- очі закочені, скошені вбік або дивляться повз камеру

З того, що лишилось, обирай кадр, де:
1. Вираз спокійний, відкритий і живий. Нейтральний погляд у камеру
   кращий за будь-яку емоцію посеред фрази.
2. Рот закритий або ледь відкритий. Кадр із закритим ротом майже завжди
   кращий за кадр із відкритим, навіть якщо другий здається живішим.
   Ніколи не обирай кадр, де видно зуби або язик посеред слова.
3. Обличчя симетричне: обидва ока розкриті однаково, рот рівний.
4. Видно контекст: зрозуміло, де людина і що відбувається.

На кадрах є субтитри, не враховуй їх.
Знизу ляже плашка з заголовком, тому нижня частина кадру буде перекрита.

Якщо всі кадри погані, обери найменш поганий.

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
