"""HTTP-сервіс генерації обкладинок."""

import hashlib
import os
import time

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import cleanup
import frames
import gemini
import quote as quote_card
import render
import scoring
import picker

app = FastAPI(title="cover-service")


class CoverRequest(BaseModel):
    file_id: str | None = None
    video_url: str | None = None
    text: str | None = None
    debug: bool = False
    clean_text: bool = False
    marker: bool = False
    bw: bool | None = None      # None = авто (за хешем відео)


def _source(req: CoverRequest) -> str:
    if req.video_url:
        return req.video_url
    if req.file_id:
        return frames.drive_url(req.file_id)
    raise HTTPException(400, "потрібен file_id або video_url")


def _pipeline(req: CoverRequest):
    frames.LETTERBOXED = False   # кожне відео оцінюємо з нуля
    path = frames.download(_source(req))
    try:
        stamps, dur = frames.candidate_timestamps(path)

        scored = []
        for ts in stamps:
            img = frames.grab(path, ts)
            if img is None:
                continue
            scored.append((ts, img, scoring.analyse(img)))

        if not scored:
            raise HTTPException(422, "не вдалось витягти жоден кадр")

        ranked, had_clean = scoring.rank(scored)
        finalists = scoring.usable(
            scored, limit=int(os.getenv("FINALISTS", "12"))
        )
        idx, reason = picker.choose(finalists)
        ts_win = finalists[idx][0]

        full = frames.grab(path, ts_win, width=None)
        if full is None:
            full = finalists[idx][1]

        try:
            face = scoring.face_box(full)
        except Exception:
            face = None

        cleaned = 0.0
        clean_src = None
        if req.clean_text:
            # спершу пробуємо Gemini: він бачить будь-який текст
            done = None
            if gemini.available():
                try:
                    done = _clean_band(full)
                    # перевіряємо результат: чи не лишився текст
                    if done is not None:
                        try:
                            left = cleanup.text_boxes(done)
                        except Exception:
                            left = []
                        if left:
                            print(f"[clean] після Gemini лишилось блоків: {len(left)}", flush=True)
                            retry = gemini.clean(done, regions=left)
                            if retry is not None:
                                done = retry
                except Exception:
                    done = None
            if done is not None:
                done = _protect_face(full, done, face)
                full, clean_src = done, "gemini"
                print("[clean] джерело: gemini", flush=True)
                cleaned = -1.0            # площу тут не рахуємо
            else:
                try:
                    print(f"[clean] Gemini не дав результату: {gemini.status().get('error')}", flush=True)
                    full, cleaned, clean_src = cleanup.clean_temporal(
                        full,
                        lambda t: frames.grab(path, t, width=None),
                        ts_win, dur, face=face,
                    )
                except Exception as e:
                    clean_src = f"fail: {type(e).__name__}"

        if req.marker:
            try:
                full, _m = cleanup.marker(full, face=face)
            except Exception:
                pass

        meta = {
            "_all": scored,
            "duration": round(dur, 2),
            "candidates": len(scored),
            "clean_frames": had_clean,
            "chosen_ts": ts_win,
            "bw": _bw_decision(req),
            "face_found": face is not None,
            "cleaned": cleaned,
            "clean_source": clean_src,
            "gemini": gemini.status(),
            "reason": reason,
            "finalists": [
                {"ts": t, **m} for t, _i, m in finalists
            ],
        }
        meta["_face"] = face
        return full, finalists, meta
    finally:
        try:
            os.remove(path)
        except OSError:
            pass



FACE_GUARD = float(os.getenv("FACE_GUARD", "0.12"))   # запас навколо обличчя


def _protect_face(orig, cleaned, face):
    """Обличчя завжди з оригіналу. Gemini при чистці трохи зсуває кадр,
    і його очі й рот, вставлені поверх наших, дають подвійні контури.
    Текст поверх обличчя допускаємо лише там, де його реально знайшов детектор."""
    if face is None or cleaned is None:
        return cleaned
    try:
        h, w = orig.shape[:2]
        x0, y0, x1, y1 = [int(v) for v in face[:4]]
        gx, gy = int((x1 - x0) * FACE_GUARD), int((y1 - y0) * FACE_GUARD)
        fx0, fy0 = max(0, x0 - gx), max(0, y0 - gy)
        fx1, fy1 = min(w, x1 + gx), min(h, y1 + gy)

        keep = np.zeros((h, w), np.uint8)
        keep[fy0:fy1, fx0:fx1] = 255
        # з обличчя прибираємо лише справжні рамки тексту
        try:
            for bx, by, bw, bh in cleanup.text_boxes(orig):
                keep[max(0, by - 6):by + bh + 6, max(0, bx - 6):bx + bw + 6] = 0
        except Exception:
            pass

        a = (cv2.GaussianBlur(keep, (0, 0), 8).astype(np.float32) / 255.0)[:, :, None]
        out = cleaned.astype(np.float32) * (1 - a) + orig.astype(np.float32) * a
        print("[clean] обличчя взято з оригіналу", flush=True)
        return np.clip(out, 0, 255).astype(np.uint8)
    except Exception:
        return cleaned


def _clean_band(full):
    """Для кадру з чорними смугами Gemini бачить лише саму картинку,
    інакше він може домалювати щось у чорному полі."""
    if not frames.LETTERBOXED:
        return gemini.clean(full)
    rows = full.max(axis=(1, 2))
    lit = np.where(rows > frames.BAR_LEVEL)[0]
    if not len(lit):
        return gemini.clean(full)
    top, bottom = int(lit[0]), int(lit[-1]) + 1
    cleaned = gemini.clean(full[top:bottom])
    if cleaned is None:
        return None
    out = full.copy()
    out[top:bottom] = cleaned
    return out


def _bw_decision(req: CoverRequest) -> bool:
    """Явний прапорець, інакше половина відео йде в ЧБ за хешем ID."""
    if req.bw is not None:
        return req.bw
    # горизонтальне відео у вертикальному кадрі: після збільшення ЧБ виглядає
    # мертво, тому такі обкладинки завжди кольорові
    if getattr(frames, "LETTERBOXED", False):
        return False
    key = (req.file_id or req.video_url or "").encode()
    if not key:
        return False
    return hashlib.sha1(key).digest()[0] % 2 == 0


def _ascii(value) -> str:
    """HTTP-заголовки не приймають не-latin1, тому чистимо."""
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


@app.get("/health")
def health():
    return {"ok": True, "model": picker.MODEL, "fonts": render.font_status()}


@app.post("/cover")
def cover(req: CoverRequest):
    t0 = time.time()
    full, finalists, meta = _pipeline(req)

    if req.debug:
        winner_ts = meta["chosen_ts"]
        tiles = []
        for i, (ts, img, m) in enumerate(meta["_all"], start=1):
            tile = cv2.resize(render.crop(img), (300, 533))
            win = abs(ts - winner_ts) < 1e-6
            colour = (0, 255, 0) if win else (
                (0, 200, 255) if not m["reject"] else (0, 0, 255)
            )
            cv2.rectangle(tile, (0, 0), (299, 532), colour, 4)
            cv2.putText(tile, f"{i} t={ts}", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            labels = m["reject"] or ["OK"]
            for j, lab in enumerate(labels):
                cv2.putText(tile, lab, (10, 66 + j * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            info = f"s{m['score']} b{m['blur']} c{m['caption']}"
            cv2.putText(tile, info, (10, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if m["face"]:
                info2 = f"ear{m['ear']} mar{m['mar']} yaw{m['yaw']}"
                cv2.putText(tile, info2, (10, 522),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(tile)

        per_row = 5
        rows = []
        for r in range(0, len(tiles), per_row):
            chunk = tiles[r:r + per_row]
            while len(chunk) < per_row:
                chunk.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(chunk))
        montage = np.vstack(rows)
        ok, buf = cv2.imencode(".jpg", montage, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(
            content=buf.tobytes(),
            media_type="image/jpeg",
            headers={
                "X-Cover-Meta": _ascii({k: v for k, v in meta.items() if not k.startswith("_")}),
                "X-Elapsed": f"{time.time() - t0:.1f}",
            },
        )

    try:
        img = render.compose(full, req.text, bw=_bw_decision(req), face=(None if (frames.LETTERBOXED and frames.KEEP_LETTERBOX) else meta.get("_face")),
            letterbox=bool(frames.LETTERBOXED and frames.KEEP_LETTERBOX))
    except MemoryError:
        img = render.compose(full, req.text, bw=_bw_decision(req), face=None)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise HTTPException(500, "не вдалось закодувати jpeg")
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": 'attachment; filename="cover.jpg"',
            "X-Chosen-Ts": _ascii(meta["chosen_ts"]),
            "X-Elapsed": f"{time.time() - t0:.1f}",
        },
    )


@app.post("/inspect")
def inspect(req: CoverRequest):
    """Тільки метрики, без картинки. Зручно для налагодження порогів."""
    _full, _f, meta = _pipeline(req)
    clean = {k: v for k, v in meta.items() if not k.startswith("_")}
    if req.debug:                      # повний список лише на вимогу
        clean["all_frames"] = [
            {"ts": t, **m} for t, _i, m in meta["_all"]
        ]
    else:
        clean.pop("finalists", None)
    return JSONResponse(clean)


class QuoteRequest(BaseModel):
    text: str = ""


@app.post("/quote")
def quote(req: QuoteRequest):
    """Картка цитати за шаблоном. На вхід лише текст."""
    if not (req.text or "").strip():
        raise HTTPException(422, "порожній текст цитати")

    img = quote_card.compose(req.text)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(500, "не вдалось закодувати картинку")

    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={"X-Quote-Size": f"{img.shape[1]}x{img.shape[0]}"},
    )
