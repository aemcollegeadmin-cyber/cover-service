"""HTTP-сервіс генерації обкладинок."""

import os
import time

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import frames
import render
import scoring
import picker

app = FastAPI(title="cover-service")


class CoverRequest(BaseModel):
    file_id: str | None = None
    video_url: str | None = None
    text: str | None = None
    debug: bool = False


def _source(req: CoverRequest) -> str:
    if req.video_url:
        return req.video_url
    if req.file_id:
        return frames.drive_url(req.file_id)
    raise HTTPException(400, "потрібен file_id або video_url")


def _pipeline(req: CoverRequest):
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
        finalists = ranked[:3]
        idx, reason = picker.choose(finalists)
        ts_win = finalists[idx][0]

        full = frames.grab(path, ts_win, width=None)
        if full is None:
            full = finalists[idx][1]

        meta = {
            "duration": round(dur, 2),
            "candidates": len(scored),
            "clean_frames": had_clean,
            "chosen_ts": ts_win,
            "reason": reason,
            "finalists": [
                {"ts": t, **m} for t, _i, m in finalists
            ],
        }
        return full, finalists, meta
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.get("/health")
def health():
    return {"ok": True, "model": picker.MODEL}


@app.post("/cover")
def cover(req: CoverRequest):
    t0 = time.time()
    full, finalists, meta = _pipeline(req)

    if req.debug:
        sheet = [render.crop(img) for _t, img, _m in finalists]
        sheet = [cv2.resize(s, (360, 640)) for s in sheet]
        for i, s in enumerate(sheet):
            cv2.putText(s, str(i + 1), (16, 56), cv2.FONT_HERSHEY_SIMPLEX,
                        1.8, (0, 255, 0), 3)
        montage = np.hstack(sheet)
        ok, buf = cv2.imencode(".jpg", montage, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(
            content=buf.tobytes(),
            media_type="image/jpeg",
            headers={"X-Cover-Meta": str(meta), "X-Elapsed": f"{time.time() - t0:.1f}"},
        )

    img = render.compose(full, req.text)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(500, "не вдалось закодувати jpeg")
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": 'attachment; filename="cover.jpg"',
            "X-Chosen-Ts": str(meta["chosen_ts"]),
            "X-Elapsed": f"{time.time() - t0:.1f}",
        },
    )


@app.post("/inspect")
def inspect(req: CoverRequest):
    """Тільки метрики, без картинки. Зручно для налагодження порогів."""
    _full, _f, meta = _pipeline(req)
    return JSONResponse(meta)
