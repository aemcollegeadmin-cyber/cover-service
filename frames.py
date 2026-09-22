"""Витягування кадрів-кандидатів з відео."""

import os
import re
import subprocess
import tempfile

import cv2
import numpy as np
import requests

SILENCE_DB = os.getenv("SILENCE_DB", "-30")
SILENCE_MIN = float(os.getenv("SILENCE_MIN", "0.25"))
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "20"))
SCORE_WIDTH = int(os.getenv("SCORE_WIDTH", "900"))


def drive_url(file_id: str) -> str:
    return (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )


def download(url: str) -> str:
    """Стрімом качає відео у тимчасовий файл, повертає шлях."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return path


def duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def silence_windows(path: str):
    """Повертає список (start, end) тихих проміжків."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
         "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    log = proc.stderr
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", log)]
    windows = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else None
        if e and e > s:
            windows.append((s, e))
    return windows


def candidate_timestamps(path: str):
    """Таймкоди для кандидатів: спершу з пауз, потім рівномірно як добивка."""
    dur = duration(path)
    head, tail = 0.5, max(0.5, dur - 0.5)

    stamps = []
    for s, e in silence_windows(path):
        mid = (s + e) / 2
        if head <= mid <= tail:
            stamps.append(round(mid, 2))

    # рівномірна добивка, щоб на «безпаузних» відео теж було з чого обирати
    if len(stamps) < MAX_CANDIDATES:
        need = MAX_CANDIDATES - len(stamps)
        step = (tail - head) / (need + 1)
        for i in range(1, need + 1):
            t = round(head + step * i, 2)
            if all(abs(t - x) > 0.4 for x in stamps):
                stamps.append(t)

    stamps = sorted(set(stamps))
    if len(stamps) > MAX_CANDIDATES:
        idx = np.linspace(0, len(stamps) - 1, MAX_CANDIDATES).astype(int)
        stamps = [stamps[i] for i in idx]
    return stamps, dur


def grab(path: str, ts: float, width: int | None = SCORE_WIDTH):
    """Один кадр на таймкоді ts як BGR-масив."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", f"{ts}", "-i", path, "-frames:v", "1"]
    if width:
        cmd += ["-vf", f"scale={width}:-2"]
    cmd += ["-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    if not out:
        return None
    buf = np.frombuffer(out, dtype=np.uint8)
    return unletterbox(cv2.imdecode(buf, cv2.IMREAD_COLOR))


LETTERBOXED = False                                # останній кадр мав смуги
KEEP_LETTERBOX = os.getenv("KEEP_LETTERBOX", "1") not in ("0", "false", "False")
BAR_LEVEL = int(os.getenv("BAR_LEVEL", "14"))      # що вважаємо чорною смугою
BAR_MIN = float(os.getenv("BAR_MIN", "0.06"))      # смуга хоча б 6% висоти


def unletterbox(img):
    """Горизонтальне відео, вставлене у вертикальний кадр з чорними смугами:
    вирізаємо саму картинку, а далі кроп сам розтягне її на весь екран."""
    global LETTERBOXED
    if img is None:
        return img
    h, w = img.shape[:2]
    if h <= w:                       # вже горизонтальний кадр, смуг нема
        return img

    rows = img.max(axis=(1, 2))      # найяскравіший піксель у кожному рядку
    lit = np.where(rows > BAR_LEVEL)[0]
    if not len(lit):
        return img
    top, bottom = int(lit[0]), int(h - 1 - lit[-1])

    # справжній letterbox симетричний і помітний; нічне небо таким не буває
    if top < h * BAR_MIN or bottom < h * BAR_MIN:
        return img
    if abs(top - bottom) > max(top, bottom) * 0.35:
        return img

    band = img[top:h - bottom]
    if band.shape[0] > h * 0.2:
        LETTERBOXED = True
        # чорні смуги лишаємо як є: кадр не розтягуємо
        return img if KEEP_LETTERBOX else band
    return img
