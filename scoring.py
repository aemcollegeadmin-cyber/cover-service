"""Оцінка кадру: різкість, очі, рот, поворот голови, капшнси."""

import os

import cv2
import mediapipe as mp
import numpy as np

_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.4,
)

# індекси Face Mesh
L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_TOP, MOUTH_BOT = 13, 14
MOUTH_L, MOUTH_R = 61, 291
NOSE, CHIN = 1, 152
CHEEK_L, CHEEK_R = 234, 454

EAR_MIN = float(os.getenv("EAR_MIN", "0.19"))              # нижче — моргання
MAR_MAX = float(os.getenv("MAR_MAX", "0.26"))              # вище — рот відкритий посеред слова
YAW_MAX = float(os.getenv("YAW_MAX", "0.38"))              # асиметрія щік
EAR_DIFF_MAX = float(os.getenv("EAR_DIFF_MAX", "0.075"))   # примружене одне око
MAR_SMIRK = float(os.getenv("MAR_SMIRK", "0.10"))          # перекіс рота, радіани
BLUR_MIN = float(os.getenv("BLUR_MIN", "45"))              # дисперсія лапласіана
CAPTION_MAX = float(os.getenv("CAPTION_MAX", "1.0"))   # 1.0 = фільтр вимкнено
CAPTION_WEIGHT = float(os.getenv("CAPTION_WEIGHT", "0"))


def _ear(pts, idx):
    p = [pts[i] for i in idx]
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    return (a + b) / (2 * c + 1e-6)


def caption_score(img) -> float:
    """Щільність текстоподібних країв у нижній третині кадру."""
    h, w = img.shape[:2]
    strip = img[int(h * 0.45):int(h * 0.95), :]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(
        gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    )
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        ratio = cw / (ch + 1e-6)
        if 2.0 < ratio < 30 and 8 < ch < strip.shape[0] * 0.35:
            area += cw * ch
    return area / (strip.shape[0] * strip.shape[1])


def analyse(img):
    """Повертає dict з метриками та вердиктом."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    caption = caption_score(img)

    res = _mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out = {
        "blur": round(blur, 1),
        "caption": round(caption, 4),
        "face": False,
        "ear": None,
        "mar": None,
        "yaw": None,
        "ear_diff": None,
        "mouth_tilt": None,
        "face_ratio": None,
        "reject": [],
        "score": 0.0,
    }

    if blur < BLUR_MIN:
        out["reject"].append("blur")
    if caption > CAPTION_MAX:
        out["reject"].append("caption")

    if not res.multi_face_landmarks:
        out["reject"].append("no_face")
        out["score"] = max(0.0, blur / 200) - caption * CAPTION_WEIGHT * 0.5
        return out

    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[p.x * w, p.y * h] for p in lm])

    ear_l, ear_r = _ear(pts, L_EYE), _ear(pts, R_EYE)
    ear = (ear_l + ear_r) / 2
    ear_diff = abs(ear_l - ear_r)

    # перекіс рота рахуємо відносно лінії очей, щоб нахил голови не заважав
    eye_lc = np.mean([pts[i] for i in L_EYE], axis=0)
    eye_rc = np.mean([pts[i] for i in R_EYE], axis=0)
    roll = np.arctan2(eye_rc[1] - eye_lc[1], eye_rc[0] - eye_lc[0])
    mouth_ang = np.arctan2(pts[MOUTH_R][1] - pts[MOUTH_L][1],
                           pts[MOUTH_R][0] - pts[MOUTH_L][0])
    mouth_tilt = abs(float(np.arctan2(np.sin(mouth_ang - roll),
                                      np.cos(mouth_ang - roll))))
    mar = np.linalg.norm(pts[MOUTH_TOP] - pts[MOUTH_BOT]) / (
        np.linalg.norm(pts[MOUTH_L] - pts[MOUTH_R]) + 1e-6
    )
    dl = np.linalg.norm(pts[NOSE] - pts[CHEEK_L])
    dr = np.linalg.norm(pts[NOSE] - pts[CHEEK_R])
    yaw = abs(dl - dr) / (dl + dr + 1e-6)

    xs, ys = pts[:, 0], pts[:, 1]
    face_ratio = ((xs.max() - xs.min()) * (ys.max() - ys.min())) / (w * h)

    out.update(
        face=True,
        ear=round(float(ear), 3),
        mar=round(float(mar), 3),
        yaw=round(float(yaw), 3),
        face_ratio=round(float(face_ratio), 4),
        ear_diff=round(float(ear_diff), 3),
        mouth_tilt=round(float(mouth_tilt), 3),
    )

    if ear < EAR_MIN:
        out["reject"].append("eyes_closed")
    if mar > MAR_MAX:
        out["reject"].append("mouth_open")
    if yaw > YAW_MAX:
        out["reject"].append("head_turned")
    if ear_diff > EAR_DIFF_MAX:
        out["reject"].append("squint")
    if mouth_tilt > MAR_SMIRK:
        out["reject"].append("smirk")

    score = 0.0
    score += min(blur / 150.0, 2.0)
    score += min((ear - EAR_MIN) * 6, 1.5)
    score += max(0.0, (MAR_MAX - mar) * 6)
    score += max(0.0, (YAW_MAX - yaw) * 2)
    score += 1.0 if 0.02 < face_ratio < 0.30 else 0.0
    score += max(0.0, (EAR_DIFF_MAX - ear_diff) * 10)      # симетричні очі краще
    score += max(0.0, (MAR_SMIRK - mouth_tilt) * 8)        # рівний рот краще
    score -= caption * CAPTION_WEIGHT
    out["score"] = round(float(score), 3)
    return out


def rank(scored):
    """scored: список (ts, img, metrics). Чисті кадри вперед."""
    clean = [s for s in scored if not s[2]["reject"]]
    pool = clean if clean else sorted(scored, key=lambda s: len(s[2]["reject"]))[:6]
    return sorted(pool, key=lambda s: s[2]["score"], reverse=True), bool(clean)


def face_box(img, max_side=720):
    """Рамка обличчя в пікселях кадру: (x0, y0, x1, y1, eye_y) або None.

    Детекція йде на зменшеній копії, координати повертаються у масштабі
    оригіналу. Так само точно, але значно дешевше по памʼяті.
    """
    h, w = img.shape[:2]
    k = 1.0
    small = img
    if max(h, w) > max_side:
        k = max_side / float(max(h, w))
        small = cv2.resize(img, (int(round(w * k)), int(round(h * k))),
                           interpolation=cv2.INTER_AREA)

    sh, sw = small.shape[:2]
    try:
        res = _mesh.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    except Exception:
        return None
    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[p.x * sw, p.y * sh] for p in lm])
    xs, ys = pts[:, 0], pts[:, 1]
    eye_y = float(np.mean([pts[i][1] for i in L_EYE + R_EYE]))
    inv = 1.0 / k
    return (float(xs.min()) * inv, float(ys.min()) * inv,
            float(xs.max()) * inv, float(ys.max()) * inv, eye_y * inv)
