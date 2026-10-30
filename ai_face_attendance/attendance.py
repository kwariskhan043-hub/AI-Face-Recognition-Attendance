import re
from pathlib import Path
from datetime import datetime, date

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
STUDENTS_DIR = DATA_DIR / "students"
MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "lbph_model.yml"
LABELS_PATH = MODEL_DIR / "labels.csv"
ATTENDANCE_PATH = DATA_DIR / "attendance.csv"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def ensure_dirs():
    STUDENTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ATTENDANCE_PATH.exists():
        pd.DataFrame(columns=["date", "time", "student_id", "name"]).to_csv(ATTENDANCE_PATH, index=False)

def valid_student_id(value):
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{2,32}", value or ""))

def detector():
    face = cv2.CascadeClassifier(CASCADE_PATH)
    if face.empty():
        raise RuntimeError("OpenCV Haar cascade could not be loaded.")
    return face

def detect_faces(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))
    return gray, faces

def load_labels():
    if LABELS_PATH.exists():
        try:
            df = pd.read_csv(LABELS_PATH)
            return {int(r.label): {"student_id": str(r.student_id), "name": str(r.name)} for r in df.itertuples()}
        except Exception:
            return {}
    return {}

def save_student_image(student_id, name, uploaded_files):
    folder = STUDENTS_DIR / student_id
    folder.mkdir(parents=True, exist_ok=True)
    saved, rejected = 0, 0
    for i, file in enumerate(uploaded_files):
        try:
            arr = np.frombuffer(file.getvalue(), dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                rejected += 1
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gray, faces = detect_faces(rgb)
            if len(faces) != 1:
                rejected += 1
                continue
            x, y, w, h = faces[0]
            crop = gray[y:y+h, x:x+w]
            crop = cv2.resize(crop, (200, 200))
            cv2.imwrite(str(folder / f"{student_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{i}.jpg"), crop)
            saved += 1
        except Exception:
            rejected += 1
    # Keep a lightweight metadata file for display/future extension.
    (folder / "name.txt").write_text(name, encoding="utf-8")
    return saved, rejected

def train_model():
    ensure_dirs()
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=75)
    faces, ids, label_rows = [], [], []
    skipped = 0
    for label, folder in enumerate(sorted([p for p in STUDENTS_DIR.iterdir() if p.is_dir()])):
        sid = folder.name
        name_file = folder / "name.txt"
        name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else sid
        label_rows.append({"label": label, "student_id": sid, "name": name})
        for img_path in folder.glob("*.jpg"):
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                skipped += 1
                continue
            faces.append(cv2.resize(image, (200, 200)))
            ids.append(label)
    if not faces:
        return 0, skipped
    recognizer.train(faces, np.array(ids, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))
    pd.DataFrame(label_rows).to_csv(LABELS_PATH, index=False)
    return len(faces), skipped

def recognize_face(rgb, threshold=65):
    if not MODEL_PATH.exists():
        return {"status": "no_model"}
    gray, faces = detect_faces(rgb)
    if len(faces) == 0:
        return {"status": "no_face"}
    if len(faces) != 1:
        return {"status": "multiple"}
    x, y, w, h = faces[0]
    crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    label, distance = recognizer.predict(crop)
    labels = load_labels()
    if label not in labels or distance > threshold:
        return {"status": "unknown", "distance": float(distance)}
    return {"status": "ok", **labels[label], "distance": float(distance)}

def mark_attendance(student_id, name):
    ensure_dirs()
    df = pd.read_csv(ATTENDANCE_PATH)
    today = str(date.today())
    if not df.empty and ((df["date"].astype(str) == today) & (df["student_id"].astype(str) == student_id)).any():
        return "already"
    now = datetime.now()
    row = pd.DataFrame([{"date": today, "time": now.strftime("%H:%M:%S"), "student_id": student_id, "name": name}])
    pd.concat([df, row], ignore_index=True).to_csv(ATTENDANCE_PATH, index=False)
    return "marked"

def get_today_attendance():
    ensure_dirs()
    df = pd.read_csv(ATTENDANCE_PATH)
    if df.empty:
        return df
    return df[df["date"].astype(str) == str(date.today())]
