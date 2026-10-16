"""
People Counter - Web Dashboard backend
Install: pip install ultralytics opencv-python numpy fastapi uvicorn python-multipart
Run:     python server.py     then open http://localhost:8000
"""
import os
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")  # stable RTSP

import re
import shutil
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

MODEL = "yolov8n.pt"      # try yolov8s.pt for better accuracy
CONF = 0.35
MAX_W = 960               # frames wider than this are downscaled for speed
BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)


def sdist(p, a, b):
    ab = np.subtract(b, a).astype(float)
    ap = np.subtract(p, a).astype(float)
    return (ab[0] * ap[1] - ab[1] * ap[0]) / (np.linalg.norm(ab) + 1e-9)


def in_seg(p, a, b, m=0.1):
    ab = np.subtract(b, a).astype(float)
    t = np.dot(np.subtract(p, a), ab) / (np.dot(ab, ab) + 1e-9)
    return -m <= t <= 1 + m


def color(i):
    h = np.uint8([[[(i * 47) % 180, 220, 255]]])
    b, g, r = cv2.cvtColor(h, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def placeholder(msg):
    img = np.full((540, 960, 3), (28, 20, 12), np.uint8)
    (w, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(img, msg, ((960 - w) // 2, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (200, 200, 200), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


class Engine:
    def __init__(self):
        self.gen = 0
        self.line = [0.0, 0.5, 1.0, 0.5]   # x1,y1,x2,y2 as fractions
        self.in_sign = 1
        self.name, self.status, self.fps, self.active = "No source", "idle", 0.0, 0
        self.jpg = placeholder("Upload a video or paste a camera link")
        self.reset()

    def reset(self):
        self.n_in = self.n_out = 0
        self.side, self.trails = {}, {}
        self.events = deque(maxlen=30)
        self.hourly = {}

    def open(self, src, name):
        self.gen += 1
        self.reset()
        self.name, self.status = name, "connecting"
        threading.Thread(target=self.run, args=(src, self.gen), daemon=True).start()

    def run(self, src, gen):
        is_file = isinstance(src, str) and Path(src).exists()
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.status = "error: cannot open source"
            self.jpg = placeholder("Cannot open this source")
            return
        model = YOLO(MODEL)       # fresh model per source -> fresh tracker IDs
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        self.status = "live" if not is_file else "playing"
        last = time.time()

        while gen == self.gen:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    self.status = "finished"
                    break
                self.status = "reconnecting"        # live stream dropped
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(src)
                continue
            self.status = "live" if not is_file else "playing"

            h, w = frame.shape[:2]
            if w > MAX_W:
                frame = cv2.resize(frame, (MAX_W, int(h * MAX_W / w)))
                h, w = frame.shape[:2]
            A = (int(self.line[0] * w), int(self.line[1] * h))
            B = (int(self.line[2] * w), int(self.line[3] * h))

            res = model.track(frame, persist=True, classes=[0], conf=CONF,
                              tracker="bytetrack.yaml", verbose=False)[0]
            active = 0
            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                ids = res.boxes.id.cpu().numpy().astype(int)
                active = len(ids)
                for (x1, y1, x2, y2), tid in zip(boxes, ids):
                    tid = int(tid)
                    foot = (int((x1 + x2) / 2), int(y2))
                    c = color(tid)
                    tr = self.trails.setdefault(tid, deque(maxlen=40))
                    tr.append(foot)
                    for i in range(1, len(tr)):
                        cv2.line(frame, tr[i - 1], tr[i], c, max(1, i // 12 + 1))

                    d = sdist(foot, A, B)
                    if abs(d) > 4:
                        s = 1 if d > 0 else -1
                        prev = self.side.get(tid)
                        if prev is not None and s != prev and in_seg(foot, A, B):
                            ev = "IN" if s == self.in_sign else "OUT"
                            if ev == "IN":
                                self.n_in += 1
                            else:
                                self.n_out += 1
                            hr = time.strftime("%H:00")
                            self.hourly.setdefault(hr, [0, 0])[0 if ev == "IN" else 1] += 1
                            self.events.appendleft(
                                {"t": time.strftime("%H:%M:%S"), "id": tid, "dir": ev})
                        self.side[tid] = s

                    cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                    cv2.circle(frame, foot, 5, c, -1)
                    cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + 56, max(0, y1 - 20) + 20), c, -1)
                    cv2.putText(frame, f"ID {tid}", (x1 + 4, max(14, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # counting line + IN arrow
            cv2.line(frame, A, B, (255, 211, 34), 3, cv2.LINE_AA)
            mid = ((A[0] + B[0]) // 2, (A[1] + B[1]) // 2)
            n = np.array([B[1] - A[1], A[0] - B[0]], float)
            n = -n / (np.linalg.norm(n) + 1e-9) * self.in_sign
            cv2.arrowedLine(frame, mid, (int(mid[0] - n[0] * 45), int(mid[1] - n[1] * 45)),
                            (160, 245, 52), 3, tipLength=0.35)

            self.active = active
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-6)
            last = now
            self.jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

            if is_file:                                   # play files at real speed
                time.sleep(max(0, 1 / src_fps - (time.time() - t0)))
        cap.release()


engine = Engine()
app = FastAPI()


class Link(BaseModel):
    url: str


class LineIn(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE / "index.html").read_text(encoding="utf-8")


@app.get("/video")
def video():
    def gen():
        while True:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + engine.jpg + b"\r\n"
            time.sleep(0.03)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    name = re.sub(r"[^\w.\-]", "_", file.filename or "video.mp4")
    path = UPLOADS / name
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    engine.open(str(path), name)
    return {"ok": True}


@app.post("/source")
def source(link: Link):
    url = link.url.strip()
    if not re.match(r"^(rtsp|rtmp|http|https)://", url):
        return {"ok": False, "error": "Link must start with rtsp://, http:// or https://"}
    engine.open(url, url if len(url) < 50 else url[:47] + "...")
    return {"ok": True}


@app.post("/line")
def set_line(l: LineIn):
    engine.line = [l.x1, l.y1, l.x2, l.y2]
    engine.side.clear()
    return {"ok": True}


@app.post("/flip")
def flip():
    engine.in_sign *= -1
    return {"ok": True}


@app.post("/reset")
def reset():
    engine.reset()
    return {"ok": True}


@app.get("/stats")
def stats():
    now = time.time()
    hours = [time.strftime("%H:00", time.localtime(now - i * 3600)) for i in range(7, -1, -1)]
    return {
        "in": engine.n_in, "out": engine.n_out,
        "inside": max(0, engine.n_in - engine.n_out),
        "active": engine.active, "fps": round(engine.fps, 1),
        "status": engine.status, "source": engine.name,
        "events": list(engine.events)[:8],
        "hourly": [[h] + engine.hourly.get(h, [0, 0]) for h in hours],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
