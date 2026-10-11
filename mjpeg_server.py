#!/usr/bin/env python3
"""最小 MJPEG 监控服务。

两种用法：
1. 被 mjpeg_bridge.py 导入：采集/推理进程把读到的每帧 push 进来，这里只负责推给浏览器。
   mjpeg_bridge 用到的接口：CAMS、Camera(name, path, w, h, fps, fourcc=, pushed=)、
   Camera.zoom、Camera.push(frame)、load_saved_zoom()、ThreadingHTTPServer、Handler。
2. 独立运行（没有采集/推理时看画面），相机参数读 station.env：
     python mjpeg_server.py
   注意：相机是独占的，采集/推理启动前必须先停掉它（启动脚本里的 pkill 会做这件事）。

浏览器打开 http://<主机IP>:8080/ 。
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: F401  (bridge 会用)
from pathlib import Path

import cv2

import station_config

ZOOM_FILE = Path.home() / ".so101_mjpeg_zoom.json"
PORT = int(station_config.get("STREAM_PORT", "8080"))
STREAM_FPS = 15
JPEG_QUALITY = 80

CAMS: dict[str, "Camera"] = {}


def load_saved_zoom() -> dict:
    """读取保存的数字变焦，例如 {"top_zoom": 1.5}。没有文件就返回空。"""
    try:
        return json.loads(ZOOM_FILE.read_text())
    except Exception:
        return {}


class Camera:
    def __init__(self, name, path, width=1280, height=720, fps=30, fourcc="", pushed=False, device=None):
        self.name, self.path = name, path
        self.width, self.height, self.fps = width, height, fps
        self.fourcc, self.pushed, self.device = fourcc, pushed, device
        self.zoom = 1.0
        self._frame = None  # BGR ndarray
        self._lock = threading.Lock()
        if not pushed and device is not None:
            threading.Thread(target=self._capture_loop, daemon=True).start()

    def push(self, frame) -> None:
        """采集进程每读一帧调用一次；只存引用，不在这里编码，避免拖慢控制循环。"""
        with self._lock:
            self._frame = frame

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        while True:
            ok, frame = cap.read()
            if ok:
                self.push(frame)
            else:
                time.sleep(0.5)

    def jpeg(self) -> bytes | None:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        if self.zoom and self.zoom > 1.0:
            h, w = frame.shape[:2]
            ch, cw = int(h / self.zoom), int(w / self.zoom)
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            frame = cv2.resize(frame[y0:y0 + ch, x0:x0 + cw], (w, h))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 不刷屏
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            imgs = "".join(
                f'<div><h3>{c.name} ({p})</h3><img src="{p}" style="max-width:48vw"></div>' for p, c in CAMS.items()
            )
            body = f'<html><body style="background:#111;color:#eee;font-family:sans-serif"><div style="display:flex;gap:8px">{imgs}</div></body></html>'
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        cam = CAMS.get(self.path)
        if cam is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                jpg = cam.jpeg()
                if jpg is not None:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                time.sleep(1.0 / STREAM_FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass


def top_camera(pushed: bool) -> Camera:
    """按 station.env 构造顶部相机；pushed=True 时不自己开相机，等采集/推理进程推帧。"""
    g = station_config.get
    dev = g("TOP_CAM")
    if not dev and not pushed:
        raise SystemExit("station.env 里没填 TOP_CAM")
    cam = Camera(
        "top", "/top",
        int(g("TOP_CAM_WIDTH", "1280")), int(g("TOP_CAM_HEIGHT", "720")), int(g("TOP_CAM_FPS", "30")),
        fourcc=g("TOP_CAM_FOURCC"), pushed=pushed, device=None if pushed else dev,
    )
    cam.zoom = float(load_saved_zoom().get("top_zoom", 1.0))
    return cam


def main() -> None:
    CAMS["/top"] = top_camera(pushed=False)
    print(f"独立监控流: http://0.0.0.0:{PORT}/  ({', '.join(CAMS)})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
