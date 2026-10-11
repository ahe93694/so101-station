#!/usr/bin/env python3
"""让采集进程自己对外推流（采集期间也能看直播）。

背景：这台机器上的 UVC 相机是独占的——监控流(mjpeg_server.py)占着相机时采集进程会
EBUSY 打不开，反过来也一样。所以「采集时也能看流」的唯一可行做法是：在采集进程内部
起同一套 MJPEG 服务，把采集实际读到的帧直接推给浏览器（看到的就是录进数据集的那一路画面）。

由 record_with_stream.py 调用 start() 后正常跑 lerobot_record 即可。
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mjpeg_server as ms  # noqa: E402
import station_config  # noqa: E402

PORT = ms.PORT


def _real(dev) -> str:
    try:
        return os.path.realpath(str(dev))
    except OSError:
        return str(dev)


def start(port=PORT):
    """起 MJPEG 服务，并把 lerobot 读到的顶部相机每一帧推给它。"""
    ms.CAMS["/top"] = ms.top_camera(pushed=True)
    top_dev = _real(station_config.get("TOP_CAM"))

    from lerobot.cameras.opencv import OpenCVCamera

    orig_read = OpenCVCamera._read_from_hardware

    def _patched(self):
        frame = orig_read(self)
        if frame is not None and _real(getattr(self, "index_or_path", "")) == top_dev:
            ms.CAMS["/top"].push(frame)
        return frame

    OpenCVCamera._read_from_hardware = _patched

    def _serve():
        try:
            srv = ms.ThreadingHTTPServer(("0.0.0.0", port), ms.Handler)
            print(f"[bridge] 采集期间推流已开启: http://0.0.0.0:{port}/ (top)", flush=True)
            srv.serve_forever()
        except OSError as e:
            print(f"[bridge] 推流端口 {port} 起不来（{e}）：采集不受影响，只是看不了直播", flush=True)

    threading.Thread(target=_serve, daemon=True).start()
    return True


if __name__ == "__main__":
    start()
    print("[bridge] 独立运行（无采集），Ctrl+C 退出", flush=True)
    threading.Event().wait()
