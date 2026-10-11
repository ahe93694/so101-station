#!/usr/bin/env python3
"""采集入口（内置推流 + 按键控制每一集）。

等价于 `python -m lerobot.scripts.lerobot_record ...`，多了两点：
1. 采集进程内部同时起一个 MJPEG 服务，把采集实际读到的相机画面推给浏览器，数采过程中也能看直播。
2. 每一集的开始和去留都由人按键决定，不再按 reset_time_s 倒计时自动开下一集：

   等待（不录，可遥操作）  →/回车 开始录          Esc 取消退出
   录制中                  → 提前结束这一集        ← 丢弃这一集          Esc 保存已录的并结束
   一集录完（不录，可复位）→/回车 保存，录下一集   ← 丢弃，重录这一集    Esc 保存并结束

   等待期间照常遥操作（从臂跟着主臂动，但不录），开录时从臂不会突然跳到主臂的位置。
   按键直接从终端读（终端窗口要在前台），不依赖 X11，SSH 里也能用；
   每次进入等待前会丢掉之前积压的按键；切换阶段后 0.5 秒内的按键忽略，连按两次 → 不会把下一集也结束掉。
"""
import atexit
import os
import select
import sys
import termios
import threading
import time
import tty

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mjpeg_bridge  # noqa: E402

BAR = "=" * 56


class _CancelledBeforeStart(Exception):
    pass


class TerminalKeys:
    """在终端里读 ← → 回车 Esc，替代 lerobot 基于 pynput 的全局键盘监听。"""

    def __init__(self, on_key):
        self.on_key = on_key
        self.fd = os.open("/dev/tty", os.O_RDONLY)
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # 不回显、逐键读取；Ctrl+C 仍然有效
        atexit.register(self.stop)
        threading.Thread(target=self._run, daemon=True).start()

    def flush(self):
        termios.tcflush(self.fd, termios.TCIFLUSH)

    def stop(self):
        if self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.fd = None

    def _getc(self, timeout=None):
        if timeout is not None and not select.select([self.fd], [], [], timeout)[0]:
            return ""
        return os.read(self.fd, 1).decode(errors="ignore")

    def _run(self):
        while self.fd is not None:
            try:
                c = self._getc()
            except (OSError, TypeError):
                return
            if c in ("\n", "\r"):
                self.on_key("enter")
            elif c == "\x1b":
                if self._getc(0.05) in ("[", "O"):
                    self.on_key({"C": "right", "D": "left"}.get(self._getc(0.05), ""))
                else:
                    self.on_key("esc")


def install_key_control():
    import lerobot.scripts.lerobot_record as lr

    orig_record_loop = lr.record_loop
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    st = {"phase": "init", "allow_left": False, "decision": None, "since": 0.0,
          "keys": None, "first_index": None, "total": _num_episodes_arg()}

    def on_key(k):
        if k == "esc":
            if st["phase"] != "init":
                print("\n[Esc] 停止采集……", flush=True)
                events["stop_recording"] = True
                events["exit_early"] = True
        elif time.monotonic() - st["since"] < 0.5:
            return  # 刚切换阶段，忽略连按带过来的键
        elif st["phase"] == "record":
            if k == "right":
                print("\n[→] 提前结束这一集", flush=True)
                events["exit_early"] = True
            elif k == "left":
                print("\n[←] 丢弃这一集", flush=True)
                events["rerecord_episode"] = True
                events["exit_early"] = True
        elif st["phase"] == "wait":
            if k in ("right", "enter"):
                st["decision"] = "next"
                events["exit_early"] = True
            elif k == "left" and st["allow_left"]:
                st["decision"] = "redo"
                events["exit_early"] = True

    def init_keyboard_listener():
        try:
            st["keys"] = TerminalKeys(on_key)
        except OSError:
            print("!! 没有可用的终端，按键无法使用：请在桌面终端里运行 start_collection.sh", flush=True)
            raise
        return st["keys"], events

    def wait(kwargs, text, allow_left):
        """不录制、照常遥操作，直到按键做出决定。返回 'next' / 'redo' / 'stop'。"""
        st.update(phase="wait", allow_left=allow_left, decision=None, since=time.monotonic())
        st["keys"].flush()
        events["exit_early"] = False
        print(f"\n{BAR}\n{text}\n{BAR}\n", flush=True)
        waited = 0
        idle = {**kwargs, "dataset": None, "control_time_s": 15}
        while st["decision"] is None and not events["stop_recording"]:
            orig_record_loop(**idle)
            if st["decision"] is None and not events["stop_recording"]:
                waited += 15
                print(f"  [未录制] 等待按键……已等 {waited} 秒", flush=True)
        st["phase"] = "busy"
        return "stop" if events["stop_recording"] else st["decision"]

    def record_loop(**kwargs):
        ds = kwargs.get("dataset")
        if ds is None:
            return  # lerobot 自带的倒计时复位，改由 wait() 里按键决定
        if st["first_index"] is None:
            st["first_index"] = ds.num_episodes
            try:
                d = wait(kwargs, "  设备已就绪，可以先用主臂试动、摆好场景（现在不录）\n"
                                 "  >>> → 或 回车：开始录第 1 集 <<<        Esc / Ctrl+C：取消", allow_left=False)
            except KeyboardInterrupt:
                raise _CancelledBeforeStart from None
            if d == "stop":
                raise _CancelledBeforeStart

        n = ds.num_episodes - st["first_index"] + 1  # 本次第几集
        total = st["total"]
        events["exit_early"] = events["rerecord_episode"] = False
        print(f"\n{BAR}\n  ● 正在录第 {n}{'/' + str(total) if total else ''} 集"
              f"（{kwargs['control_time_s']:.0f} 秒）    → 提前结束   ← 丢弃   Esc 结束\n{BAR}\n", flush=True)
        st.update(phase="record", since=time.monotonic())
        orig_record_loop(**kwargs)
        st["phase"] = "busy"
        if events["stop_recording"]:
            return

        last = total is not None and n >= total
        if events["rerecord_episode"]:
            # rerecord 标志保持为真：lerobot 随后会清掉这一集；若按了 Esc，清掉后直接结束
            wait(kwargs, f"  第 {n} 集已丢弃。复位好场景后：\n"
                         f"  → 或 回车：重录第 {n} 集        Esc：结束（不保存这一集）", allow_left=False)
            return

        nxt = "保存，结束采集" if last else "保存，录下一集"
        d = wait(kwargs, f"  第 {n} 集录完（不录了，可以遥操作复位场景）\n"
                         f"  → 或 回车：{nxt}    ←：丢弃，重录第 {n} 集    Esc：保存并结束", allow_left=True)
        if d == "redo":
            events["rerecord_episode"] = True
            print(f"  第 {n} 集已丢弃，马上重录", flush=True)

    lr.record_loop = record_loop
    lr.init_keyboard_listener = init_keyboard_listener


def _num_episodes_arg():
    for a in sys.argv:
        if a.startswith("--dataset.num_episodes="):
            return int(a.split("=", 1)[1])
    return None


if __name__ == "__main__":
    mjpeg_bridge.start()
    install_key_control()
    from lerobot.scripts.lerobot_record import main

    try:
        main()
    except _CancelledBeforeStart:
        print("\n已在开始录制前取消，本次没有录任何数据", flush=True)
        sys.exit(0)
