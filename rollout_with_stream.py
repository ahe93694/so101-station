#!/usr/bin/env python3
"""推理入口（内置推流）。

等价于 `python -m lerobot.scripts.lerobot_rollout ...`，区别只是：推理进程内部会同时
起一个 MJPEG 服务，把策略实际读到的相机画面推给浏览器，因此推理过程中也能看直播。

SO101_STREAM=0 可关闭内置推流（默认开）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mjpeg_bridge  # noqa: E402


def _install_action_trace() -> None:
    """SO101_TRACE_ACTIONS=1 时，把每拍真正发出去的动作和当时关节位置打到 stdout。"""
    import time

    from lerobot.robots.so_follower.so_follower import SOFollower

    _orig_send = SOFollower.send_action
    _orig_obs = SOFollower.get_observation

    def _fmt(d):
        return {k.split(".")[0]: round(float(v), 2) for k, v in d.items() if k.endswith(".pos")}

    _counter = {"n": 0}
    _state = {"present": None}

    def _traced_obs(self):
        obs = _orig_obs(self)
        _state["present"] = _fmt(obs)
        return obs

    def _traced_send(self, action):
        sent = _orig_send(self, action)

        import os as _os

        _counter["n"] += 1

        _max_steps = int(_os.environ.get("SO101_MAX_STEPS", "0") or 0)
        if _max_steps and _counter["n"] >= _max_steps:
            print(f"[STEP] 已达 {_max_steps} 拍上限，主动停止（先回位再断开）", flush=True)
            raise SystemExit(0)

        if _counter["n"] % 200 == 0:
            try:
                temp = self.bus.sync_read("Present_Temperature", normalize=False, num_retry=2)
                hottest = max(int(v) for v in temp.values())
                print(
                    f"[TEMP {time.strftime('%H:%M:%S')}] tick={_counter['n']} 最高温={hottest}C "
                    + " ".join(f"{k}={int(v)}" for k, v in temp.items()),
                    flush=True,
                )
                if hottest >= float(_os.environ.get("SO101_MAX_TEMP", "65")):
                    print("[TEMP] !! 温度超阈值，主动停止（会先回位再断开）", flush=True)
                    raise SystemExit(3)
            except SystemExit:
                raise
            except Exception as e:
                print(f"[TEMP] 读温度失败: {type(e).__name__}: {e}", flush=True)
        print(
            f"[TRACE {time.strftime('%H:%M:%S')}] present={_state['present']}\n"
            f"                        goal_in={_fmt(action)}\n"
            f"                        sent   ={_fmt(sent)}",
            flush=True,
        )
        return sent

    SOFollower.get_observation = _traced_obs
    SOFollower.send_action = _traced_send
    print("[bridge] 动作追踪已开启 (SO101_TRACE_ACTIONS=1)", flush=True)


if __name__ == "__main__":
    if os.environ.get("SO101_STREAM", "1") != "0":
        mjpeg_bridge.start()
    if os.environ.get("SO101_TRACE_ACTIONS", "0") == "1":
        _install_action_trace()
    from lerobot.scripts.lerobot_rollout import main

    main()