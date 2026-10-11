#!/usr/bin/env python3
"""只读体检：逐个舵机读电压、温度和错误位，不上力、不动臂。

用法（在项目目录、so101_runtime 环境里）：
  python check_motors.py            # 两条臂都查
  python check_motors.py leader     # 只查主臂
  python check_motors.py follower   # 只查从臂

遇到 `Input voltage error` / `Missing motor IDs` 等连不上的报错时先跑它，看是哪条臂、哪个舵机。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import station_config  # noqa: E402
from lerobot.motors import Motor, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402
from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE as T  # noqa: E402

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARMS = {"follower": "FOLLOWER_PORT", "leader": "LEADER_PORT"}


def check(arm: str) -> None:
    port = station_config.get(ARMS[arm])
    print(f"\n== {arm}  {port}")
    if not os.path.exists(port):
        print("   !! 串口不在位：USB 没插，或 station.env 写错")
        return
    motors = {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(NAMES, 1)}
    bus = FeetechMotorsBus(port, motors)
    bus.connect(handshake=False)
    ph = bus.packet_handler
    print(f"   {'舵机':<14}{'电压':>8}{'允许范围':>14}{'温度':>7}   状态")
    try:
        for name, m in motors.items():
            def rd(reg):
                addr, n = T[reg]
                return bus._read(addr, n, m.id, num_retry=2, raise_on_error=False)

            v, comm, err = rd("Present_Voltage")
            if not bus._is_comm_success(comm):
                print(f"   {name:<14}{'—':>8}{'':>14}{'':>7}   !! 无应答（没供电或线松了）")
                continue
            lo, *_ = rd("Min_Voltage_Limit")
            hi, *_ = rd("Max_Voltage_Limit")
            t, *_ = rd("Present_Temperature")
            status = ph.getRxPacketError(err) if bus._is_error(err) else "正常"
            print(f"   {name:<14}{v / 10:>7.1f}V{lo / 10:>7.1f}~{hi / 10:.1f}V{t:>6}℃   {status}")
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    for arm in sys.argv[1:] or ARMS:
        check(arm)
