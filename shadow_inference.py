#!/usr/bin/env python3
"""SO101 影推理（shadow inference）：用真实输入跑一次策略，动作只落盘、绝不发给机械臂。

和 lerobot-rollout 用的是同一套装配：policy + 预/后处理管线 + SyncInferenceEngine +
dataset_features。唯一区别：拿到动作之后不调用 robot.send_action()。

设备、任务、相机改名默认读 station.env；模型默认取 models/ 下唯一的 pretrained_model。

用法：
  # 不接机械臂的自检（合成黑图 + 零状态），用来验证模型/管线能不能起来
  python shadow_inference.py --dummy --samples 1

  # 真实输入：相机画面和关节状态都是真的，动作不出电脑
  python shadow_inference.py --samples 1
"""
from __future__ import annotations

import argparse
import builtins
import json
import os
import time
from copy import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import station_config
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.rollout.inference.sync import SyncInferenceEngine
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts

OBS_STR = "observation"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

ROOT = station_config.ROOT
_cfg = station_config.get
_models = sorted(ROOT.glob("models/*/pretrained_model"))
DEFAULT_MODEL = os.environ.get("MODEL_PATH") or (str(_models[0]) if len(_models) == 1 else "")
DEFAULT_TOKENIZER = str(ROOT / "models/paligemma_tokenizer")
CAM_W, CAM_H, CAM_FPS = int(_cfg("TOP_CAM_WIDTH", "1280")), int(_cfg("TOP_CAM_HEIGHT", "720")), int(_cfg("TOP_CAM_FPS", "30"))
CAM_FOURCC = _cfg("TOP_CAM_FOURCC") or None


def top_cam_config(dev: str) -> OpenCVCameraConfig:
    return OpenCVCameraConfig(
        index_or_path=Path(dev), width=CAM_W, height=CAM_H, fps=CAM_FPS, fourcc=CAM_FOURCC, warmup_s=3
    )


def parse_rename(spec: str) -> dict[str, str]:
    out = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(f"RENAME_MAP 片段格式错误: {pair!r}")
        src, dst = pair.split("=", 1)
        out[src.strip()] = dst.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="SO101 影推理：真实输入跑一次策略，动作不回机器人")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL, required=not DEFAULT_MODEL)
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER)
    ap.add_argument("--task", default=_cfg("TASK"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rename-map", default=_cfg("RENAME_MAP"))
    ap.add_argument("--port", default=_cfg("FOLLOWER_PORT"))
    ap.add_argument("--robot-id", default=_cfg("FOLLOWER_ID", "follower"))
    ap.add_argument("--top-cam", default=_cfg("TOP_CAM"))
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--dummy", action="store_true", help="不接机械臂：合成黑图 + 零状态")
    ap.add_argument(
        "--readonly",
        action="store_true",
        help="只读真实输入：直连串口读关节位置 + 直接开相机，不调用 robot.connect()、不写任何电机寄存器",
    )
    ap.add_argument("--yes", action="store_true", help="跳过交互确认")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = args.device
    rename_map = parse_rename(args.rename_map)

    print("=" * 64)
    print("SO101 影推理（shadow inference）")
    print(f"模型  : {args.model_dir}")
    print(f"任务  : {args.task}")
    mode_label = "自检 dummy（不接臂）" if args.dummy else ("真实输入·只读（相机 + 关节位置，不写电机）" if args.readonly else "真实输入（完整 robot.connect）")
    print(f"设备  : {device}   模式: {mode_label}")
    print("说明  : 本脚本不会向机械臂发送任何动作")
    print("=" * 64)

    # --- 1. 策略 ---
    print("\n[1/5] 加载策略 ...")
    pcfg = PreTrainedConfig.from_pretrained(args.model_dir)
    policy_cls = get_policy_class(pcfg.type)
    policy = policy_cls.from_pretrained(args.model_dir, config=pcfg)
    policy = policy.to(device).eval()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"      type={pcfg.type}  class={policy_cls.__name__}  参数={n_params / 1e6:.1f}M  运行设备={device}")

    # --- 2. 预/后处理管线（覆盖掉 checkpoint 里那些跨机路径与 CPU 设定）---
    print("[2/5] 装配预/后处理管线 ...")
    pre, post = make_pre_post_processors(
        policy_cfg=pcfg,
        pretrained_path=args.model_dir,
        dataset_stats=None,
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": args.tokenizer_dir},
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    print(f"      步骤: {' -> '.join(type(s).__name__ for s in pre.steps)}")
    print(f"      相机改名: {rename_map}")

    # --- 3. 观测源 ---
    print("[3/5] 准备观测源 ...")
    robot = None
    readonly_bus = None
    readonly_cams: dict = {}
    robot_type = "so_follower"

    if args.dummy:
        observation_features_hw = {f"{j}.pos": float for j in JOINTS}
        observation_features_hw["top"] = (CAM_H, CAM_W, 3)
        action_features_hw = {f"{j}.pos": float for j in JOINTS}

        def read_obs() -> dict:
            obs = {f"{j}.pos": 0.0 for j in JOINTS}
            obs["top"] = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
            return obs

        print("      dummy：黑图 + 零状态")
    elif args.readonly:
        robot_cfg = SOFollowerRobotConfig(
            port=args.port,
            id=args.robot_id,
            cameras={},
            disable_torque_on_disconnect=False,  # 只读模式：断开时也绝不碰力矩
        )
        robot = make_robot_from_config(robot_cfg)
        readonly_bus = robot.bus
        readonly_bus.connect()  # handshake 只做 ping，不写寄存器
        cam = OpenCVCamera(top_cam_config(args.top_cam))
        cam.connect()
        readonly_cams["top"] = cam

        observation_features_hw = {f"{j}.pos": float for j in JOINTS}
        for name in readonly_cams:
            observation_features_hw[name] = (CAM_H, CAM_W, 3)
        action_features_hw = {f"{j}.pos": float for j in JOINTS}

        def read_obs() -> dict:
            obs = {f"{m}.pos": float(v) for m, v in readonly_bus.sync_read("Present_Position").items()}
            for name, cam in readonly_cams.items():
                obs[name] = cam.read_latest()
            return obs

        print("      只读模式：串口仅读取 Present_Position，相机直连；未写任何电机寄存器")
    else:
        if not args.yes:
            print("\n即将连接从臂：只读关节状态 + 读顶部相机，不会发送动作。")
            if input("输入 yes 继续: ").strip() != "yes":
                print("已取消。")
                return 0
        robot_cfg = SOFollowerRobotConfig(
            port=args.port,
            id=args.robot_id,
            disable_torque_on_disconnect=False,  # 断开时保持力矩，别让臂掉下来
            cameras={"top": top_cam_config(args.top_cam)},
        )
        robot = make_robot_from_config(robot_cfg)
        _orig_input = builtins.input
        builtins.input = lambda *a, **k: ""  # 标定询问一律回车 = 用现有标定文件
        try:
            robot.connect()
        finally:
            builtins.input = _orig_input
        robot_type = robot.name
        all_obs = robot.observation_features
        observation_features_hw = {
            k: v for k, v in all_obs.items() if isinstance(v, tuple) or (v is float and k.endswith(".pos"))
        }
        action_features_hw = {k: v for k, v in robot.action_features.items() if k.endswith(".pos")}

        def read_obs() -> dict:
            return robot.get_observation()

        print(f"      从臂已连接: {args.port} (id={args.robot_id})；断开时保留力矩")

    # --- 4. dataset features + 引擎 ---
    print("[4/5] 构造 dataset features 与推理引擎 ...")
    teleop_ap, robot_ap, robot_op = make_default_processors()
    action_feats = aggregate_pipeline_dataset_features(
        pipeline=teleop_ap, initial_features=create_initial_features(action=action_features_hw), use_videos=True
    )
    obs_feats = aggregate_pipeline_dataset_features(
        pipeline=robot_op, initial_features=create_initial_features(observation=observation_features_hw), use_videos=True
    )
    dataset_features = combine_feature_dicts(action_feats, obs_feats)
    ordered_action_keys = list(action_features_hw.keys())
    state_names = list(dataset_features["observation.state"]["names"])

    engine = SyncInferenceEngine(
        policy=policy,
        preprocessor=pre,
        postprocessor=post,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task=args.task,
        device=device,
        robot_type=robot_type,
    )
    engine.start()
    print(f"      状态键: {state_names}")
    print(f"      动作键: {ordered_action_keys}")

    # --- 5. 推理 ---
    print("\n[5/5] 推理 ...\n")
    records = []
    try:
        for i in range(args.samples):
            t0 = time.perf_counter()
            obs = read_obs()
            t_read = time.perf_counter() - t0

            obs_frame = build_dataset_frame(dataset_features, obs, prefix=OBS_STR)
            state = np.asarray(obs_frame["observation.state"], dtype=np.float32)

            # 诊断：整条 chunk 先算一遍，同时起到 CUDA 预热作用（避免把首次调用开销算进耗时）
            batch = prepare_observation_for_inference(copy(obs_frame), torch.device(device), args.task, robot_type)
            batch = pre(batch)
            t2 = time.perf_counter()
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(batch)
            t_chunk = time.perf_counter() - t2
            chunk = chunk.squeeze(0).float().cpu()
            try:
                chunk = post(chunk.to(device)).float().cpu().numpy()  # 反归一化回真实单位
                chunk_units = "真实单位"
            except Exception:
                chunk = chunk.numpy()
                chunk_units = "归一化单位（该策略的后处理管线不适用于整条 chunk）"

            t1 = time.perf_counter()
            action = engine.get_action(obs_frame)  # 生产路径：pre -> select_action -> post
            t_action = time.perf_counter() - t1

            a = action.float().cpu().numpy()
            print(f"--- 第 {i + 1} 次推理 ---")
            print("真实关节状态: " + "  ".join(f"{n.split('.')[0]}={v:7.2f}" for n, v in zip(state_names, state)))
            print("策略输出动作: " + "  ".join(f"{n.split('.')[0]}={v:7.2f}" for n, v in zip(ordered_action_keys, a)))
            print(f"动作 chunk  : shape={tuple(chunk.shape)}（{chunk_units}） 首步={np.round(chunk[0], 2).tolist()}")
            print("              注：chunk 与上面的单步动作是两次独立随机采样，数值不完全相同")
            print(f"耗时        : 读观测 {t_read * 1000:.0f}ms | 单步推理 {t_action * 1000:.0f}ms | chunk 推理 {t_chunk * 1000:.0f}ms")
            print()

            records.append(
                {
                    "sample": i + 1,
                    "state": state.tolist(),
                    "state_names": state_names,
                    "action": a.tolist(),
                    "action_names": ordered_action_keys,
                    "chunk_shape": list(chunk.shape),
                    "chunk_first": chunk[0].tolist(),
                    "latency_ms": {"read": t_read * 1000, "action": t_action * 1000, "chunk": t_chunk * 1000},
                }
            )
            if i + 1 < args.samples:
                time.sleep(args.interval)
    finally:
        engine.stop()
        if args.readonly:
            for cam in readonly_cams.values():
                cam.disconnect()
            if readonly_bus is not None:
                readonly_bus.disconnect(disable_torque=False)  # 只关串口，力矩不动
                print("只读模式收尾：相机与串口已关闭，力矩未改动，未写任何电机寄存器")
        elif robot is not None:
            robot.disconnect()
            print("从臂已断开（disable_torque_on_disconnect=False，力矩保留，姿态不动）")

    out = (
        Path(args.out).expanduser()
        if args.out
        else ROOT / "logs" / f"shadow_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "stamp": datetime.now().isoformat(timespec="seconds"),
                "mode": "dummy" if args.dummy else ("real_readonly" if args.readonly else "real_connected"),
                "model_dir": args.model_dir,
                "policy_type": pcfg.type,
                "task": args.task,
                "device": device,
                "rename_map": rename_map,
                "samples": records,
                "sent_action_to_robot": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"记录已写入: {out}")
    print("\n本次没有向机械臂发送任何动作。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())