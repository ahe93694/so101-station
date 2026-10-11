#!/usr/bin/env python3
"""SO101 推理前置检查（不接机械臂，只读文件 + 读环境）。

由 start_inference.sh 调用，也可以单独跑：

    python inference_preflight.py                                    # 只查环境/设备
    python inference_preflight.py --model-dir M --cameras top        # 追加查模型接口
    python inference_preflight.py --model-dir M --load               # 再真加载一次策略
    python inference_preflight.py --model-dir M --manifest x.json    # 顺带写运行台账

退出码：0 = 通过（可能带 WARN）；1 = 有硬错误。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import station_config

N_JOINTS = 6
# 推理只需要从臂 + 顶部相机
SERIAL_IDS = [p for p in (station_config.get("FOLLOWER_PORT"),) if p]
CAM_IDS = [p for p in (station_config.get("TOP_CAM"),) if p]
BUSY_PATTERNS = ("record_with_stream", "lerobot_record", "rollout_with_stream", "lerobot_rollout")

_failed = [False]


def _exists(path) -> bool:
    """Path.exists() 遇到不可穿越的父目录（例如 /root）会抛 PermissionError。"""
    try:
        return Path(path).exists()
    except OSError:
        return False



def ok(msg: str) -> None:
    print(f"[ OK ] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def bad(msg: str) -> None:
    _failed[0] = True
    print(f"[FAIL] {msg}", flush=True)


_SKIP_EXEC = {"grep", "rg", "find", "pgrep", "fgrep", "egrep", "cat", "head", "tail", "less", "more", "vim", "nano", "sed", "awk", "git", "ps"}


def _busy_processes() -> list[str]:
    """找出真正在跑的数采/推理进程。

    只认 argv[0] 不是查看类命令（grep/find/ps/...）的进程，否则自己搜代码时的
    `grep -rl lerobot_rollout ~` 会被误判成「正在推理」。
    """
    busy: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().decode(errors="replace").replace("\0", " ").strip()
        except OSError:
            continue
        if not cmd:
            continue
        if os.path.basename(cmd.split()[0]) in _SKIP_EXEC:
            continue
        if any(k in cmd for k in BUSY_PATTERNS):
            busy.append(f"{entry.name}  {cmd[:160]}")
    return busy


def check_environment(device: str) -> None:
    print("== 环境 ==")
    import lerobot
    import torch

    ok(f"python {sys.version.split()[0]} | lerobot {lerobot.__file__}")
    ok(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()}")
    if device == "cuda":
        if torch.cuda.is_available():
            ok(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            bad("DEVICE=cuda 但 torch.cuda.is_available() 为 False（没显卡，或装的是 CPU 版 torch）")
    else:
        warn(f"DEVICE={device}：CPU 推理一般跟不上 30Hz 控制循环，只适合排查加载问题")

    calib = os.environ.get("HF_LEROBOT_CALIBRATION", "")
    if calib and _exists(Path(calib) / "robots" / "so_follower" / "follower.json"):
        ok(f"标定目录: {calib}")
    else:
        bad(f"HF_LEROBOT_CALIBRATION 没指向有效标定目录（当前 {calib!r}，缺 robots/so_follower/follower.json）")

    print("== 设备 ==")
    for path in SERIAL_IDS + CAM_IDS:
        if _exists(path):
            ok(f"在线   {path}")
        else:
            warn(f"不在线 {path}")
    if not all(_exists(p) for p in SERIAL_IDS):
        warn("串口不在线不会让前置检查失败，但真跑必然连不上；确认从臂供电与 USB 接线")

    busy = _busy_processes()
    if busy:
        bad("已有数采/推理进程在跑（相机与串口都是独占设备）：\n       " + "\n       ".join(busy))
    else:
        ok("没有其它数采/推理进程占用设备")


def _iter_abs_paths(obj, prefix="$"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _iter_abs_paths(value, f"{prefix}.{key}")
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield from _iter_abs_paths(value, f"{prefix}[{idx}]")
    elif isinstance(obj, str) and obj.startswith("/"):
        yield prefix, obj


def _feature_dim(features, key):
    entry = features.get(key) or {}
    shape = entry.get("shape") or []
    return shape[0] if shape else None


def check_model(model_dir: Path, cameras, rename_map, device, task, load, manifest, meta) -> None:
    print("== 模型 ==")
    if not model_dir.is_dir():
        bad(f"模型目录不存在: {model_dir}")
        return
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        bad(f"缺少 config.json: {model_dir}（--policy.path 要的是导出的 pretrained_model 目录，不是单个权重文件）")
        return
    if not (model_dir / "model.safetensors").is_file():
        bad(f"缺少 model.safetensors: {model_dir}")
        return
    cfg = json.loads(cfg_path.read_text())
    ok(f"模型目录: {model_dir}")

    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        path = model_dir / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            warn(f"{name} 解析失败: {exc}")
            continue
        for where, value in _iter_abs_paths(data):
            if not _exists(value):
                hint = "（/root/... 是训练机上的路径）" if value.startswith("/root/") else ""
                msg = f"{name} 里 {where} = {value} 在本机不可用{hint}：跨机拷贝后没改路径就会这样"
                if where.endswith("pretrained_path"):
                    warn(msg + "（该字段只在训练时用于初始化基座，推理不使用，放行）")
                else:
                    bad(msg)

    policy_type = cfg.get("type")
    try:
        from lerobot.policies.factory import get_policy_class

        policy_cls = get_policy_class(policy_type)
        ok(f"策略类型 {policy_type} 已注册 -> {policy_cls.__name__}")
    except Exception as exc:
        bad(f"策略类型 {policy_type} 不在本机 runtime 的注册表里（factory.py 只认它列出的类型）: {exc}")
        return

    input_features = cfg.get("input_features") or {}
    model_cams = sorted(k for k in input_features if k.startswith("observation.images."))
    effective = {}
    for name in cameras:
        key = f"observation.images.{name}"
        effective[rename_map.get(key, key)] = name
    missing = [k for k in model_cams if k not in effective]
    extra = [k for k in effective if k not in model_cams]
    if missing:
        if len(missing) == len(model_cams):
            bad(f"模型声明的相机一个都对不上：模型要 {model_cams}，本次给 {sorted(effective)}（键名不同就用 RENAME_MAP 映射）")
        else:
            warn(f"模型声明了这些相机、本次没有: {missing}（pi05 会用零掩码补空位，能跑；确认训练时这些视角本来就是空的）")
    if extra:
        warn(f"本次会多出模型没用到的相机: {extra}")
    if model_cams and not missing:
        ok(f"相机 schema 对齐: {model_cams}")
    if not model_cams:
        warn("config.json 里没有 observation.images.*，确认是不是纯状态策略")

    pre_json = model_dir / "policy_preprocessor.json"
    if pre_json.is_file():
        try:
            pdata = json.loads(pre_json.read_text())
        except Exception:
            pdata = {}
        for step in pdata.get("steps", []) or []:
            if step.get("registry_name") == "rename_observations_processor":
                model_rename = (step.get("config") or {}).get("rename_map") or {}
                if model_rename and not rename_map:
                    warn(
                        f"checkpoint 自带改名表 {model_rename}，但本次 RENAME_MAP 为空："
                        "rollout 会用空表覆盖它，观测键名会对不上——必须显式传 RENAME_MAP"
                    )
                elif model_rename and rename_map and model_rename != rename_map:
                    warn(f"checkpoint 自带改名表 {model_rename} 与本次 RENAME_MAP 不同，确认这是有意的")

    action_dim = _feature_dim(cfg.get("output_features") or {}, "action")
    state_dim = _feature_dim(input_features, "observation.state")
    if action_dim == N_JOINTS:
        ok(f"action 维度 {action_dim} 与本机 6 关节一致")
    elif action_dim is None:
        warn("config.json 没有 action 维度，跳过检查")
    else:
        bad(f"模型 action 维度是 {action_dim}，本机机械臂是 {N_JOINTS} 维")
    if state_dim == N_JOINTS:
        ok(f"observation.state 维度 {state_dim} 与数采一致")
    elif state_dim is None:
        warn("config.json 没有 observation.state 维度，跳过检查")
    elif state_dim > N_JOINTS:
        warn(f"模型 observation.state 期望 {state_dim} 维、本机只有 {N_JOINTS} 维：依赖策略侧 padding，确认训练时就是这么设计的")
    else:
        bad(f"模型 observation.state 维度是 {state_dim}，小于本机 {N_JOINTS} 维")

    if device == "cuda" and str(cfg.get("device", "")).lower() == "cpu":
        warn("config.json 里写的是 device=cpu，本次用 --device=cuda 覆盖；加载报设备相关错误时回来看这行")

    if manifest is not None:
        data = dict(meta)
        data.update(
            {
                "stamp": datetime.now().isoformat(timespec="seconds"),
                "model_dir": str(model_dir),
                "model_bytes": (model_dir / "model.safetensors").stat().st_size,
                "model_mtime": datetime.fromtimestamp(
                    (model_dir / "model.safetensors").stat().st_mtime
                ).isoformat(timespec="seconds"),
                "policy_type": policy_type,
                "cameras": cameras,
                "rename_map": rename_map,
                "task": task,
            }
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        ok(f"运行台账: {manifest}")

    if not load:
        return

    print("== 策略实载 ==")
    try:
        import torch

        from lerobot.configs import PreTrainedConfig

        policy_cfg = PreTrainedConfig.from_pretrained(model_dir)
        policy = policy_cls.from_pretrained(model_dir, config=policy_cfg).to(device).eval()
        params = sum(p.numel() for p in policy.parameters())
        ok(f"加载成功: {policy_cls.__name__}  {params / 1e6:.1f}M 参数  device={device}")
    except Exception as exc:
        bad(f"策略加载失败: {type(exc).__name__}: {exc}")
        return

    try:
        import torch

        batch = {}
        for key, feat in input_features.items():
            shape = feat.get("shape") or []
            if feat.get("type") == "VISUAL" and len(shape) == 3:
                batch[key] = torch.zeros(1, *shape, device=device)
            elif feat.get("type") == "STATE" and shape:
                batch[key] = torch.zeros(1, shape[0], device=device)
        batch["task"] = task or ""
        with torch.inference_mode():
            out = policy.predict_action_chunk(batch)
        ok(f"前向冒烟通过，动作 chunk 形状 {tuple(out.shape)}")
    except Exception as exc:
        warn(f"前向冒烟失败（dummy 观测的构造跟策略相关，不一定是模型的问题）: {type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--cameras", default="top")
    ap.add_argument("--rename-map", default=station_config.get("RENAME_MAP"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", default=station_config.get("TASK"))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--inference", default="sync")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--manifest", default="")
    args = ap.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    rename_map = {}
    for pair in (args.rename_map.split(",") if args.rename_map else []):
        if "=" in pair:
            key, value = pair.split("=", 1)
            rename_map[key.strip()] = value.strip()

    meta = {"fps": args.fps, "duration_s": args.duration, "device": args.device, "inference": args.inference}

    check_environment(args.device)
    if args.model_dir:
        check_model(
            Path(args.model_dir).expanduser(),
            cameras,
            rename_map,
            args.device,
            args.task,
            args.load,
            Path(args.manifest).expanduser() if args.manifest else None,
            meta,
        )
    else:
        print("== 模型 ==")
        warn("本次没有给模型：只做环境/设备自检，不接机械臂")

    print()
    if _failed[0]:
        print("前置检查：未通过，别接硬件。")
        return 1
    print("前置检查：通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())