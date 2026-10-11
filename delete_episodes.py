#!/usr/bin/env python3
"""按 episode_index 删除若干条，生成一个干净的新数据集（原数据集保持不变）。

用法:
  python delete_episodes.py --root <源数据集> --out <新数据集目录> --delete 22,27,29
  python delete_episodes.py --root <源数据集> --delete 22,27,29 --dry-run
"""
import argparse
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--delete", required=True, help="逗号分隔的 episode_index（从 0 开始，和录制终端显示的一致）")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

idx = sorted({int(x) for x in a.delete.replace(" ", "").split(",") if x != ""})

ds = LeRobotDataset(repo_id="local/" + Path(a.root).name, root=a.root)
total = ds.meta.total_episodes
print(f"源数据集: {a.root}")
print(f"  共 {total} 集 / {ds.meta.total_frames} 帧 / {ds.meta.fps}fps")
bad = [i for i in idx if i < 0 or i >= total]
if bad:
    sys.exit(f"错误: episode_index 越界 {bad}（有效范围 0..{total-1}）")
print(f"  将要删除 {len(idx)} 集: {idx}")
print(f"  删除后剩 {total - len(idx)} 集")
keep = [i for i in range(total) if i not in idx]
print(f"  保留的集（原编号）: {keep}")

if a.dry_run:
    print("空跑结束：未做任何修改")
    sys.exit(0)

if not a.out:
    sys.exit("错误: 需要 --out 指定新数据集目录")

new = delete_episodes(ds, idx, output_dir=a.out, repo_id="local/" + Path(a.out).name)
print(f"完成: {new.meta.total_episodes} 集 / {new.meta.total_frames} 帧 -> {a.out}")
