#!/usr/bin/env python3
"""训练入口。

等价于 `python -m lerobot.scripts.lerobot_train ...`，多做两件事，都是为了能直接从
models/pi05_base 微调：

1. π0.5 的分词器 google/paligemma-3b-pt-224 改从项目自带的 models/paligemma_tokenizer 读。
   这个仓库在 HuggingFace 上要申请权限，国内服务器也连不上；checkpoint 里记录的仍是原名。
2. pi05_base 是 HuggingFace 上较新的版本，预处理配置里把相对动作这一步叫
   relative_actions_processor，项目自带的 LeRobot 0.5.2 里同一个类叫 delta_actions_processor，
   不补这个别名就加载不了。保存的 checkpoint 用的是 0.5.2 的名字，推理机照常加载。
3. 只保留数据集真有的相机。pi05_base 声明了三路相机（smolvla_base 也是 camera1/2/3），我们只有
   顶部一路，另外两路会被补成掩码掉的空图：注意力看不到、位置编码也跳过，结果和没有一样，但照样过
   视觉编码器和整个 VLM，白占算力。去掉后输出不变，训练和推理都快一截；checkpoint 里也只剩这一路相机。

多卡时 accelerate 在每张卡上各跑一份本脚本，上面的补丁每个进程都要打。
"""
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PALIGEMMA_TOKENIZER = "google/paligemma-3b-pt-224"
LOCAL_TOKENIZER = ROOT / "models" / "paligemma_tokenizer"


def use_local_tokenizer() -> None:
    if not LOCAL_TOKENIZER.is_dir():
        return
    from transformers import AutoTokenizer

    orig = AutoTokenizer.from_pretrained.__func__

    def from_pretrained(cls, name, *args, **kwargs):
        if str(name) == PALIGEMMA_TOKENIZER:
            name = str(LOCAL_TOKENIZER)
        return orig(cls, name, *args, **kwargs)

    AutoTokenizer.from_pretrained = classmethod(from_pretrained)


def alias_newer_processor_names() -> None:
    from lerobot.processor import ProcessorStepRegistry, RelativeActionsProcessorStep

    # 直接写注册表，不用 register()：它会改掉类上记录的名字，checkpoint 就会存成新名字
    ProcessorStepRegistry._registry.setdefault("relative_actions_processor", RelativeActionsProcessorStep)


def drop_empty_cameras(policy_cfg, ds_meta, rename_map) -> list[str]:
    from lerobot.configs import FeatureType

    provided = {(rename_map or {}).get(k, k) for k in ds_meta.camera_keys}
    dropped = [
        k for k, ft in policy_cfg.input_features.items() if ft.type is FeatureType.VISUAL and k not in provided
    ]
    if dropped and len(dropped) < len(policy_cfg.image_features):
        for k in dropped:
            del policy_cfg.input_features[k]
        return dropped
    return []


def drop_empty_cameras_on_make_policy() -> None:
    import lerobot.scripts.lerobot_train as lerobot_train

    orig = lerobot_train.make_policy

    def make_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None):
        if ds_meta is not None and cfg.input_features:
            dropped = drop_empty_cameras(cfg, ds_meta, rename_map)
            if dropped and os.environ.get("RANK", "0") == "0":
                logging.info(f"数据集里没有这些相机，不再补空图: {dropped}")
        return orig(cfg, ds_meta=ds_meta, env_cfg=env_cfg, rename_map=rename_map)

    lerobot_train.make_policy = make_policy


if __name__ == "__main__":
    use_local_tokenizer()
    alias_newer_processor_names()
    drop_empty_cameras_on_make_policy()
    from lerobot.scripts.lerobot_train import main

    main()
