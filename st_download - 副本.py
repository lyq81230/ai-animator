#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""st_download.py — 下载 AI 拆图所需的 3 个模型仓库到 ST_MODELS 目录。"""
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seethrough import MODEL_JOBS, paths  # noqa: E402

from huggingface_hub import snapshot_download  # noqa: E402

root = os.environ.get("ST_MODELS") or paths()[2]
os.makedirs(root, exist_ok=True)

for repo, sub, patterns in MODEL_JOBS:
    dst = os.path.join(root, sub)
    if os.path.isdir(dst) and os.listdir(dst):
        print(f"=== SKIP {repo} (已存在)", flush=True)
        continue
    os.makedirs(dst, exist_ok=True)
    print(f"=== 下载 {repo} -> {dst}", flush=True)
    t0 = time.time()
    for attempt in range(1, 6):
        try:
            snapshot_download(repo_id=repo, local_dir=dst, max_workers=8,
                              allow_patterns=patterns)
            print(f"=== 完成 {repo} ({time.time() - t0:.0f}s)", flush=True)
            break
        except Exception as ex:  # noqa: BLE001
            print(f"!!! 第 {attempt} 次失败: {ex}", flush=True)
            time.sleep(8)
    else:
        print(f"### 放弃 {repo}", flush=True)
        sys.exit(1)

print("模型全部就绪", flush=True)
