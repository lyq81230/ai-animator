#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""st_download.py — 下载 AI 拆图所需的 3 个模型仓库到 ST_MODELS 目录。"""
import fnmatch
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Xet 通道在本机/镜像上会 401 (cas-server.xethub 未授权), 走普通 HTTP 才稳
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seethrough import MODEL_JOBS, OPTIONAL_JOBS, paths  # noqa: E402

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

api = HfApi()


def expected(repo, patterns):
    """远端应有文件 {相对路径: 字节数}; patterns 为 None 表示全量。"""
    info = api.model_info(repo, files_metadata=True)
    out = {}
    for sib in info.siblings:
        name = sib.rfilename
        if name.split("/")[-1] == ".gitattributes":
            continue
        if patterns and not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        out[name] = sib.size or 0
    return out


def missing_files(dst, want):
    """返回缺失或大小不符的文件 [(路径, 本地字节, 期望字节)]。"""
    bad = []
    for name, size in want.items():
        p = os.path.join(dst, name.replace("/", os.sep))
        have = os.path.getsize(p) if os.path.isfile(p) else -1
        if have < 0 or (size and have != size):
            bad.append((name, have, size))
    return bad


def report(repo, bad):
    print(f"### 未完成 {repo}: 缺/损 {len(bad)} 个文件", flush=True)
    for name, have, size in bad:
        print(f"    - {name}  {have / 1e6:.1f}/{size / 1e6:.1f} MB", flush=True)


root = os.environ.get("ST_MODELS") or paths()[2]
os.makedirs(root, exist_ok=True)
dirty = []

JOBS = list(MODEL_JOBS)
if "--full" in sys.argv or os.environ.get("ST_FULL") == "1":
    print(f"=== 追加可选模型 {len(OPTIONAL_JOBS)} 个", flush=True)
    JOBS += OPTIONAL_JOBS

for repo, sub, patterns in JOBS:
    dst = os.path.join(root, sub)
    os.makedirs(dst, exist_ok=True)

    try:
        want = expected(repo, patterns)
    except Exception as ex:  # noqa: BLE001
        print(f"!!! 取 {repo} 文件清单失败: {ex}", flush=True)
        want = {}

    bad = missing_files(dst, want) if want else []
    if want and not bad:
        print(f"=== SKIP {repo} (已完整 {len(want)} 个文件)", flush=True)
        continue
    if bad:
        print(f"=== 续传 {repo}: 已有 {len(want) - len(bad)}/{len(want)}"
              f" -> {dst}", flush=True)
    else:
        print(f"=== 下载 {repo} -> {dst}", flush=True)

    t0 = time.time()
    for attempt in range(1, 6):
        try:
            snapshot_download(repo_id=repo, local_dir=dst, max_workers=8,
                              allow_patterns=patterns)
            left = missing_files(dst, want) if want else []
            if left:
                print(f"!!! 第 {attempt} 次未下全, 还缺 {len(left)} 个",
                      flush=True)
                continue
            print(f"=== 完成 {repo} ({time.time() - t0:.0f}s)", flush=True)
            break
        except Exception as ex:  # noqa: BLE001
            print(f"!!! 第 {attempt} 次失败: {ex}", flush=True)
        time.sleep(8)
    else:
        dirty.append(repo)

if dirty:
    for repo, sub, patterns in JOBS:
        if repo not in dirty:
            continue
        try:
            report(repo, missing_files(os.path.join(root, sub),
                                       expected(repo, patterns)))
        except Exception as ex:  # noqa: BLE001
            print(f"### 放弃 {repo}: {ex}", flush=True)
    sys.exit(1)

print("模型全部就绪", flush=True)
