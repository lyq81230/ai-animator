# -*- coding: utf-8 -*-
"""把"多切进去的其它部位"从某个部件里剥掉。

AI 分层常把一整片上本身叠进某个部件（最典型：head 里把整个上半身都包了）。
这些像素和它**下面那层**的像素几乎一样 —— 直接把重复的部分从上层挖掉，
露出的还是同一张图，画面不变；但这个部件就不会再带着别人的身子动了。

用法:
    python trim_overlap.py output/xxx --parts head,arm_upper_L,arm_upper_R
"""
import json
import os
import sys

import numpy as np
from PIL import Image


def load_parts(work):
    coords = json.load(open(os.path.join(work, "coords.json"), encoding="utf-8"))
    imgs = {}
    for p in coords["parts"]:
        fp = os.path.join(work, "parts", p["image"])
        imgs[p["name"]] = np.array(Image.open(fp).convert("RGBA")).astype(np.int16) \
            if os.path.isfile(fp) else None
    return coords, imgs


def composite(parts, imgs, W, H, skip=None, before_index=None):
    """把部件叠成一张画布; skip 指定跳过的部件名。"""
    out = np.zeros((H, W, 4), np.float32)
    for i, p in enumerate(parts):
        if skip and p["name"] in skip:
            continue
        if before_index is not None and i >= before_index:
            break
        im = imgs.get(p["name"])
        if im is None:
            continue
        x, y = p["x"], p["y"]
        h, w = im.shape[0], im.shape[1]
        src = im.astype(np.float32)
        a = src[..., 3:4] / 255.0
        dst = out[y:y + h, x:x + w]
        dst[..., :3] = src[..., :3] * a + dst[..., :3] * (1 - a)
        dst[..., 3:4] = np.clip(src[..., 3:4] + dst[..., 3:4] * (1 - a), 0, 255)
        out[y:y + h, x:x + w] = dst
    return out


def trim(work, names, tol=14, under_alpha=200, dry=False):
    coords, imgs = load_parts(work)
    W, H = coords["canvas"]["w"], coords["canvas"]["h"]
    parts = coords["parts"]
    before = composite(parts, imgs, W, H)

    report = {}
    for idx, p in enumerate(parts):
        if p["name"] not in names:
            continue
        im = imgs.get(p["name"])
        if im is None:
            continue
        under = composite(parts, imgs, W, H, skip={p["name"]}, before_index=idx)
        x, y = p["x"], p["y"]
        h, w = im.shape[0], im.shape[1]
        u = under[y:y + h, x:x + w]
        cur = im.astype(np.int16)
        a = cur[..., 3]
        diff = np.abs(cur[..., :3].astype(np.int16) - u[..., :3]).max(2)
        dup = (a > 12) & (u[..., 3] > under_alpha) & (diff <= tol)
        n = int(dup.sum())
        report[p["name"]] = n
        if n == 0 or dry:
            continue
        out = cur.copy()
        out[..., 3] = np.where(dup, 0, a)
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(
            os.path.join(work, "parts", p["image"]))
        imgs[p["name"]] = out
        print(f"  {p['name']:14s} 剥掉重复 {n}px "
              f"(剩 {int((out[...,3]>12).sum())}px)")

    after = composite(parts, imgs, W, H)
    d = np.abs(before[..., :3] - after[..., :3]).max(2)
    changed = int(((d > 10) & ((before[..., 3] > 12) | (after[..., 3] > 12))).sum())
    print(f"  合成图变化像素: {changed}")
    return report, changed, before, after


if __name__ == "__main__":
    args = sys.argv[1:]
    work = args[0]
    names = []
    for i, a in enumerate(args):
        if a == "--parts":
            names = args[i + 1].split(",")
    dry = "--dry" in args
    print(f"== {work}  目标 {names}")
    trim(work, names, dry=dry)
