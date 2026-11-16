#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 head 的像素从身体/衣服部件里抠掉。

问题: AI 分层把下巴(脸的一部分)同时给了 topwear, 结果身体部件里带着下巴。
head 在 spine 里的绘制顺序在 topwear 之后(盖在上面), 所以从 topwear 里扣掉
head 覆盖的像素不会露洞, 却能让"头动/身不动"时不再拖着一块下巴。

用法:
    python cut_head_from_body.py output/fashu_hero_338 [--only topwear,leg_L] [--erode 2] [--dry]

只处理 BODY_PARTS 里列出的部件(默认 topwear), 其余部件不动。

--erode N: 蒙版先内缩 N px 再扣(默认 2)。spine 网格的边界顶点会内缩 1px,
如果按 head 的精确轮廓切, 下巴和衣领之间会露出一条 1px 透明缝(看着就是白边)。
留 2px 垫边被 head 盖住看不见, 却能把缝焊死。
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

KEEP = {"head", "headwear_overlay", "irides", "objects"}
BODY_PARTS = ("topwear",)


def load_mask(parts, p):
    im = np.array(Image.open(os.path.join(parts, p["image"])).convert("RGBA"))
    m = np.zeros((im.shape[0], im.shape[1]), bool)
    m[im[:, :, 3] > 20] = True
    return im, m


def to_canvas(parts, p):
    im, m = load_mask(parts, p)
    cv = np.zeros((4096, 4096), bool)
    cv[p["y"]:p["y"] + m.shape[0], p["x"]:p["x"] + m.shape[1]] = m
    return im, m, cv


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    if not args:
        print("用法: cut_head_from_body.py <输出目录> [--only a,b] [--dry]")
        return 1
    D = args[0]
    parts = os.path.join(D, "parts")
    cp = os.path.join(parts, "coords.json")
    if not os.path.isfile(cp):
        print(f"没有 coords.json: {cp}")
        return 1
    dry = "--dry" in flags
    only = None
    erode = 2
    for f in flags:
        if f.startswith("--only="):
            only = tuple(x.strip() for x in f.split("=", 1)[1].split(","))
        if f.startswith("--erode="):
            erode = int(f.split("=", 1)[1])
    targets = only or BODY_PARTS

    co = json.load(open(cp, encoding="utf-8"))
    pmap = {p["name"]: p for p in co["parts"]}
    hp = pmap.get("head")
    if hp is None:
        print("没有 head 部件")
        return 1

    _, _, head_cv = to_canvas(parts, hp)
    if erode > 0:
        k = np.ones((erode * 2 + 1, erode * 2 + 1), np.uint8)
        head_cv = cv2.erode(head_cv.astype(np.uint8), k).astype(bool)
    H, W = head_cv.shape

    changed_any = False
    for name in targets:
        p = pmap.get(name)
        if p is None or name in KEEP:
            continue
        im, m, cv = to_canvas(parts, p)
        cut = cv & head_cv
        if not cut.any():
            print(f"  {name}: 与 head 无重叠")
            continue
        # 只抠掉 head 覆盖的部分, 保留部件自己的其余像素
        local = cut[p["y"]:p["y"] + m.shape[0], p["x"]:p["x"] + m.shape[1]]
        rem = int(local.sum())
        print(f"  {name}: 抠掉 {rem}px (与 head 重叠)")
        if dry:
            continue
        new = im.copy()
        new[local] = 0
        ys, xs = np.nonzero(new[:, :, 3] > 20)
        if len(xs) == 0:
            print(f"  {name}: 抠完什么都不剩, 跳过")
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        crop = new[y0:y1, x0:x1]
        Image.fromarray(crop, "RGBA").save(os.path.join(parts, p["image"]))
        nx, ny = p["x"] + x0, p["y"] + y0
        if (nx, ny, crop.shape[1], crop.shape[0]) != (p["x"], p["y"], p["width"], p["height"]):
            print(f"  {name}: 窗口 ({p['x']},{p['y']},{p['width']},{p['height']}) -> "
                  f"({nx},{ny},{crop.shape[1]},{crop.shape[0]})")
        p["x"], p["y"] = nx, ny
        p["width"], p["height"] = crop.shape[1], crop.shape[0]
        changed_any = True

    if changed_any and not dry:
        for path in (cp, os.path.join(D, "coords.json")):
            if os.path.isfile(path) or path == cp:
                json.dump(co, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("coords 已更新")
    return 0


if __name__ == "__main__":
    sys.exit(main())
