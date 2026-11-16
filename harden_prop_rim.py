# -*- coding: utf-8 -*-
"""把武器(prop)轮廓上的半透明像素压成不透明。

为什么: 原画是在白底上画的, 刀身边缘留了一圈半透明抗锯齿像素。武器在游戏里要
压在手/披风上, 这圈半透明像素就会把**背后的东西透出来** —— 背后的手臂每帧在动,
刀刃边缘的颜色跟着一帧一个样, 看着就是"刀身在轻微折断/闪"。把轮廓压成实心之后,
边缘颜色只由武器自己决定, 后面怎么动都不影响。

只动"轮廓一圈"(mask 减去腐蚀后的差集), 内部的半透明(光晕)保持原样。

用法: python harden_prop_rim.py <work_dir>
"""
import json
import os
import sys

import numpy as np
from PIL import Image

MIN_A = 40          # 低于这个当背景(和全流程一致)
CLS = {"prop", "weapon", "shield"}


def run(work):
    cp = os.path.join(work, "coords.json")
    pdir = os.path.join(work, "parts")
    if not os.path.isfile(cp):
        return 0
    co = json.load(open(cp, encoding="utf-8"))
    try:
        from scipy import ndimage as ndi
    except Exception:                                  # noqa: BLE001
        ndi = None
    n = 0
    for p in co["parts"]:
        if str(p.get("cls") or "") not in CLS:
            continue
        f = os.path.join(pdir, p["image"])
        if not os.path.isfile(f):
            continue
        a = np.array(Image.open(f).convert("RGBA"))
        al = a[..., 3]
        mask = al >= MIN_A
        if not mask.any():
            continue
        if ndi is not None:
            rim = mask & ~ndi.binary_erosion(mask, np.ones((3, 3), bool))
        else:                                          # 退化: 直接用边缘判定
            pad = np.pad(mask, 1, constant_values=False)
            er = (pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
            rim = mask & ~er
        fix = rim & (al >= MIN_A) & (al < 255)
        c = int(fix.sum())
        if not c:
            continue
        a[..., 3][fix] = 255
        Image.fromarray(a, "RGBA").save(f)
        print("  武器轮廓压平: %-14s %d px" % (p["name"], c))
        n += c
    if n:
        print("  武器轮廓合计 %d px" % n)
    return n


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    run(sys.argv[1])
