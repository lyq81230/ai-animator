# -*- coding: utf-8 -*-
"""把某个矩形区域里的像素从一个部件"搬"到另一个部件。

用途: 手/武器被 AI 分进了躯干或腿的层里, 而目标部件的 bbox 又没盖住这一块 ——
只删会丢内容(拳头整个消失), 所以要搬过去并扩张目标部件的包围盒。
颜色一律取源图, alpha 取源图不透明处。

用法:
    python move_region.py output/sprite_quan_white_336 \
        --from topwear --to arm_hand_R --box 196,214,246,266
"""
import json
import os
import sys

import numpy as np
from PIL import Image


def arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def main():
    D = sys.argv[1]
    src_p = arg("--from")
    dst_p = arg("--to")
    box = [int(v) for v in arg("--box").split(",")]
    x0, y0, x1, y1 = box
    cp = os.path.join(D, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    W, H = co["canvas"]["w"], co["canvas"]["h"]
    char = os.path.basename(os.path.normpath(D))
    src = np.array(Image.open(os.path.join("E:/pic", char + ".png")).convert("RGBA"))
    by = {p["name"]: p for p in co["parts"]}
    A, B = by[src_p], by[dst_p]

    def canvas_of(p):
        a = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))
        m = np.zeros((H, W, 4), np.uint8)
        m[p["y"]:p["y"] + a.shape[0], p["x"]:p["x"] + a.shape[1]] = a
        return m

    ca, cb = canvas_of(A), canvas_of(B)
    reg = np.zeros((H, W), bool)
    reg[y0:y1 + 1, x0:x1 + 1] = True
    take = reg & (src[..., 3] > 40) & (ca[..., 3] > 8)
    if not take.any():
        print("区域里没有可搬的像素")
        return
    ca[..., 3][take] = 0                       # 从原部件移除
    cb[take] = src[take]                       # 搬到目标部件(颜色取源图)

    for p, c in ((A, ca), (B, cb)):
        ys, xs = np.nonzero(c[..., 3] > 8)
        a0, a1 = int(xs.min()), int(xs.max())
        b0, b1 = int(ys.min()), int(ys.max())
        Image.fromarray(c[b0:b1 + 1, a0:a1 + 1]).save(
            os.path.join(D, "parts", p["image"]))
        p.update({"x": a0, "y": b0, "width": a1 - a0 + 1, "height": b1 - b0 + 1})
    print(f"搬了 {int(take.sum())}px: {src_p} -> {dst_p}")
    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pc = os.path.join(D, "parts", "coords.json")
    if os.path.isfile(pc):
        json.dump(co, open(pc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
