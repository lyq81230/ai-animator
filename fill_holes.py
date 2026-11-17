# -*- coding: utf-8 -*-
"""把拼合后"没有任何部件覆盖"的像素按最近部件补回去。

拆图/改图之后总会漏下几条 1px 的缝(多边形的抗锯齿边、部件边界),
这里统一收尾: 谁离得近就归谁, 颜色取源图。

用法: python fill_holes.py <角色输出目录> [源图路径]
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

DEFAULT_SRC = "E:/pic/sprite_boss_lvbu_white_336.png"


def main():
    D = sys.argv[1]
    # 第二个参数可以是源图; 不传就按输出目录名去 E:/pic 找同名 png
    SRC = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else ""
    if SRC and not os.path.isfile(SRC):
        SRC = ""
    if not SRC:
        cand = os.path.join("E:/pic", os.path.basename(os.path.normpath(D)) + ".png")
        SRC = cand if os.path.isfile(cand) else DEFAULT_SRC
    # 阈值: 默认 8 只补"完全没人画"的地方; 给 200 会把"两边都只有半透明边"的
    # 1px 白缝也算洞补上(渲染出来才不会有一条亮线)
    al = 8
    if "--alpha" in sys.argv:
        al = int(sys.argv[sys.argv.index("--alpha") + 1])
    c = json.load(open(os.path.join(D, "coords.json"), encoding="utf-8"))
    W, H = c["canvas"]["w"], c["canvas"]["h"]
    src = np.array(Image.open(SRC).convert("RGBA"))

    def canvas_of(p):
        a = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))
        m = np.zeros((H, W, 4), np.uint8)
        m[p["y"]:p["y"] + a.shape[0], p["x"]:p["x"] + a.shape[1]] = a
        return m

    lab = np.zeros((H, W), np.int32)
    for i, p in enumerate(c["parts"], 1):
        lab[canvas_of(p)[..., 3] > al] = i
    hole = (src[..., 3] > al) & (lab == 0)
    print(f"空洞 {int(hole.sum())}px")
    if not hole.any():
        return
    _d, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    ys, xs = np.nonzero(hole)
    owner = lab[iy[ys, xs], ix[ys, xs]]
    for i, p in enumerate(c["parts"], 1):
        m = owner == i
        if not m.any():
            continue
        px, py = xs[m], ys[m]
        im = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))
        x0, y0 = min(p["x"], int(px.min())), min(p["y"], int(py.min()))
        x1 = max(p["x"] + im.shape[1] - 1, int(px.max()))
        y1 = max(p["y"] + im.shape[0] - 1, int(py.max()))
        big = np.zeros((y1 - y0 + 1, x1 - x0 + 1, 4), np.uint8)
        big[p["y"] - y0:p["y"] - y0 + im.shape[0],
            p["x"] - x0:p["x"] - x0 + im.shape[1]] = im
        big[py - y0, px - x0] = src[py, px]
        Image.fromarray(big).save(os.path.join(D, "parts", p["image"]))
        p.update({"x": int(x0), "y": int(y0),
                  "width": int(x1 - x0 + 1), "height": int(y1 - y0 + 1)})
        print(f"  {int(m.sum()):4d}px -> {p['name']}")

    lab2 = np.zeros((H, W), np.int32)
    for i, p in enumerate(c["parts"], 1):
        lab2[canvas_of(p)[..., 3] > al] = i
    print(f"剩余空洞 {int(((src[...,3]>al)&(lab2==0)).sum())}px")
    json.dump(c, open(os.path.join(D, "coords.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    # 同步 parts/coords.json: 只写顶层的话, app 收尾"parts/coords.json -> coords.json"
    # 的拷贝会把旧条目盖回来(与 merge_tiny_parts 同一个坑)。
    pc = os.path.join(D, "parts", "coords.json")
    if os.path.isdir(os.path.dirname(pc)):
        json.dump(c, open(pc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
