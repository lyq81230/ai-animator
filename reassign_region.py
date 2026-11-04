# -*- coding: utf-8 -*-
"""把某个部件里"本来属于别人"的像素搬到另一个部件去。

不是删、不是重切：像素原样搬到画布同一位置的目标部件里，再从上件抹掉。
因为叠放顺序不变、坐标不变，**合成图应该完全不变**，只是"归属"改了 ——
于是该部位就不会再跟着错误的骨走（典型：head 里的整个上半身、大臂里的烟雾）。
"""
import json
import os

import numpy as np
from PIL import Image


def _canvas_of(p, arr, W, H):
    c = np.zeros((H, W, 4), np.int16)
    h, w = arr.shape[:2]
    c[p["y"]:p["y"] + h, p["x"]:p["x"] + w] = arr
    return c


def move_region(work, src, dst, mask, verbose=True):
    coords = json.load(open(os.path.join(work, "coords.json"), encoding="utf-8"))
    W, H = coords["canvas"]["w"], coords["canvas"]["h"]
    by = {p["name"]: p for p in coords["parts"]}
    ps, pd = by[src], by[dst]

    a_s = np.array(Image.open(os.path.join(work, "parts", ps["image"])).convert("RGBA"))
    cs = _canvas_of(ps, a_s, W, H)
    sel = mask & (cs[..., 3] > 0)
    n = int(sel.sum())
    if n == 0:
        if verbose:
            print(f"  {src} -> {dst}: 没有要搬的像素")
        return 0

    # 目标部件: 拿画布坐标里的现有内容, 叠加搬过来的像素(只补空缺, 不覆盖已有内容)
    a_d = np.array(Image.open(os.path.join(work, "parts", pd["image"])).convert("RGBA"))
    cd = _canvas_of(pd, a_d, W, H)
    add = sel & (cd[..., 3] <= 12)
    cd[add] = cs[add]

    # 重新裁到有内容的包围盒
    def tight(c):
        m = c[..., 3] > 8
        ys, xs = np.nonzero(m)
        return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

    x0, y0, x1, y1 = tight(cd)
    Image.fromarray(np.clip(cd[y0:y1, x0:x1], 0, 255).astype(np.uint8)).save(
        os.path.join(work, "parts", pd["image"]))
    pd.update({"x": int(x0), "y": int(y0),
               "width": int(x1 - x0), "height": int(y1 - y0)})

    # 源部件: 抹掉搬走的
    cs[sel] = 0
    x0, y0, x1, y1 = tight(cs)
    Image.fromarray(np.clip(cs[y0:y1, x0:x1], 0, 255).astype(np.uint8)).save(
        os.path.join(work, "parts", ps["image"]))
    ps.update({"x": int(x0), "y": int(y0),
               "width": int(x1 - x0), "height": int(y1 - y0)})

    with open(os.path.join(work, "coords.json"), "w", encoding="utf-8") as f:
        json.dump(coords, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"  {src} -> {dst}: 搬 {n}px (目标新增 {int(add.sum())}px)  "
              f"{src} 剩 {ps['width']}x{ps['height']} @({ps['x']},{ps['y']})")
    return n
