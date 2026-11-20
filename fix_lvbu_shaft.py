# -*- coding: utf-8 -*-
"""把吕布武器杆的暗色描边带从 topwear 搬到 objects_L。

原画里杆 = 金色受光面 + 暗色背光面。AI 分层把金色芯判给 objects, 暗色杆身
判给了 topwear —— 武器一摆, 暗色"杆影"留在身上, 看着像武器部件残留在身体
上(用户: "topwear 上有武器部件的残留")。

做法:
    1. 用 objects_L 里的金色像素拟合杆轴 (x = a*y + b, 只取 y[145,260] 段);
    2. topwear 中轴走廊 ±13px 内的暗色像素为候选, 保留与核心种子(±6px)
       连通的连通块(避免把裙甲/袍子的暗边一起卷进来);
    3. 整块搬到 objects_L —— 描边随武器走, 杆才是一根完整的杆。

用法:
    python fix_lvbu_shaft.py [输出目录]   # 默认 output/sprite_boss_lvbu_white_336
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

D = sys.argv[1] if len(sys.argv) > 1 else "output/sprite_boss_lvbu_white_336"
Y0, Y1 = 143, 256          # 杆身走廊的 y 范围(不含斧头区)
HALF = 13                  # 走廊半宽
SEED_HALF = 6              # 核心种子半宽
LUM_MAX = 320              # "暗色"阈值(RGB 和)


def main():
    cp = os.path.join(D, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    W, H = co["canvas"]["w"], co["canvas"]["h"]
    by = {p["name"]: p for p in co["parts"]}
    for n in ("objects_L", "topwear"):
        if n not in by:
            print(f"没有部件 {n}")
            return 1

    def load(p):
        return np.array(Image.open(os.path.join(D, "parts", p["image"]))
                        .convert("RGBA"))

    def canvas_of(p, a):
        m = np.zeros((H, W, 4), np.uint8)
        h, w = a.shape[:2]
        y0, x0 = max(p["y"], 0), max(p["x"], 0)
        y1, x1 = min(p["y"] + h, H), min(p["x"] + w, W)
        m[y0:y1, x0:x1] = a[y0 - p["y"]:y1 - p["y"], x0 - p["x"]:x1 - p["x"]]
        return m

    po, pt = by["objects_L"], by["topwear"]
    a_o, a_t = load(po), load(pt)
    c_o = canvas_of(po, a_o)
    ol_m = c_o[..., 3] > 40
    rgb = c_o[..., :3].astype(int)
    gold = (ol_m & (rgb[..., 0] > 140) & (rgb[..., 1] > 100)
            & (rgb[..., 2] < 130) & (rgb[..., 0] - rgb[..., 2] > 30))
    ys, xs = np.where(gold)
    sel = (ys > 145) & (ys < 260)
    ys, xs = ys[sel], xs[sel]
    if len(ys) < 10:
        print("  objects_L 金色杆芯太少(%d px), 跳过" % len(ys))
        return 1
    A = np.vstack([ys, np.ones(len(ys))]).T
    a, b = np.linalg.lstsq(A, xs, rcond=None)[0]

    c_t = canvas_of(pt, a_t)
    tw_m = c_t[..., 3] > 40
    lum = c_t[..., :3].astype(int).sum(-1)
    wide = np.zeros((H, W), bool)
    seed = np.zeros((H, W), bool)
    for yy in range(Y0, Y1):
        ax = a * yy + b
        for half, tgt in ((HALF, wide), (SEED_HALF, seed)):
            for xx in range(int(ax - half), int(ax + half + 1)):
                if 0 <= xx < W and tw_m[yy, xx] and lum[yy, xx] < LUM_MAX:
                    tgt[yy, xx] = True
    lab, _k = ndi.label(wide, structure=np.ones((3, 3)))
    band = np.zeros((H, W), bool)
    for i in range(1, _k + 1):
        sel = lab == i
        if (sel & seed).any():
            band |= sel
    n = int(band.sum())
    if n == 0:
        print("  没有可搬的暗杆身像素")
        return 0

    # 搬: topwear 清 alpha, objects 放像素(目标已有不透明像素则只清源)
    ys, xs = np.where(band)
    moved = cleared = 0
    for y, x in zip(ys, xs):
        ly, lx = y - po["y"], x - po["x"]
        ty, tx = y - pt["y"], x - pt["x"]
        sv = a_t[y - pt["y"], x - pt["x"]] if (
            0 <= y - pt["y"] < a_t.shape[0]
            and 0 <= x - pt["x"] < a_t.shape[1]) else None
        if sv is None or sv[3] <= 8:
            continue
        in_t = (0 <= ty < a_t.shape[0] and 0 <= tx < a_t.shape[1])
        if in_t and a_t[ty, tx, 3] > 200:
            a_t[ty, tx] = 0            # objects 已有内容: 只清 topwear
            cleared += 1
            continue
        # objects 画布扩容
        need_h = ty + 1 - a_o.shape[0] if ty + 1 > a_o.shape[0] else 0
        need_w = tx + 1 - a_o.shape[1] if tx + 1 > a_o.shape[1] else 0
        top = -ty if ty < 0 else 0
        left = -tx if tx < 0 else 0
        if need_h or need_w or top or left:
            nh = a_o.shape[0] + need_h + top
            nw = a_o.shape[1] + need_w + left
            pad = np.zeros((nh, nw, 4), np.uint8)
            pad[top:top + a_o.shape[0], left:left + a_o.shape[1]] = a_o
            a_o = pad
            ty += top
            tx += left
        a_o[ty, tx] = sv
        a_t[y - pt["y"], x - pt["x"]] = 0
        moved += 1
    print(f"  暗杆身 {n}px: 搬入 objects_L {moved}px, 清重叠 {cleared}px")

    def paste(p, arr):
        ys_, xs_ = np.nonzero(arr[..., 3] > 8)
        if len(ys_) == 0:
            return
        x0_, y0_ = int(xs_.min()), int(ys_.min())
        x1_, y1_ = int(xs_.max()) + 1, int(ys_.max()) + 1
        Image.fromarray(arr[y0_:y1_, x0_:x1_]).save(
            os.path.join(D, "parts", p["image"]))
        p.update({"x": p["x"] + x0_, "y": p["y"] + y0_,
                  "width": x1_ - x0_, "height": y1_ - y0_})

    paste(po, a_o)
    paste(pt, a_t)
    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pc = os.path.join(D, "parts", "coords.json")
    if os.path.isfile(pc):
        json.dump(co, open(pc, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=2)
    print("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
