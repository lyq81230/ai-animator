# -*- coding: utf-8 -*-
"""给部件加一圈"出血"(bleed): 沿边缘向外扩 N 像素, 颜色取最近的已有着色。

为什么需要: 拆图时相邻部件是硬切的, 只要两块之间有相对位移(弯关节、透视缩短),
切口就会露出背景 —— 屏幕上就是一道白缝/白楔形。多留 2~3px 余量, 让"切口"藏到
邻居下面, 动起来就不会漏底。

只往【原图不透明】的方向扩, 所以不会把部件边缘撑到空气里变胖。

用法: python bleed_parts.py output/sprite_pi_white_336 [像素=3]
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# 五官不扩: 眼睛/嘴一圈扩出去会在脸上糊出色块
SKIP = {"eyes_closed", "eyewhite", "irides", "nose", "mouth", "ears"}


def main():
    D = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    # 新补的像素用半透明: 满不透明会在绑定姿态下把邻居那 1px 深色描边糊掉,
    # 整个角色看起来像水彩(用户: "完全不一样")。半透明既能救缝, 又不吃掉描边。
    A = float(os.environ.get("BLEED_ALPHA", "0.55"))
    cp = os.path.join(D, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    W, H = co["canvas"]["w"], co["canvas"]["h"]
    char = os.path.basename(os.path.normpath(D))
    src_p = os.path.join("E:/pic", char + ".png")
    src = None
    if os.path.isfile(src_p):
        src = np.array(Image.open(src_p).convert("RGBA"))
        if src.shape[1] != W or src.shape[0] != H:
            src = np.array(Image.fromarray(src).resize((W, H), Image.LANCZOS))
    ok_src = (src[..., 3] > 40) if src is not None else np.ones((H, W), bool)

    disc = np.zeros((2 * N + 1, 2 * N + 1), bool)
    yy, xx = np.mgrid[-N:N + 1, -N:N + 1]
    disc[yy * yy + xx * xx <= N * N] = True

    # 出血必须"藏在别人下面"才有意义。只往【绘制顺序在自己之后】的部件身上长:
    # 否则给一个 z 很高的部件(比如右肩)扩边, 扩出来的那圈会盖在躯干铠甲的上面,
    # 看着就像肩甲肿了一圈(用户: "右肩膀补齐太多, 遮盖了躯干铠甲")。
    order = sorted(range(len(co["parts"])),
                   key=lambda i: (float(co["parts"][i].get("z", 0) or 0), i))
    rank = {idx: k for k, idx in enumerate(order)}
    later = [None] * len(co["parts"])          # 每个部件的"后面还有谁"掩膜
    run_union = np.zeros((H, W), bool)
    for idx in reversed(order):
        later[idx] = run_union.copy()
        p = co["parts"][idx]
        a = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))
        h, w = a.shape[:2]
        m = np.zeros((H, W), bool)
        m[p["y"]:p["y"] + h, p["x"]:p["x"] + w] = a[..., 3] > 24
        run_union |= m

    total = 0
    for idx, p in enumerate(co["parts"]):
        if p["name"] in SKIP:
            continue
        f = os.path.join(D, "parts", p["image"])
        a = np.array(Image.open(f).convert("RGBA"))
        h, w = a.shape[:2]
        canvas = np.zeros((H, W, 4), np.uint8)
        canvas[p["y"]:p["y"] + h, p["x"]:p["x"] + w] = a
        m = canvas[..., 3] > 24
        cover = later[idx]
        grow = ndi.binary_dilation(m, disc) & ~m & ok_src & cover
        if not grow.any():
            continue
        # 用最近的已有像素着色(向外"拉"颜色)
        _d, (iy, ix) = ndi.distance_transform_edt(~m, return_indices=True)
        fill = canvas[iy[grow], ix[grow]]
        fill[:, 3] = (fill[:, 3].astype(np.float32) * A).astype(np.uint8)
        canvas[grow] = fill
        # 重裁 bbox
        mm = canvas[..., 3] > 8
        ys, xs = np.nonzero(mm)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        Image.fromarray(canvas[y0:y1 + 1, x0:x1 + 1]).save(f)
        add = int(grow.sum())
        total += add
        p.update({"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1})
        print(f"  {p['name']:<14} +{add:5d}px  bbox ({x0},{y0}) {x1-x0+1}x{y1-y0+1}")
    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pc = os.path.join(D, "parts", "coords.json")
    if os.path.isfile(pc):
        json.dump(co, open(pc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"共补 {total}px ({N}px 出血)")


if __name__ == "__main__":
    main()
