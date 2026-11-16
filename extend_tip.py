# -*- coding: utf-8 -*-
"""把被截断的刀尖按【刀身自己的两条边】延长出来。

之前是用"镜像复制另一把武器"的办法补尖(左侧那把是完整的), 但两把刀不是严格的
镜像 —— 横截面的宽度序列差 1~2px, 直接贴过去在接缝处会留下一个台阶(左边突然缩
进去 2px), 看着像"刀尖和刀身不在一条线上"。

这里改成纯几何延长:
  1. 取接缝上方若干行, 分别拟合【左边缘 x(y)】和【右边缘 x(y)】的直线;
  2. 从接缝往下按这两条直线继续推 20 来行, 宽度自然收窄到 1~2px;
  3. 每行的颜色从"接缝上一行"的横截面按比例重采样 —— 亮边、暗面、紫色条纹
     的相对位置跟着宽度一起收, 所以颜色是连续的。
用法: python extend_tip.py <png> --x0 260 --x1 320 --ycut 273 [--len 24] [--dry]
"""
import argparse
import sys

import numpy as np
from PIL import Image


def row_span(mask, y, x0, x1):
    xs = np.nonzero(mask[y, x0:x1])[0]
    if not len(xs):
        return None
    return x0 + xs.min(), x0 + xs.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png")
    ap.add_argument("--x0", type=int, default=260)
    ap.add_argument("--x1", type=int, default=320)
    ap.add_argument("--ycut", type=int, required=True, help="最后一行完好像素")
    ap.add_argument("--len", type=int, default=24, help="往下延长多少行")
    ap.add_argument("--fit", type=int, default=8, help="用上方多少行拟合边缘")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    img = Image.open(a.png).convert("RGBA")
    arr = np.array(img)
    mask = arr[..., 3] > 40

    lefts, rights, ys = [], [], []
    for y in range(a.ycut - a.fit + 1, a.ycut + 1):
        sp = row_span(mask, y, a.x0, a.x1)
        if sp:
            lefts.append(sp[0]); rights.append(sp[1]); ys.append(y)
    if len(ys) < 3:
        print("拟合样本不足")
        return 1
    ys = np.array(ys, float)
    kl, bl = np.polyfit(ys, np.array(lefts, float), 1)
    kr, br = np.polyfit(ys, np.array(rights, float), 1)
    print("左边缘 dx/dy=%.2f, 右边缘 dx/dy=%.2f (宽度每行收 %.2f px)"
          % (kl, kr, kl - kr))

    # 参考横截面: 最后一行完好的像素
    ref_y = a.ycut
    xl0, xr0 = row_span(mask, ref_y, a.x0, a.x1)
    ref = arr[ref_y, xl0:xr0 + 1].astype(np.float32)
    nref = len(ref)

    out = arr.copy()
    n_fill = 0
    last = None
    for i in range(1, a.len + 1):
        y = a.ycut + i
        if y >= arr.shape[0]:
            break
        xl = kl * y + bl
        xr = kr * y + br
        w = int(round(xr - xl)) + 1
        if w <= 0:
            break
        if w == 1:
            # 收成一点: 取参考横截面的中心色
            px = ref[nref // 2]
            out[y, int(round(xl))] = px.astype(np.uint8)
            n_fill += 1
            last = y
            continue
        # 把参考横截面按比例重采样到新宽度(最近邻, 保留亮边/暗面结构)
        idx = np.clip((np.arange(w) * (nref - 1) / max(1, w - 1)).round().astype(int),
                      0, nref - 1)
        seg = ref[idx]
        # 最后两行做轻微淡出, 免得尖端是个硬方块
        if i >= a.len - 1:
            seg[:, 3] *= 0.7
        xs = int(round(xl))
        out[y, xs:xs + w] = seg.astype(np.uint8)
        n_fill += 1
        last = y

    dst = a.png.replace(".png", "_tipext_preview.png") if a.dry else a.png
    Image.fromarray(out, "RGBA").save(dst)
    print("延长 %d 行(到 y=%s), 写入 %d 像素 -> %s" % (n_fill, last, n_fill, dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
