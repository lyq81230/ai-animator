# -*- coding: utf-8 -*-
"""从镜像的另一把武器上「整块复制」尖端, 补到被原画截断的那一把上。

原画的成对武器通常是镜像的 (同一把刀的左右翻转)。当其中一把的尖被画布/层级截掉,
直接复制另一把的尖并镜像过来, 颜色/亮边/暗部/收尖斜率都与原作完全一致,
比"按横截面颜色外推"自然得多。

用法:
    python clone_tip.py <png> --src 30,272,58,296 --mirror 339 --dx 1 --dy 0 \
                             --clear 280,273,308,300

    --src   模板区域 x0,y0,x1,y1 (含端点), 取镜像那一把的尖端
    --mirror 镜像常数 C: x' = C - x  (C = 两把武器对称轴 x 的 2 倍, 可为奇数)
    --dx    镜像后再整体平移 (向右为正)
    --dy    纵向平移
    --clear 先清空的矩形 x0,y0,x1,y1 (要替换掉的旧尖/残段)
    --dry   只出预览, 不覆盖原图
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png")
    ap.add_argument("--src", required=True, help="模板区域 x0,y0,x1,y1 含端点")
    ap.add_argument("--mirror", type=int, required=True, help="镜像常数 C: x' = C - x")
    ap.add_argument("--dx", type=int, default=0)
    ap.add_argument("--dy", type=int, default=0)
    ap.add_argument("--clear", default=None, help="清空矩形 x0,y0,x1,y1 含端点")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    sx0, sy0, sx1, sy1 = [int(v) for v in a.src.split(",")]
    src = Image.open(a.png).convert("RGBA")
    arr = np.array(src)
    W, H = src.size

    canvas = arr.copy()
    if a.clear:
        cx0, cy0, cx1, cy1 = [int(v) for v in a.clear.split(",")]
        canvas[cy0:cy1 + 1, cx0:cx1 + 1] = 0

    C = a.mirror
    n = 0
    for y in range(sy0, sy1 + 1):
        ty = y + a.dy
        if not (0 <= ty < H):
            continue
        for x in range(sx0, sx1 + 1):
            tx = C - x + a.dx
            if not (0 <= tx < W):
                continue
            px = arr[y, x]
            if px[3] == 0:
                continue
            canvas[ty, tx] = px
            n += 1

    out = Image.fromarray(canvas, "RGBA")
    dst = a.out or (a.png.replace(".png", "_tipfix_preview.png") if a.dry
                    else a.png)
    if not a.dry and not a.out:
        bak = a.png[:-4] + ".bak_clonetip.png"
        if not os.path.exists(bak):
            src.save(bak, "PNG")
            print("备份 ->", bak)
    out.save(dst)

    ys, xs = np.nonzero(canvas[..., 3])
    print("复制 %d 像素 -> %s" % (n, dst))
    if a.clear:
        x0, y0, x1, y1 = [int(v) for v in a.clear.split(",")]
    else:
        x0, y0, x1, y1 = 0, 0, W - 1, H - 1
    m = np.zeros_like(canvas[..., 3], bool)
    m[y0:y1 + 1, x0:x1 + 1] = True
    m &= canvas[..., 3] > 40
    if m.any():
        yy, xx = np.nonzero(m)
        print("该区域现在的非透明范围 x%d-%d y%d-%d" %
              (xx.min(), xx.max(), yy.min(), yy.max()))


if __name__ == "__main__":
    sys.exit(main())
