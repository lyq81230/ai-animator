"""给原画里被截断的武器补出枪尖。

背景:
    魔神右手武器在 y=273 处宽 11px 直接断掉(左手那把正常收到 1px 尖), 原画本身就缺。
    这里按刀身最后几行的走向与配色, 沿轴向外推一段逐渐收窄的尖, 颜色取自刀身
    同一横截面的相对位置, 所以亮边/暗面/紫光都能跟着延续。

用法:
    python fix_weapon_tip.py <png> --row 273 --x0 285 --x1 295 [--len 11] [--slope 0.9] [--dry]
"""
import argparse
import os
import shutil

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png")
    ap.add_argument("--row", type=int, required=True, help="最后一个完整行的 y")
    ap.add_argument("--x0", type=int, required=True)
    ap.add_argument("--x1", type=int, required=True)
    ap.add_argument("--len", type=int, default=11, help="向外延伸多少行")
    ap.add_argument("--slope", type=float, default=0.9, help="每行中心右移多少像素")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    im = Image.open(a.png).convert("RGBA")
    A = np.array(im)
    src = A[a.row, a.x0:a.x1 + 1].copy()      # 最后一行横截面(颜色 + alpha)
    w0 = a.x1 - a.x0 + 1
    cx = (a.x0 + a.x1) / 2.0

    added = 0
    for k in range(1, a.len + 1):
        y = a.row + k
        t = 1.0 - k / float(a.len + 1)         # 1 -> 趋于 0
        w = max(1, int(round(w0 * t)))
        c = cx + a.slope * k
        x0 = int(round(c - (w - 1) / 2.0))
        for i in range(w):
            x = x0 + i
            if x < 0 or x >= A.shape[1] or y >= A.shape[0]:
                continue
            u = 0.0 if w == 1 else i / float(w - 1)
            si = int(round(u * (w0 - 1)))
            px = src[si].copy()
            if px[3] > 8:
                A[y, x] = px
                added += 1
    print(f"补出 {added} 个像素, 尖端到 y={a.row + a.len}")
    if a.dry:
        Image.fromarray(A).save(a.png.replace(".png", "_tipfix_preview.png"))
        print("预览 ->", a.png.replace(".png", "_tipfix_preview.png"))
        return
    bak = a.png + ".bak_tipfix"
    if not os.path.isfile(bak):
        shutil.copyfile(a.png, bak)
        print("备份 ->", bak)
    Image.fromarray(A).save(a.png)
    print("已写入", a.png)


if __name__ == "__main__":
    main()
