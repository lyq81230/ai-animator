# -*- coding: utf-8 -*-
"""清掉部件里"颜色对不上"的小碎块。

为什么需要
----------
拆图链尾(repair_scarf / merge_smoke / bleed / fill_gaps)会往部件里补像素,
偶尔把邻居的一小角(纯黑描边残片、棕袍角)粘进头巾层这类"覆盖层"里。它们的
alpha 是实心的, 在编辑器里就是头巾上凭空多出几笔; 动画里还会跟着头巾一起飘。

做法
----
只处理白名单里的部件(默认 headwear_overlay、head): 把每个连通块和"最大块
的平均色"比一下, 面积小于 --max 且色差大于 --dist 的, 整块删掉。
最大块不动, 所以部件本体一个像素都不会少。

用法
----
    python clean_frags.py output/fashu_hero_338
    python clean_frags.py output/fashu_hero_338 --parts headwear_overlay --max 200
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

DEF_PARTS = ("headwear_overlay", "head")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--parts", default=",".join(DEF_PARTS))
    ap.add_argument("--max", type=int, default=150, help="只清小于这么多像素的块")
    ap.add_argument("--dist", type=float, default=60.0, help="与主块平均色的 RGB 距离阈值")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    D = os.path.abspath(a.work)
    cp = os.path.join(D, "parts", "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    pmap = {p["name"]: p for p in co["parts"]}
    targets = [s.strip() for s in a.parts.split(",") if s.strip()]

    total = 0
    for nm in targets:
        p = pmap.get(nm)
        if p is None:
            print(f"  {nm}: 没有这个部件, 跳过")
            continue
        fp = os.path.join(D, "parts", p["image"])
        im = np.array(Image.open(fp).convert("RGBA"))
        al = im[:, :, 3]
        m = (al > 20).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        if n <= 2:
            continue
        areas = [(int(st[k, 4]), k) for k in range(1, n)]
        areas.sort(reverse=True)
        big = lab == areas[0][1]
        ref = im[:, :, :3][big].mean(0)
        drop = np.zeros_like(m, bool)
        for ar, k in areas[1:]:
            if ar >= a.max:
                continue
            sel = lab == k
            d = float(np.linalg.norm(im[:, :, :3][sel].mean(0) - ref))
            x, y, w, h = (int(v) for v in st[k, :4])
            if d > a.dist:
                drop |= sel
                print(f"  {nm}: 删 {ar}px @({x},{y}) {w}x{h}  色差 {d:.0f}")
        if drop.any():
            total += int(drop.sum())
            if not a.dry:
                im[drop] = 0
                Image.fromarray(im, "RGBA").save(fp)
    if total and not a.dry:
        json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        wc = os.path.join(D, "coords.json")
        if os.path.isfile(wc):
            json.dump(co, open(wc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"清除碎块 {total}px" + (" (dry)" if a.dry else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
