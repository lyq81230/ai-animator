# -*- coding: utf-8 -*-
"""把某个"孤岛"从一个部件搬到另一个部件(按画布坐标点种子)。

矩形切图总会剩下几块小岛 —— 比如刀柄那截落在躯干里、护腕旁边一截落在腿上。
这些岛看着像"别的部位残留", 但直接删会露底。按种子点找岛、整块搬过去最稳。

用法:
    python move_island.py output/<角色> --to objects_L --at 85,190 --at 237,240
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def main():
    D = sys.argv[1]
    to = sys.argv[sys.argv.index("--to") + 1]
    seeds = []
    for i, a in enumerate(sys.argv):
        if a == "--at":
            x, y = sys.argv[i + 1].split(",")
            seeds.append((int(x), int(y)))
    cp = os.path.join(D, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    W, H = co["canvas"]["w"], co["canvas"]["h"]
    by = {p["name"]: p for p in co["parts"]}

    def canvas_of(p):
        a = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))
        m = np.zeros((H, W, 4), np.uint8)
        m[p["y"]:p["y"] + a.shape[0], p["x"]:p["x"] + a.shape[1]] = a
        return m

    canv = {p["name"]: canvas_of(p) for p in co["parts"]}
    tgt = canv[to]
    for (sx, sy) in seeds:
        for name, c in canv.items():
            if name == to:
                continue
            m = c[..., 3] > 40
            if not m[sy, sx]:
                continue
            lab, _k = ndi.label(m, structure=np.ones((3, 3)))
            i = lab[sy, sx]
            if i == 0:
                continue
            isl = lab == i
            tgt[isl] = c[isl]
            c[..., 3][isl] = 0
            print(f"  ({sx},{sy}) {int(isl.sum())}px: {name} -> {to}")
            break
        else:
            print(f"  ({sx},{sy}) 没找到岛")

    for name, c in canv.items():
        p = by[name]
        ys, xs = np.nonzero(c[..., 3] > 8)
        if len(ys) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        Image.fromarray(c[y0:y1 + 1, x0:x1 + 1]).save(
            os.path.join(D, "parts", p["image"]))
        p.update({"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1})
    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pc = os.path.join(D, "parts", "coords.json")
    if os.path.isfile(pc):
        json.dump(co, open(pc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("完成")


if __name__ == "__main__":
    main()
