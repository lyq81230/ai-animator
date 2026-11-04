# -*- coding: utf-8 -*-
"""把混进"道具/武器"部件里的衣服像素还回去。

现象: objects_L(大斧) 里除了斧头, 还连着一块 307px 的深色布片
      (画布 126,203 - 135,262), 那是裙摆的左边缘, 被 AI 的 objects 层
      一起圈进来了。武器跟手走, 这块布就会跟着斧头飞。

做法:
    1. 按连通块拆 objects_L, 只留最大的一块(斧头本体)
    2. 被剔掉的像素按"离哪个部件最近"归给那个部件
       (候选排除 objects_L 自己, 免得又还给自己)
    3. 目标部件包围盒不够就扩, 像素颜色取源图
    4. 自检: 空洞必须为 0

用法: python fix_lvbu_weapon.py [部件名]
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

D = "output/sprite_boss_lvbu_white_336"
SRC = "E:/pic/sprite_boss_lvbu_white_336.png"
CANVAS = 336


def canvas_of(D, p):
    a = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))
    m = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    m[p["y"]:p["y"] + a.shape[0], p["x"]:p["x"] + a.shape[1]] = a
    return m


def write_part(D, p, add_mask, src, keep_mask=None):
    """add_mask: 新加入的像素(取源图色); keep_mask: 只保留原内容里这部分。"""
    old = canvas_of(D, p)
    if keep_mask is not None:
        old[..., 3][~keep_mask] = 0
    content = (old[..., 3] > 8) | add_mask
    ys, xs = np.nonzero(content)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    buf = old[y0:y1 + 1, x0:x1 + 1].copy()
    sub = add_mask[y0:y1 + 1, x0:x1 + 1]
    buf[sub] = src[y0:y1 + 1, x0:x1 + 1][sub]
    Image.fromarray(buf).save(os.path.join(D, "parts", p["image"]))
    p.update({"x": x0, "y": y0,
              "width": int(x1 - x0 + 1), "height": int(y1 - y0 + 1)})


def main():
    part_name = sys.argv[1] if len(sys.argv) > 1 else "objects_L"
    c = json.load(open(os.path.join(D, "coords.json"), encoding="utf-8"))
    parts = {p["name"]: p for p in c["parts"]}
    src = np.array(Image.open(SRC).convert("RGBA"))

    p = parts[part_name]
    can = canvas_of(D, p)
    al = can[..., 3] > 8
    lab, n = ndi.label(al, np.ones((3, 3)))
    sizes = ndi.sum(al, lab, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    junk = al & (lab != keep)
    print(f"{part_name}: {n} 个连通块; 保留 {int(sizes[keep-1])}px, "
          f"剔出 {int(junk.sum())}px")
    ys, xs = np.nonzero(junk)
    if len(xs):
        print(f"  剔出区域 ({xs.min()},{ys.min()})-({xs.max()},{ys.max()})")

    # 1) 只留本体
    newp = al & (lab == keep)
    write_part(D, p, np.zeros_like(al), src, keep_mask=newp)
    print(f"  {part_name} 缩到 ({p['x']},{p['y']}) {p['width']}x{p['height']}")

    # 2) 剔出的像素按最近部件归还
    cover = np.zeros((CANVAS, CANVAS), np.int32)
    order = [q for q in c["parts"] if q["name"] != part_name]
    for i, q in enumerate(order, 1):
        cover[canvas_of(D, q)[..., 3] > 8] = i
    _d, (iy, ix) = ndi.distance_transform_edt(cover == 0, return_indices=True)
    jy, jx = np.nonzero(junk)
    owner = cover[iy[jy, jx], ix[jy, jx]]
    for i, q in enumerate(order, 1):
        selm = np.zeros((CANVAS, CANVAS), bool)
        m = owner == i
        if not m.any():
            continue
        selm[jy[m], jx[m]] = True
        write_part(D, q, selm, src)
        print(f"  {int(m.sum()):4d}px -> {q['name']}")

    # 3) 自检
    cov = np.zeros((CANVAS, CANVAS), np.int32)
    for q in c["parts"]:
        cov += (canvas_of(D, q)[..., 3] > 8)
    hole = (src[..., 3] > 8) & (cov == 0)
    print(f"覆盖: 空缺 {int(hole.sum())}px")
    json.dump(c, open(os.path.join(D, "coords.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
