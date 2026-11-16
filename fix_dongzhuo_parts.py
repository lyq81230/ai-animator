# -*- coding: utf-8 -*-
"""董卓: objects_L / objects_R 是 AI 分层给的"手 + 披风 + 裙甲"大杂烩, 要拆干净。

现状:
  * objects_L/R 里装着一只拳头 + 一大块披风/裙甲(它们跟着手骨的 prop 骨走);
  * 反过来 topwear 里又画着整只护腕的上半截(手只拆出去一半)。
结果: 手臂一动, 一小块披风跟着手飘; 而护腕上半截留在躯干上, 手像被切开。

做法:
  1) objects_* 的像素并进 topwear(只贴 topwear 透明处) —— 披风回到身体, 像素不丢;
  2) 删掉 objects_* 部件(里面的拳头和 arm_fore/arm_hand 完全重复);
  3) topwear 上被"手"盖住的那块:
       · 贴着外形轮廓的(手臂伸到身体外面那圈) -> alpha 清零, 本来就不该有东西;
       · 躯干内部的 -> 用 TELEA 补色(alpha 保留), 免得手臂移开时露出破洞。

用法: python fix_dongzhuo_parts.py output/sprite_boss_dongzhuo_white_336 [--dry]
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HOST = "topwear"
HAND = ("arm_fore_L", "arm_fore_R", "arm_hand_L", "arm_hand_R")


def main():
    D = sys.argv[1].rstrip("/\\")
    dry = "--dry" in sys.argv
    cp = os.path.join(D, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    H, W = co["canvas"]["h"], co["canvas"]["w"]
    by = {p["name"]: p for p in co["parts"]}

    def canvas(nm):
        p = by[nm]
        a = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))
        m = np.zeros((H, W, 4), np.uint8)
        m[p["y"]:p["y"] + a.shape[0], p["x"]:p["x"] + a.shape[1]] = a
        return m

    host = canvas(HOST)
    objs = [n for n in by if n.startswith("objects")]
    if not objs:
        print("  没有 objects 部件, 跳过")
        return

    # 1) objects -> topwear(只往空的 alpha 处贴, 不覆盖已有像素)
    moved = 0
    for n in objs:
        o = canvas(n)
        m = (o[..., 3] > 40) & (host[..., 3] <= 40)
        host[m] = o[m]
        moved += int(m.sum())
        print("  %-10s %d px 并入 %s" % (n, int(m.sum()), HOST))

    # 2) 抹掉 topwear 上的手
    hand = np.zeros((H, W), bool)
    for n in HAND:
        if n in by:
            hand |= canvas(n)[..., 3] > 40
    op = host[..., 3] > 40
    hit = op & hand
    edge = ndi.binary_dilation(~op, np.ones((7, 7))) & hit        # 轮廓外圈 3px
    inner = hit & ~edge
    if inner.any():
        # TELEA 补色会把旁边的金边糊成浅褐; 这片在手后面, 真实应该是披风的暗部。
        # 所以: 边缘按 inpaint 走(不出硬边), 越往里越压到周围布色(中位色)。
        ring = ndi.binary_dilation(hit, np.ones((13, 13))) & op & ~hit
        med = np.median(host[..., :3][ring], axis=0) if ring.any() else np.zeros(3)
        res = cv2.inpaint(host[..., :3].copy(), (inner.astype(np.uint8)) * 255, 6,
                          cv2.INPAINT_TELEA).astype(np.float64)
        dst = ndi.distance_transform_edt(inner)
        wgt = np.clip(dst / 7.0, 0, 0.92)[..., None]
        host[..., :3][inner] = (res * (1 - wgt) + med * wgt)[inner].astype(np.uint8)
    host[..., 3][edge] = 0
    host[..., :3][edge] = 0
    print("  %s 里: 轮廓切掉 %d px, 内部补色 %d px" % (HOST, int(edge.sum()), int(inner.sum())))

    # 裁边阈值要和 gen_idle_bones._alpha_bbox(110) 一致: 并进来的披风边缘有
    # alpha 41~110 的软像素, 只按 >8 裁会让 PNG 比"内容框"大 1~4px, uv_box 与
    # 图集区域对不上, uv 偏差变成 0.75px(其它部件都是 0)。
    _th = 110 if (host[..., 3] > 110).any() else 8
    if dry:
        ys, xs = np.nonzero(host[..., 3] > _th)
        Image.fromarray(host[ys.min():ys.max() + 1, xs.min():xs.max() + 1], "RGBA").save(
            "_dry_%s.png" % HOST)
        print("  [dry] 未写入")
        return

    ys, xs = np.nonzero(host[..., 3] > _th)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    Image.fromarray(host[y0:y1, x0:x1], "RGBA").save(
        os.path.join(D, "parts", by[HOST]["image"]))
    by[HOST].update({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0})

    # 3) 删掉 objects
    for n in objs:
        f = os.path.join(D, "parts", by[n]["image"])
        if os.path.isfile(f):
            os.remove(f)
        co["parts"].remove(by[n])
        print("  删除部件 %s" % n)
    for p in co["parts"]:
        for k in ("follow", "follow_part"):
            if p.get(k) in objs:
                p[k] = None
    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pc = os.path.join(D, "parts", "coords.json")
    if os.path.isfile(pc):
        json.dump(co, open(pc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("  完成: 并入 %d px, %s %dx%d, 余 %d 个部件"
          % (moved, HOST, x1 - x0, y1 - y0, len(co["parts"])))


if __name__ == "__main__":
    main()
