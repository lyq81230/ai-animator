# -*- coding: utf-8 -*-
"""董卓第二刀: 腰带上的佩剑要单独拆出来; 腿上那块红布要还给躯干。

两个问题:
  1) 挂在腰间的剑被拆进了 topwear / leg_R_low(跟着右腿走), 剑随腿摆 -> 看着像腰上
     挂的东西在漂。拆图时它是"斜的一条", AI 没给独立层。
  2) leg_L_low / leg_R_low 里各带一条红布(裙摆), 那是躯干上的东西, 分到腿上以后
     两条腿一动就把裙摆撕成两半。

做法:
  · 剑: 用颜色(钢/金)在腰部窗口里取种子连通块 -> 膨胀成整条 -> 得到掩码;
        从所有部件的像素里剪掉, 再单独建一个 `sword` 部件(cls=prop, bind_bone=torso),
        绘制顺序放在腿之后、手之前(手在剑前面)。
  · 红布: 腿部件里"红且暗"的大块(>=80px)剪出来并进 topwear。

用法: python fix_dongzhuo_legs.py output/sprite_boss_dongzhuo_white_336 [--dry]
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ARM_R = ("arm_upper_R", "arm_fore_R", "arm_hand_R")
LEGS = ("leg_L_up", "leg_L_low", "leg_R_up", "leg_R_low")
SWORD_WIN = (148, 300, 224, 272)      # y0,y1,x0,x1
SWORD_SEED = (250, 248)               # (y,x) 落在剑鞘上


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

    # 颜色/alpha 一律取【源图】: 用部件拼出来的"整图"在部件之间有空隙, 判据会漏。
    src_png = None
    for i, a in enumerate(sys.argv):
        if a == "--src":
            src_png = sys.argv[i + 1]
    if not src_png:
        cand = os.path.join("E:/pic", os.path.basename(os.path.normpath(D)) + ".png")
        src_png = cand if os.path.isfile(cand) else None
    if not src_png or not os.path.isfile(src_png):
        print("  找不到源图, 用 --src 指定")
        return
    rgb = np.array(Image.open(src_png).convert("RGBA"))
    if rgb.shape[0] != H or rgb.shape[1] != W:
        tmp = np.zeros((H, W, 4), np.uint8)
        tmp[:min(H, rgb.shape[0]), :min(W, rgb.shape[1])] = rgb[:min(H, rgb.shape[0]), :min(W, rgb.shape[1])]
        rgb = tmp

    # ---------- 1) 剑 ----------
    # 剑斜挂在腰带上, 跨在躯干和右腿之间, AI 没给独立层。
    # 手描多边形怎么调都会带进躯干: 剑柄/护手那块金色本身就是"贴着腰带甲片"的,
    # 多边形一宽就把腰带大金饰包进来(用户看到的"剑里有躯干")。改成:
    #   ① 沿剑轴开一条窄走廊(剑是直的东西, 躯干/腰带都长在走廊外);
    #   ② 在走廊内按"暗描边不许穿过"做连通生长 —— 像素画的物体边界是深色描边,
    #      生长到描边就停, 所以刚好停在剑自己的轮廓上, 不会越界吃进腰带;
    #   ③ 只保留跟剑刃有连接的连通块, 靴子高光那种远处孤岛自动丢掉。
    # 否据仍然保留: 紫布(剑柄后的飘带)、酒红布、拳头(剑从拳头后面过)。
    P0 = np.array([221.0, 146.0])          # 剑首
    P1 = np.array([265.0, 296.0])          # 剑尖方向
    axis = P1 - P0
    LL = float(np.hypot(*axis))
    axis /= LL
    _yy, _xx = np.mgrid[0:H, 0:W]
    _vx, _vy = _xx - P0[0], _yy - P0[1]
    _t = _vx * axis[0] + _vy * axis[1]
    _perp = np.abs(-_vx * axis[1] + _vy * axis[0])
    corridor = (_t >= -3) & (_t <= LL + 3) & (_perp <= 17)

    hand = np.zeros((H, W), bool)
    if "arm_hand_R" in by:
        hand |= canvas("arm_hand_R")[..., 3] > 40

    px = rgb[..., :3].astype(int)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    mx = px.max(axis=2)
    mn = px.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(1, mx), 0.0)
    purple = (b > g + 5) & (r >= g) & (sat > 0.14) & (mx < 235)   # 紫布/飘带
    darkred = (mx < 112) & (sat > 0.30) & (r > g + 8)             # 酒红布
    # 剑首上方那片"褪色酒红"是袍子的暗部(约 129,92,95): 红偏亮但蓝没掉下去。
    # 金色是 b 远小于 g, 所以用 b>g-6 把布和金属分开。
    mauve = (r > g + 10) & (b > g - 6) & (mx < 176) & (sat > 0.12)
    outline = mx < 62                                             # 描边: 不许生长
    solid = rgb[..., 3] > 60
    allowed = solid & (~outline) & (~purple) & (~darkred) & (~mauve) & (~hand) & corridor
    steel = solid & (sat < 0.20) & (mx > 92) & (mx < 238) & (_yy >= 195) & corridor

    lab, n = ndi.label(allowed, np.ones((3, 3)))
    keep = np.unique(lab[steel])
    keep = keep[keep > 0]
    sword = np.isin(lab, keep)

    # 丢掉离剑太远的孤岛(靴子高光): 只留重心在剑轴向附近的块
    lab, n = ndi.label(sword, np.ones((3, 3)))
    _k = np.zeros_like(sword)
    for idx in range(1, n + 1):
        m = lab == idx
        if m.sum() < 45:
            continue
        yy2, xx2 = np.nonzero(m)
        if np.median(yy2) > 284:               # 剑尖到 284 为止, 再往下是靴子
            continue
        _k |= m
    sword = ndi.binary_fill_holes(_k)

    # 描边本身(mx<62)不属于"可生长"像素, 但它是剑自己的外轮廓, 少了会显得剑边发虚。
    # 只把紧贴掩码一圈、且在走廊内的描边补回来。
    ring = ndi.binary_dilation(sword, np.ones((3, 3))) & (~sword)
    sword |= ring & outline & corridor & solid & (~hand)
    sword = ndi.binary_fill_holes(sword)
    if sword.any():
        ys, xs = np.nonzero(sword)
        print("  剑: %d px  x%d-%d y%d-%d" % (sword.sum(), xs.min(), xs.max(), ys.min(), ys.max()))

    # ---------- 2) 腿上的红布 ----------
    leg = np.zeros((H, W), bool)
    for nm in LEGS:
        if nm in by:
            leg |= canvas(nm)[..., 3] > 40
    cloth = leg & (rgb[..., 3] > 40) & (r > 55) & ((r - g) > 20) & ((r - b) > 12)
    lab2, n2 = ndi.label(cloth, np.ones((3, 3)))
    sz = ndi.sum(cloth, lab2, range(1, n2 + 1))
    cloth = np.zeros((H, W), bool)
    for idx, size in enumerate(sz, 1):          # 注意: enumerate 给的是 (序号, 大小)
        if size >= 80:
            cloth |= lab2 == idx
    print("  腿上红布: %d px" % int(cloth.sum()))

    if dry:
        vis = rgb[..., :3].astype(float)
        vis[sword] = vis[sword] * 0.55 + np.array([255, 40, 40]) * 0.45
        edge = sword & ~ndi.binary_erosion(sword, np.ones((3, 3)))
        vis[edge] = [0, 255, 0]
        vis[cloth] = [255, 0, 255]
        box = (205, 138, 285, 305)
        Image.fromarray(np.clip(vis, 0, 255).astype(np.uint8)).crop(box).resize(
            ((box[2] - box[0]) * 9, (box[3] - box[1]) * 9),
            Image.NEAREST).save("_dry_legs.png")
        print("  [dry] 未写入")
        return

    # ---------- 写回 ----------
    def take(part_name, mask):
        """把 mask 上的像素从该部件里剪掉; 返回 (canvas, 有变化?)"""
        c = canvas(part_name)
        m = (c[..., 3] > 40) & mask
        if not m.any():
            return c, False
        c[..., 3][m] = 0
        c[..., :3][m] = 0
        return c, True

    changed = []
    for p in co["parts"]:
        nm = p["name"]
        if nm == "sword":
            continue
        # 剑要从所有部件里剪; 红布只从腿里剪(躯干本来就没有)
        c, ch = take(nm, sword | (cloth if nm in LEGS else np.zeros_like(cloth)))
        if not ch:
            continue
        # 红布那块并回 topwear
        if nm != "topwear" and nm in LEGS:
            pass
        a = c[..., 3]
        if not a.any():
            continue
        ys, xs = np.nonzero(a > 110)
        if not len(ys):
            ys, xs = np.nonzero(a > 8)
        yy0, yy1 = int(ys.min()), int(ys.max()) + 1
        xx0, xx1 = int(xs.min()), int(xs.max()) + 1
        Image.fromarray(c[yy0:yy1, xx0:xx1], "RGBA").save(
            os.path.join(D, "parts", p["image"]))
        p.update({"x": xx0, "y": yy0, "width": xx1 - xx0, "height": yy1 - yy0})
        changed.append(nm)

    # 红布 -> topwear(只补空位)
    tw = by["topwear"]
    host = canvas("topwear")
    # 注意用 <250 而不是 <=40: 原 AI 图层在这一带本来就有一条缝, 之前的 bleed 用
    # 55% 半透明补过(alpha≈140), 腿件那份满不透明像素把缝挡住了; 腿一剪, 半透明
    # 的 bleed 就露出来, 屏幕上是一条发亮的横线。这里用源图整块填实。
    near = ndi.binary_dilation(cloth, np.ones((13, 13)))
    add = (cloth | near) & (host[..., 3] < 250) & (rgb[..., 3] > 40)
    host[add] = rgb[add]
    ys, xs = np.nonzero(host[..., 3] > 110)
    yy0, yy1 = int(ys.min()), int(ys.max()) + 1
    xx0, xx1 = int(xs.min()), int(xs.max()) + 1
    Image.fromarray(host[yy0:yy1, xx0:xx1], "RGBA").save(
        os.path.join(D, "parts", tw["image"]))
    tw.update({"x": xx0, "y": yy0, "width": xx1 - xx0, "height": yy1 - yy0})
    print("  红布并入 topwear %d px" % int(add.sum()))

    # 新建 sword 部件
    if sword.any():
        ys, xs = np.nonzero(sword)
        yy0, yy1 = int(ys.min()), int(ys.max()) + 1
        xx0, xx1 = int(xs.min()), int(xs.max()) + 1
        # 只取掩码内的像素! 之前直接切 bbox, 结果把整块矩形原图(裙摆、甲片、
        # 背景)都做成了"剑"部件 —— 用户看到的"剑里包含躯干"就是这个。
        sub = np.zeros((yy1 - yy0, xx1 - xx0, 4), np.uint8)
        _m = sword[yy0:yy1, xx0:xx1]
        sub[_m] = rgb[yy0:yy1, xx0:xx1][_m]
        Image.fromarray(sub, "RGBA").save(os.path.join(D, "parts", "sword.png"))
        sp = {"name": "sword", "image": "sword.png", "x": xx0, "y": yy0,
              "width": xx1 - xx0, "height": yy1 - yy0,
              "cls": "prop", "bind_bone": "torso"}
        # 画在腿之后、手之前(手在剑前面)
        idx = max(i for i, q in enumerate(co["parts"]) if q["name"] in LEGS) + 1
        co["parts"].insert(idx, sp)
        print("  新建 sword 部件 (%d,%d) %dx%d 插在第 %d 位"
              % (xx0, yy0, xx1 - xx0, yy1 - yy0, idx))

    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pc = os.path.join(D, "parts", "coords.json")
    if os.path.isfile(pc):
        json.dump(co, open(pc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("  改动部件:", ", ".join(changed))


if __name__ == "__main__":
    main()
