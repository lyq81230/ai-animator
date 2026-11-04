#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""measure_motion.py — 量一份导出的 Spine JSON 里各骨骼的实际行程(像素)。

为什么要它: 预览缩略图太小, 4px 和 12px 的摆幅看着差不多, 靠肉眼调参数就是瞎调。
这里直接按骨骼时间线做一遍正向运动学, 打印骨原点的 x/y 行程, 用数字对比。

用法:
    python measure_motion.py output/<名字>/spine/<名字>.json walk

判定基准(正面角色行走):
    肩→手 缩短 8~15%；手/脚 横向 0.1~0.3、竖向 1.0~1.6
    (2026-11 起 肩/肘/腕 的支点按真实接缝放, 臂长翻倍 -> 缩短% 的分母是全长)
再小就是"手僵住了"(实测有过 4px vs 23px 的 5 倍差), 再大就会甩得像跑步。
"""
import bisect
import json
import math
import sys

import numpy as np

DEFAULT_BONES = ("arm_L_up", "arm_L_fore", "arm_L_hand",
                 "leg_L_foot", "leg_R_foot")


def _order(bones):
    """父骨在前, 子骨在后(时间线要按这个顺序累乘)。"""
    out, seen = [], set()

    def visit(name):
        if name in seen:
            return
        par = bones[name].get("parent")
        if par and par in bones:
            visit(par)
        seen.add(name)
        out.append(name)

    for b in bones:
        visit(b)
    return out


def _val(fr, key):
    if key == "rotate":
        return float(fr.get("angle", 0.0))
    if key == "translate":
        return (float(fr.get("x", 0.0)), float(fr.get("y", 0.0)))
    return (float(fr.get("x", 1.0)), float(fr.get("y", 1.0)))


def _at(tl, t, key):
    ts = [f["time"] for f in tl]
    if t <= ts[0]:
        return _val(tl[0], key)
    if t >= ts[-1]:
        return _val(tl[-1], key)
    i = bisect.bisect_right(ts, t) - 1
    a, b = tl[i], tl[i + 1]
    u = (t - a["time"]) / max(1e-9, b["time"] - a["time"])
    va, vb = _val(a, key), _val(b, key)
    if key == "rotate":
        return va + (vb - va) * u
    return (va[0] + (vb[0] - va[0]) * u, va[1] + (vb[1] - va[1]) * u)


def world_at(bones, order, anim, t):
    """返回 {骨名: 3x3 世界矩阵}。Spine 的局部变换 = T(x,y)·R·S。"""
    out = {}
    for name in order:
        bb = bones[name]
        tl = anim.get(name, {})
        x, y = bb.get("x", 0.0), bb.get("y", 0.0)
        rot = bb.get("rotation", 0.0)
        sx, sy = bb.get("scaleX", 1.0), bb.get("scaleY", 1.0)
        if "translate" in tl:
            tx, ty = _at(tl["translate"], t, "translate")
            x += tx
            y += ty
        if "rotate" in tl:
            rot += _at(tl["rotate"], t, "rotate")
        if "scale" in tl:
            ax, ay = _at(tl["scale"], t, "scale")
            sx *= ax
            sy *= ay
        c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        L = np.array([[c * sx, -s * sy, x],
                      [s * sx, c * sy, y],
                      [0.0, 0.0, 1.0]])
        par = bb.get("parent")
        out[name] = (out[par] @ L) if par in out else L
    return out


def measure(path, action="walk", names=DEFAULT_BONES):
    data = json.load(open(path, encoding="utf-8"))
    bones = {b["name"]: b for b in data["bones"]}
    anim = data["animations"][action]["bones"]
    order = _order(bones)
    times = sorted({f["time"] for tls in anim.values()
                    for tl in tls.values() for f in tl})
    res = {}
    for nm in names:
        if nm not in bones:
            continue
        pts = []
        for t in times:
            M = world_at(bones, order, anim, t)[nm]
            pts.append((M[0, 2], M[1, 2]))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        res[nm] = (max(xs) - min(xs), max(ys) - min(ys))
    # 肩→手 的实际长度变化 = 观众看到的"手臂往镜头前伸、变短"的幅度
    if "arm_L_up" in bones and "arm_L_hand" in bones:
        lens = []
        for t in times:
            W = world_at(bones, order, anim, t)
            sh, hd = W["arm_L_up"], W["arm_L_hand"]
            lens.append(math.hypot(hd[0, 2] - sh[0, 2], hd[1, 2] - sh[1, 2]))
        res["_arm_len"] = (min(lens), max(lens),
                           (max(lens) - min(lens)) / max(1e-6, max(lens)) * 100.0)
    # 手相对【躯干】的横向偏移: 取"手最高(前摆到头)"和"手最低(后摆到头)"两帧比。
    # 必须看相对值 —— 整个身体在左右移重心(sway), 绝对坐标会被它盖掉。
    # 正数 = 前摆时手往身体中线走。
    if "torso" in bones and "arm_L_hand" in bones:
        rows = []
        for t in times:
            W = world_at(bones, order, anim, t)
            hd, tp = W["arm_L_hand"], W["torso"]
            rows.append((hd[0, 2] - tp[0, 2], hd[1, 2]))
        hi = max(rows, key=lambda r: r[1])
        lo = min(rows, key=lambda r: r[1])
        res["_inward"] = hi[0] - lo[0]
    return res


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: python measure_motion.py <spine.json> [walk]")
    path = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else "walk"
    res = measure(path, action)
    for nm, val in res.items():
        if nm.startswith("_"):
            continue
        dx, dy = val
        print(f"{nm:12s} x Δ{dx:6.2f}  y Δ{dy:6.2f}")
    arm = res.get("_arm_len")
    hand = res.get("arm_L_hand")
    foot = res.get("leg_L_foot")
    if arm:
        lo, hi, pct = arm
        print(f"\n肩→手 长度 {lo:.1f}..{hi:.1f}   缩短 {pct:.1f}%"
              f"   (按肩→手全长算: 8~15% 能看出\"往前伸\"；<6% 只剩左右摆)")
    inward = res.get("_inward")
    if inward is not None:
        print(f"前摆时手向身体中线 {inward:+.2f} px"
              f"   (相对躯干量; 4~8px = \"少许内收\", 负值就是往外撇)")
    if hand and foot and foot[0] > 0.1:
        print(f"手/脚 横向行程比 = {hand[0] / foot[0]:.2f}"
              f"   (绝对坐标, 含身体左右移重心; 0.1~0.3 正常, >0.45 就是\"左右甩\")")
    if hand and foot and foot[1] > 0.1:
        print(f"手/脚 竖向行程比 = {hand[1] / foot[1]:.2f}   (抬手的幅度对抬脚)")


if __name__ == "__main__":
    main()
