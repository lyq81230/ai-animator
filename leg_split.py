#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leg_split.py — 把 see-through 的 'legwear' 层拆成左右两条腿。

大模型把两条腿(连同鞋)输出在同一层, 下游没法给腿挂骨、也没法迈步。
这里按连通域把它们分开: 每条腿一个部件, 左右按 x 中心区分 (L/R)。

之所以只拆左右、不再拆大腿/小腿: 一条腿是一整块网格, 挂 髋-膝-踝 三节骨、
沿高度做权重渐变就能在膝盖处平滑弯曲, 不会像切片那样有接缝。

输出: list[dict(side, rgba(full-canvas H,W,4))]。分不出两条(腿并在一起/
只有一条) 时返回原来的单片, 由调用方决定怎么处理。
"""
import cv2
import numpy as np

MIN_LEG_AREA = 400          # 单条腿最小面积(像素)


def split_legs(rgba, min_area=MIN_LEG_AREA):
    """返回 [{'side': 'L'/'R', 'rgba': ...}] ; 分不开则返回 []。"""
    H, W = rgba.shape[:2]
    a = (rgba[..., 3] > 24).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(a, 8)

    comps = [(i, float(cents[i][0])) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if len(comps) < 2:
        return []
    # 取最大的两块(偶尔会有小碎块挂在旁边)
    comps.sort(key=lambda t: -int(stats[t[0], cv2.CC_STAT_AREA]))
    comps = sorted(comps[:2], key=lambda t: t[1])      # 再按 x 排: 左 -> 右

    out = []
    for lab, _cx in comps:
        mask = (labels == lab)
        rgb = rgba.copy()
        rgb[..., 3] = np.where(mask, rgba[..., 3], 0)
        out.append({"rgba": rgb})
    out[0]["side"] = "L"
    out[1]["side"] = "R"
    return out
