#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arm_split.py — see-through 'handwear' 层手臂精细拆分。

see-through 大模型把整条手臂(大臂+小臂+手/手套)都输出在单个 handwear 层,
导致下游无法对大小臂做独立 FK。此模块把 handwear 切成:
  每条手臂(连通域) -> 沿主轴(PCA)切 3 段: arm_upper(大臂) / arm_fore(小臂) /
  arm_hand(手)。左右臂按 x 中心区分 (L/R)。

切分依据(无监督几何): 主轴投影位置 + depth 辅助。肩在主轴近躯干端, 手在远端。
切点用比例(肩->手): 0~0.42 大臂, 0.42~0.74 小臂, 0.74~1 手。

输出: 多个 RGBA crop (full-canvas, 与输入同尺寸, 便于下游 bbox 还原) +
每片的 cls。被 build_from_seethrough 调用。
"""
import numpy as np
import cv2


# 主轴比例切点 (肩端=0 -> 手端=1)
SEG_CUTS = (0.42, 0.74)            # 大臂|小臂, 小臂|手
SEG_CLS = ("arm_upper", "arm_fore", "arm_hand")
MIN_COMP_AREA = 800                # 手臂连通域最小面积
FEATHER = 3                        # 切缝羽化(px) 避免硬边缝隙


def _principal_axis(pts):
    """PCA 主轴: 返回 (mean(2,), axis(2,) 单位向量, 投影值数组)。"""
    mean = pts.mean(0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    proj = (pts - mean) @ axis
    # 让主轴正方向指向 y 增大(向下=朝手), 保证 0=肩 1=手 的稳定性
    if axis[1] < 0:
        axis = -axis
        proj = -proj
    return mean, axis, proj


def _segment_mask(label_mask, proj, lo, hi, shape):
    """按主轴投影区间 [lo,hi] 取该段像素掩码 (在 label_mask 内)。"""
    ys, xs = np.where(label_mask)
    sel = (proj >= lo) & (proj < hi)
    seg = np.zeros(shape, np.uint8)
    seg[ys[sel], xs[sel]] = 255
    if FEATHER > 0:
        seg = cv2.dilate(seg, np.ones((FEATHER, FEATHER), np.uint8))
        seg &= (label_mask * 255).astype(np.uint8)
    return seg


def split_handwear(rgba, depth=None):
    """
    rgba: (H,W,4) handwear 层。depth: (H,W) 可选, 暂未强用。
    返回: list[dict(cls, side, rgba(full-canvas H,W,4))], 已按 肩->手 顺序。
    若连通域不像手臂(太短/太宽)则原样返回单片 arm_hand(回退)。
    """
    H, W = rgba.shape[:2]
    a = (rgba[..., 3] > 24).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(a, 8)

    arms = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_COMP_AREA:
            continue
        arms.append((i, cents[i][0]))        # (label, cx)
    if not arms:
        return [{"cls": "arm_hand", "side": "", "rgba": rgba}]

    # 左右: cx 小=左(L), 大=右(R)
    arms.sort(key=lambda t: t[1])
    sides = {}
    if len(arms) == 1:
        sides[arms[0][0]] = ""
    else:
        sides[arms[0][0]] = "L"
        sides[arms[-1][0]] = "R"
        for (lab, _) in arms[1:-1]:
            sides[lab] = "M"                 # 中间块(少见)

    out = []
    for (lab, cx) in arms:
        mask = labels == lab
        ys, xs = np.where(mask)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        mean, axis, proj = _principal_axis(pts)
        span = proj.max() - proj.min()
        # 垂直主轴方向的投影标准差 = 臂宽
        perp = np.array([-axis[1], axis[0]])
        width = float(np.std((pts - mean) @ perp)) if len(pts) > 2 else 1.0
        ratio = span / max(1.0, width)
        side = sides.get(lab, "")

        # 太短/太胖(长宽比<2) 不像手臂 -> 整块当 arm_hand
        if ratio < 2.0 or span < 60:
            seg = np.zeros((H, W), np.uint8)
            seg[ys, xs] = 255
            rgb = rgba.copy()
            rgb[..., 3] = np.minimum(rgba[..., 3], seg)
            out.append({"cls": "arm_hand", "side": side, "rgba": rgb})
            continue

        p0, p1 = proj.min(), proj.max()
        cuts = [p0, p0 + SEG_CUTS[0] * span, p0 + SEG_CUTS[1] * span, p1 + 1]
        for si in range(3):
            seg = _segment_mask(mask, proj, cuts[si], cuts[si + 1], (H, W))
            if int((seg > 0).sum()) < 200:
                continue
            rgb = rgba.copy()
            rgb[..., 3] = np.minimum(rgba[..., 3], seg)
            out.append({"cls": SEG_CLS[si], "side": side, "rgba": rgb})

    return out
