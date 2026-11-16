# -*- coding: utf-8 -*-
"""把"肩甲 + 上臂"连在一起的 arm_upper_? 拆成两块, 好让它们各自有 z。

背景: 皮甲的 arm_upper_L 一块里既有棕色肩甲, 又有灰绿色上手臂。整块设 z=-1 时
肩甲正确地被躯干盖住, 但上臂也跟着被切掉了(用户: "右手臂上半部分 Z 值应比躯干大")。
拆开后: 肩甲 z=-1(被躯干遮), 上臂 z=+1(在躯干前面)。

判据: 棕色(暖) vs 灰绿(冷), 再按最近的已知像素补齐描边/抗锯齿的杂色, 保证不漏像素。

用法:
    python split_arm_pad.py output/sprite_pi_white_336 arm_upper_L
    python split_arm_pad.py output/sprite_pi_white_336 arm_upper_L --preview
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def masks(a):
    """棕色肩甲 vs 灰绿上臂。

    先取两块"高置信"的核心(饱和棕 / 明显偏绿), 再把描边、高光、抗锯齿这些
    判不准的像素按**离哪个核心近**归过去。直接按颜色阈值分会把肩甲的深色描边
    分给上臂, 上臂的 z=+1 就会把一圈描边画到躯干上(渲染出来是一坨)。
    """
    m = a[..., 3] > 60
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    warm = m & (r > g + 14) & (r > b + 30) & (r > 55)
    cool = m & (g >= r - 2) & (b >= r - 8) & (g > 55)
    # 去掉零星误判
    warm = ndi.binary_opening(warm, np.ones((2, 2)))
    cool = ndi.binary_opening(cool, np.ones((2, 2)))
    lab = np.zeros(a.shape[:2], np.int8)
    lab[warm] = 1
    lab[cool & ~warm] = 2
    if not (lab == 1).any() or not (lab == 2).any():
        return m, np.zeros_like(m)
    _d, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    assign = lab[iy, ix]
    pad = m & (assign == 1)
    arm = m & (assign == 2)
    return pad, arm


def main():
    D = sys.argv[1]
    name = sys.argv[2]
    preview_only = "--preview" in sys.argv
    cp = os.path.join(D, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    by = {p["name"]: p for p in co["parts"]}
    if name not in by:
        print("没有部件", name)
        return
    p = by[name]
    im = Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA")
    a = np.array(im)
    pad, arm = masks(a)
    print(f"{name}: 肩甲 {int(pad.sum())}px, 上臂 {int(arm.sum())}px")

    char = os.path.basename(os.path.normpath(D))
    SRC = os.environ.get("ARM_SRC") or os.path.join("E:/pic", char + ".png")
    if not os.path.isfile(SRC):
        print("找不到原图:", SRC)
        return

    if preview_only:
        out = np.zeros_like(a)
        out[pad] = [210, 90, 50, 255]
        out[arm] = [60, 160, 210, 255]
        Image.fromarray(out).resize((im.size[0] * 6, im.size[1] * 6),
                                    Image.NEAREST).save("_arm_split_preview.png")
        print("预览 -> _arm_split_preview.png")
        return

    pad_name = name.replace("arm_upper", "arm_pad")
    if pad_name in by:
        print("已经有", pad_name, "跳过")
        return

    # 上臂要放到躯干**前面**(z>0), 但部件里那截"被躯干挡住的手臂"是 AI 补出来的,
    # 原图那里其实是棕色胸甲。直接提上来会在胸口糊一块灰绿色。
    # 所以先按 topwear 的不透明区把这段剪掉 —— 剪完的形状正好等于原图里看得见的手臂。
    H, W = co["canvas"]["h"], co["canvas"]["w"]
    src = np.array(Image.open(SRC).convert("RGBA")).astype(int)
    ss = src[p["y"]:p["y"] + a.shape[0], p["x"]:p["x"] + a.shape[1]]
    sr, sg, sb = ss[..., 0], ss[..., 1], ss[..., 2]
    # 原图这里是棕色胸甲 -> 说明这块手臂是 AI 补出来的, 剪掉(深色描边不算棕, 会保留下来)
    is_chest = (sr > sg + 14) & (sr > sb + 30) & (ss[..., 3] > 60)
    trimmed = int((arm & is_chest).sum())
    arm = arm & ~is_chest
    print(f"  上臂剪掉 AI 补的 {trimmed}px(原图那里是胸甲)")

    def emit(mask, pname, z):
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        buf = np.zeros((y1 - y0 + 1, x1 - x0 + 1, 4), np.uint8)
        sub = a[y0:y1 + 1, x0:x1 + 1]
        sel = mask[y0:y1 + 1, x0:x1 + 1]
        buf[sel] = sub[sel]
        Image.fromarray(buf).save(os.path.join(D, "parts", pname + ".png"))
        q = dict(by[name])          # 继承 cls 等字段
        q.update({"name": pname, "image": pname + ".png",
                  "x": p["x"] + x0, "y": p["y"] + y0,
                  "width": x1 - x0 + 1, "height": y1 - y0 + 1, "z": z})
        q.pop("grid", None)
        return q

    pad_part = emit(pad, pad_name, -1.0)
    arm_part = emit(arm, name, 1.0)

    idx = [i for i, q in enumerate(co["parts"]) if q["name"] == name][0]
    co["parts"][idx] = arm_part
    co["parts"].insert(idx, pad_part)      # 肩甲先画
    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pc = os.path.join(D, "parts", "coords.json")
    if os.path.isfile(pc):
        json.dump(co, open(pc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"已拆: {pad_name} z=-1(先画), {name} z=+1(后画)")


if __name__ == "__main__":
    main()
