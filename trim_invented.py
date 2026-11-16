# -*- coding: utf-8 -*-
"""清掉部件里"AI 补出来的"内容 —— 原画那里其实是别的部件。

现象(皮甲): arm_upper_L 里有一块灰绿色, 位置正好压在胸甲左片上, 把铆钉和甲片
分界盖住了。原因是拆图时 AI 把"被躯干挡住的那截手臂"也画了出来, 而手臂的绘制
顺序在躯干之后, 于是这块假的手臂盖到了胸甲上。原画那里是棕色胸甲。

判据: 部件某个像素的 RGB 和【源图同一像素】差很多, 且该像素被【先画的部件】盖着
      (说明"真实内容"是下面那个部件), 就把这个像素从部件里去掉。
      只去掉有下家的像素, 所以不会在轮廓上挖洞。

用法: python trim_invented.py output/sprite_pi_white_336 [阈值=45] [--dry]
"""
import json
import os
import sys

import numpy as np
from PIL import Image


# 只清四肢: 那里才会出现"AI 把被躯干挡住的那截也画出来"。五官(闭眼贴片、
# 眼球)本来就是和原图不同的内容, 清掉就把表情弄没了。
LIMB = {"arm_upper", "arm_fore", "arm_hand", "leg", "leg_up", "leg_low", "foot"}


def main():
    D = sys.argv[1]
    T = 45.0
    for a in sys.argv[2:]:
        try:
            T = float(a)
        except ValueError:
            pass
    dry = "--dry" in sys.argv
    cp = os.path.join(D, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    W, H = co["canvas"]["w"], co["canvas"]["h"]
    char = os.path.basename(os.path.normpath(D))
    src_p = os.path.join("E:/pic", char + ".png")
    src = np.array(Image.open(src_p).convert("RGBA"))
    if src.shape[1] != W or src.shape[0] != H:
        src = np.array(Image.fromarray(src).resize((W, H), Image.LANCZOS))

    order = sorted(range(len(co["parts"])),
                   key=lambda i: (float(co["parts"][i].get("z", 0) or 0), i))

    def canvas_of(p):
        a = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))
        m = np.zeros((H, W, 4), np.uint8)
        m[p["y"]:p["y"] + a.shape[0], p["x"]:p["x"] + a.shape[1]] = a
        return m

    below = np.zeros((H, W), bool)          # 先画的部件覆盖区
    total = 0
    for idx in order:
        p = co["parts"][idx]
        if str(p.get("cls") or "") not in LIMB:
            c = canvas_of(p)
            below |= c[..., 3] > 128
            continue
        c = canvas_of(p)
        m = c[..., 3] > 128
        diff = np.abs(c[..., :3].astype(int) - src[..., :3].astype(int)).mean(axis=2)
        bad = m & below & (src[..., 3] > 128) & (diff > T)
        if bad.any():
            n = int(bad.sum())
            total += n
            print(f"  {p['name']:<14} 清掉 {n:5d}px (与源图平均差 {diff[bad].mean():.0f})")
            if not dry:
                c[..., 3][bad] = 0
                # 重裁 bbox
                mm = c[..., 3] > 8
                ys, xs = np.nonzero(mm)
                x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
                Image.fromarray(c[y0:y1 + 1, x0:x1 + 1]).save(
                    os.path.join(D, "parts", p["image"]))
                p.update({"x": x0, "y": y0,
                          "width": x1 - x0 + 1, "height": y1 - y0 + 1})
                c = canvas_of(p)
                m = c[..., 3] > 128
        below |= m

    print(f"共清 {total}px" + ("  (dry)" if dry else ""))
    if not dry:
        json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        pc = os.path.join(D, "parts", "coords.json")
        if os.path.isfile(pc):
            json.dump(co, open(pc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
