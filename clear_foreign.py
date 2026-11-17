# -*- coding: utf-8 -*-
"""清掉"跑到别的部件里"的残片(手、武器等)。

现象(全): topwear 右下角带着一块拳头, leg_R 右侧带着一条刀身 —— 那是 AI 分层把
手臂/武器也画进了躯干和腿的层里。生成后, 手动、腿动, 那些残片会留在原地,
看着像"手和武器粘在身上"。

做法: 找出【躯干类部件】和【四肢/道具】的重叠区, 只处理**小块**(默认 <= 600px 的
连通块, 拳头/刀刃这个量级), 把重叠处的颜色用周围像素 inpaint 补掉 —— alpha 不动,
所以绑定时被上面那层盖住、看不出来; 上面那层移开后露出的是补好的盔甲色, 不是拳头。

大块重叠(整条胳膊压在躯干上)不处理: 那里本来就是"被挡住的部分", 一补就是大片糊。

用法: python clear_foreign.py output/sprite_quan_white_336 [上限px=600]
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# bottomwear(裙甲)也算宿主: suo 的右手整块被判进裙甲右缘(342px), 手一动就露出
# 一只"粘在裙子上的手"。补色只动 RGB 不动 alpha, 手静止时被手盖住看不见。
HOST = ("topwear", "bottomwear", "leg_L", "leg_R", "headwear", "body")
FOREIGN_CLS = {"arm_upper", "arm_fore", "arm_hand", "prop"}


def main():
    D = sys.argv[1]
    CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    cp = os.path.join(D, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    W, H = co["canvas"]["w"], co["canvas"]["h"]
    by = {p["name"]: p for p in co["parts"]}

    def canvas_of(p):
        a = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))
        m = np.zeros((H, W, 4), np.uint8)
        m[p["y"]:p["y"] + a.shape[0], p["x"]:p["x"] + a.shape[1]] = a
        return m

    # 先把"外来者"的轮廓都算好
    foreign = np.zeros((H, W), bool)
    for p in co["parts"]:
        if str(p.get("cls") or "") in FOREIGN_CLS or p["name"].startswith("objects"):
            foreign |= canvas_of(p)[..., 3] > 40

    import cv2

    total = 0
    for p in co["parts"]:
        if p["name"] not in HOST:
            continue
        c = canvas_of(p)
        m = c[..., 3] > 40
        ov = m & foreign
        if not ov.any():
            continue
        lab, k = ndi.label(ov)
        hit = np.zeros_like(ov)
        for i in range(1, k + 1):
            blk = lab == i
            if blk.sum() <= CAP:
                hit |= blk
        if not hit.any():
            continue
        # 用"最近的正常像素"补色(alpha 保留)。比 cv2.inpaint 好: inpaint 会把
        # 几十像素的条带糊成一团, 最近邻补色只是把旁边的盔甲色延过去, 看着是平的。
        keep = (c[..., 3] > 40) & ~hit
        if keep.any():
            _d, (iy, ix) = ndi.distance_transform_edt(~keep, return_indices=True)
            c[..., :3][hit] = c[iy[hit], ix[hit]][:, :3]
        ys, xs = np.nonzero(c[..., 3] > 8)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        Image.fromarray(c[y0:y1 + 1, x0:x1 + 1]).save(
            os.path.join(D, "parts", p["image"]))
        p.update({"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1})
        n = int(hit.sum())
        total += n
        print(f"  {p['name']:<12} 补掉 {n:4d}px 残片")

    print(f"共 {total}px" + ("" if total else "  (没有残片)"))
    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pc = os.path.join(D, "parts", "coords.json")
    if os.path.isfile(pc):
        json.dump(co, open(pc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
