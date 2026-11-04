# -*- coding: utf-8 -*-
"""把"没有归到任何部件"的大块像素补成一个道具部件。

起因(皮甲): 拆图后盾牌那 2563px 谁都没接 —— 它既不在 topwear 里(被手臂分离挖掉了),
也没被 arms/props 分开步接收到, 于是渲染时整块消失。
判据: 源图 alpha 有、但所有部件都没覆盖的连通块, 面积 >= 400px 就当漏掉的道具。

用法: python add_missing_props.py output/sprite_pi_white_336
之后建议再跑一次 fill_holes.py 收掉残余的 1px 缝。
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def main():
    D = sys.argv[1]
    c = json.load(open(os.path.join(D, "coords.json"), encoding="utf-8"))
    W, H = c["canvas"]["w"], c["canvas"]["h"]
    # 源图: 按输出目录名去 E:/pic 找同名 png
    name = os.path.basename(os.path.abspath(D))
    src = None
    for cand in (os.path.join("E:/pic", name + ".png"),):
        if os.path.isfile(cand):
            src = np.array(Image.open(cand).convert("RGBA"))
            break
    if src is None:
        sys.exit("找不到源图")
    if src.shape[:2] != (H, W):
        src = np.array(Image.fromarray(src).resize((W, H), Image.LANCZOS))

    cov = np.zeros((H, W), np.int32)
    for p in c["parts"]:
        a = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))[..., 3] > 8
        cov[p["y"]:p["y"] + a.shape[0], p["x"]:p["x"] + a.shape[1]] += a
    hole = (src[..., 3] > 8) & (cov == 0)
    lab, n = ndi.label(hole)
    made = []
    # 躯干中线: 用来判断左右
    sub = [p for p in c["parts"] if p["name"] in ("topwear", "body")]
    cx0 = (sub[0]["x"] + sub[0]["width"] / 2) if sub else W / 2

    for i in range(1, n + 1):
        m = lab == i
        if int(m.sum()) < 400:
            continue
        ys, xs = np.nonzero(m)
        side = "R" if xs.mean() > cx0 else "L"
        pname = "objects_" + side
        k = 1
        while any(q["name"] == pname for q in c["parts"]):
            k += 1
            pname = f"objects_{side}{k}"
        canvas = np.zeros((H, W, 4), np.uint8)
        canvas[m] = src[m]
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        crop = canvas[y0:y1 + 1, x0:x1 + 1]
        Image.fromarray(crop).save(os.path.join(D, "parts", pname + ".png"))
        # 跟最近的那只手
        hands = [q for q in c["parts"] if q.get("cls") == "arm_hand"]
        follow = None
        best = 1e9
        for q in hands:
            hx = q["x"] + q["width"] / 2
            d = abs(hx - xs.mean()) + abs((q["y"] + q["height"] / 2) - ys.mean())
            if d < best:
                best, follow = d, q["name"]
        c["parts"].append({
            "name": pname, "image": pname + ".png",
            "x": int(x0), "y": int(y0),
            "width": int(x1 - x0 + 1), "height": int(y1 - y0 + 1),
            "grid": [2, 4] if (y1 - y0) > 60 else [2, 2],
            "cls": "prop", **({"follow_part": follow} if follow else {}),
        })
        made.append((pname, int(m.sum()), (int(x0), int(y0)), follow))

    json.dump(c, open(os.path.join(D, "coords.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    for nm, px, xy, fol in made:
        print(f"  补出漏掉的道具: {nm} {px}px @{xy} 跟随 {fol}")
    if not made:
        print("  没有需要补的道具")


if __name__ == "__main__":
    main()
