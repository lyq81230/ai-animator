"""清掉部件里的小杂点，并把画布裁到真实内容。

为什么需要:
    拆图时按 alpha 取像素, 原画里偶尔有 1~3px 的碎屑(比如武器旁边的光点)。
    它会把部件的画布撑大一大截: 魔神右手的剑真实内容 49x52, 因为刀尖下方 40px
    处有一条 2px 宽的碎屑, 画布变成 49x90 —— 刀身只占上面一半, 在编辑器里看着
    就像"刀尖到这里就断掉了"。

做什么:
    1. 连通块面积 < N 的岛直接删掉(默认 12px, 五官高光通常比这大);
    2. 画布裁到剩余内容的包围盒, 同步更新 coords(根目录 + parts 两份)。

用法:
    python trim_parts.py output/sprite_boss_moshen_white_336 [最小岛面积]
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def main():
    D = sys.argv[1].rstrip("/\\")
    N = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    cp = os.path.join(D, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    touched = 0
    for p in co["parts"]:
        f = os.path.join(D, "parts", p["image"])
        if not os.path.isfile(f):
            continue
        a = np.array(Image.open(f).convert("RGBA"))
        h, w = a.shape[:2]
        m = a[..., 3] > 0
        if not m.any():
            continue
        # 只有武器(prop)才做"远离主体的碎屑"清理: 四肢/躯干本来就是几块拼起来的,
        # 阈值一放就会把它们自己的部件当碎屑删掉(实测 arm_fore_L 被削碎)。
        is_prop = str(p.get("cls") or "") == "prop"
        lab, k = ndi.label(m, structure=np.ones((3, 3)))
        if k > 1:
            sizes = ndi.sum(m, lab, range(1, k + 1))
            big = int(np.argmax(sizes)) + 1
            main = lab == big
            # "远离主体的碎屑"也删: 面积不到主体的 6% + 离主体 > 8px + 自身 < 60px。
            # 只看绝对面积会漏掉 魔神右手剑 那种 18px、却离刀身 32px 的杂点 ——
            # 它把画布从 49x52 撑到 49x90, 刀身在编辑器里就只剩上半截。
            grown = ndi.binary_dilation(main, np.ones((17, 17)), iterations=1)
            small = np.zeros_like(m)
            for i, s in enumerate(sizes, 1):
                if i == big:
                    continue
                isl = lab == i
                far = is_prop and s < 0.06 * sizes[big - 1] and s < 60 and not (isl & grown).any()
                if s < N or far:
                    small |= isl
            if small.any():
                a[small] = 0
                m &= ~small
                if not m.any():
                    continue
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if (x1 - x0, y1 - y0) == (w, h):
            continue
        Image.fromarray(a[y0:y1, x0:x1]).save(f)
        p["x"] = p["x"] + int(x0)
        p["y"] = p["y"] + int(y0)
        p["width"] = int(x1 - x0)
        p["height"] = int(y1 - y0)
        touched += 1
        print(f"  {p['name']:<13} {w}x{h} -> {x1-x0}x{y1-y0}")
    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    cp2 = os.path.join(D, "parts", "coords.json")
    if os.path.isfile(cp2):
        json.dump(co, open(cp2, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"清理/裁边完成: {touched} 个部件有变化")


if __name__ == "__main__":
    main()
