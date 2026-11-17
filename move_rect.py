# -*- coding: utf-8 -*-
"""把某个部件里一个画布矩形内的像素整块搬到另一个部件。

场景: AI 分层把"甲片/布料"判给了武器层 —— 武器跟手骨甩, 甲片留在武器上
看着像"剑带了块盔甲"(董卓 sword 左下 328px 裙甲)。甲片在原画里被剑压住的
部分属于底层(衣服), 矩形里若混有武器自己的像素(剑刃), 用矩形边界避开。

和 move_island 的区别: move_island 按"连通孤岛"搬(岛必须与本体断开);
本脚本按【画布矩形】搬 —— 残块与本体相连时用它。画法上被搬走的像素在
目标部件里画在更底层, 原来盖着它的东西(武器刃)依然盖着, 观感不变。

用法:
    python move_rect.py output/<角色> --from sword --to topwear --rect x0,y0,x1,y1
    (rect 为画布坐标, 左闭右开)
"""
import json
import os
import sys

import numpy as np
from PIL import Image


def main():
    D = sys.argv[1]
    src_name = sys.argv[sys.argv.index("--from") + 1]
    to_name = sys.argv[sys.argv.index("--to") + 1]
    x0, y0, x1, y1 = (int(v) for v in
                      sys.argv[sys.argv.index("--rect") + 1].split(","))
    cp = os.path.join(D, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    W, H = co["canvas"]["w"], co["canvas"]["h"]
    by = {p["name"]: p for p in co["parts"]}
    for n in (src_name, to_name):
        if n not in by:
            print(f"没有部件 {n}")
            return 1

    def load(p):
        return np.array(Image.open(os.path.join(D, "parts", p["image"]))
                        .convert("RGBA"))

    def paste(p, a):
        ys, xs = np.nonzero(a[..., 3] > 8)
        if len(ys) == 0:
            return
        x0_, y0_ = int(xs.min()), int(ys.min())
        x1_, y1_ = int(xs.max()) + 1, int(ys.max()) + 1
        Image.fromarray(a[y0_:y1_, x0_:x1_]).save(
            os.path.join(D, "parts", p["image"]))
        p.update({"x": p["x"] + x0_, "y": p["y"] + y0_,
                  "width": x1_ - x0_, "height": y1_ - y0_})

    ps, pt = by[src_name], by[to_name]
    a_s, a_t = load(ps), load(pt)

    def to_local(p, cx, cy):
        return cy - p["y"], cx - p["x"]

    moved = 0
    for cy in range(max(y0, 0), min(y1, H)):
        for cx in range(max(x0, 0), min(x1, W)):
            ly, lx = to_local(ps, cx, cy)
            if not (0 <= ly < a_s.shape[0] and 0 <= lx < a_s.shape[1]):
                continue
            if a_s[ly, lx, 3] <= 8:
                continue
            ty, tx = to_local(pt, cx, cy)
            # 目标部件该处已有不透明像素时不覆盖(保留原有内容)
            if (0 <= ty < a_t.shape[0] and 0 <= tx < a_t.shape[1]
                    and a_t[ty, tx, 3] > 200):
                a_s[ly, lx] = 0
                moved += 1
                continue
            # 目标画布不够大则扩到能装下
            need_h = ty + 1 - a_t.shape[0] if ty + 1 > a_t.shape[0] else 0
            need_w = tx + 1 - a_t.shape[1] if tx + 1 > a_t.shape[1] else 0
            if need_h or need_w or ty < 0 or tx < 0:
                top = -ty if ty < 0 else 0
                left = -tx if tx < 0 else 0
                nh = a_t.shape[0] + need_h + top
                nw = a_t.shape[1] + need_w + left
                pad = np.zeros((nh, nw, 4), np.uint8)
                pad[top:top + a_t.shape[0], left:left + a_t.shape[1]] = a_t
                a_t = pad
                ty += top
                tx += left
                ps_shift = None  # 目标窗口平移在下面统一处理
            a_t[ty, tx] = a_s[ly, lx]
            a_s[ly, lx] = 0
            moved += 1

    if moved == 0:
        print("  矩形内没有可搬的像素")
        return 0
    # 目标部件画布扩过时, 窗口原点要跟着移(向左/上扩了 top/left)
    paste(ps, a_s)
    paste(pt, a_t)
    print(f"  {moved}px: {src_name} -> {to_name} (rect {x0},{y0},{x1},{y1})")
    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pc = os.path.join(D, "parts", "coords.json")
    if os.path.isfile(pc):
        json.dump(co, open(pc, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=2)
    print("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
