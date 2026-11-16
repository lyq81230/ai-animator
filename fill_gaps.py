# -*- coding: utf-8 -*-
"""把所有"源图有像素、但部件没盖实"的地方并回最近的部件。

两类毛病都由它兜
----------------
1) **洞**: 没有任何部件覆盖 -> 合成后露出背景(白点/白边)。缝合处最常见的
   "头和衣服之间那道白缝"就是它。
2) **半透明缝**: 有部件盖着, 但只是出血(bleed)补出来的 0.55 半透明, 背后还是
   背景 -> 看上去是"淡白虚线", 一放大特别明显。源图那里是不透明的, 所以直接
   把源图像素按原 alpha 写回部件即可, 不会吃掉邻居的描边。

做法
----
1. 算 acc = 该像素在所有部件里的最大 alpha;
2. 需要补的地方 = 源图有内容(alpha>60) ∧ acc < 235;
3. 从"已经盖实(acc>=235)"的像素往外做若干轮传播, 让每个待补像素认领最近的部件;
   传播不到的孤岛(烟雾碎缕)用最近邻(<=20px)认领;
4. 认领到的像素按**源图原样**写进那个部件, 必要时扩窗 + 更新 coords。

只补不删: 已有像素一律不动, 最差也只是"本来露底的地方被邻居补上"。

用法:
    python fill_gaps.py output/<角色> [--solid 235] [--rounds 6] [--dry]
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--solid", type=int, default=235,
                    help="acc 低于它就算没盖实(默认 235)")
    ap.add_argument("--src-alpha", type=int, default=60,
                    help="源图 alpha 高于它才算有内容(默认 60)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--grow", type=int, default=0,
                    help="缝里让相邻部件重叠几像素(默认 0; 出 Spine 包用 2)")
    ap.add_argument("--src", default="",
                    help="原画路径(默认按角色名去 E:/pic 找; app 会显式传)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    D = a.work
    pc = os.path.join(D, "parts", "coords.json")
    wc = os.path.join(D, "coords.json")
    cp = wc
    if os.path.isfile(pc) and (not os.path.isfile(wc)
                               or os.path.getmtime(pc) > os.path.getmtime(wc) + 0.5):
        cp = pc
    if not os.path.isfile(cp):
        raise SystemExit(f"没有 coords.json: {pc}")
    co = json.load(open(cp, encoding="utf-8"))
    W, H = int(co["canvas"]["w"]), int(co["canvas"]["h"])

    char = os.path.basename(os.path.normpath(D))
    src_p = a.src if (a.src and os.path.isfile(a.src)) else os.path.join("E:/pic", char + ".png")
    if not os.path.isfile(src_p):
        print(f"找不到源图 {src_p}, 跳过")
        return 0
    src = np.array(Image.open(src_p).convert("RGBA"))
    if src.shape[1] != W or src.shape[0] != H:
        src = np.array(Image.fromarray(src).resize((W, H), Image.LANCZOS))

    parts = co["parts"]
    imgs = []
    acc = np.zeros((H, W), np.uint8)
    label = np.full((H, W), -1, np.int16)
    for i, p in enumerate(parts):
        im = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))
        imgs.append(im)
        h, w = im.shape[:2]
        x0, y0 = max(0, p["x"]), max(0, p["y"])
        x1, y1 = min(W, p["x"] + w), min(H, p["y"] + h)
        if x1 <= x0 or y1 <= y0:
            continue
        al = im[y0 - p["y"]:y1 - p["y"], x0 - p["x"]:x1 - p["x"], 3]
        acc[y0:y1, x0:x1] = np.maximum(acc[y0:y1, x0:x1], al)
        label[y0:y1, x0:x1][al > 20] = i

    solid = acc >= a.solid
    gap = (src[:, :, 3] > a.src_alpha) & ~solid
    print(f"未盖实 {int(gap.sum())}px (全空 {int((gap & (label < 0)).sum())}px)")
    if not gap.any():
        return 0

    lab = np.where(solid, label, -1).astype(np.int16)
    for _ in range(a.rounds):
        newly = (lab < 0) & gap
        if not newly.any():
            break
        cand = np.full((H, W), -1, np.int16)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.full((H, W), -1, np.int16)
            ys = slice(max(0, dy), H + min(0, dy))
            yd = slice(max(0, -dy), H + min(0, -dy))
            xs = slice(max(0, dx), W + min(0, dx))
            xd = slice(max(0, -dx), W + min(0, -dx))
            sh[yd, xd] = lab[ys, xs]
            take = (cand < 0) & (sh >= 0)
            cand[take] = sh[take]
        fill = newly & (cand >= 0)
        lab[fill] = cand[fill]

    left = int(((lab < 0) & gap).sum())
    if left:
        ys, xs = np.nonzero(lab >= 0)
        if len(xs):
            from scipy.spatial import cKDTree
            tree = cKDTree(np.column_stack([xs, ys]))
            ry, rx = np.nonzero(lab < 0)
            dist, idx = tree.query(np.column_stack([rx, ry]), k=1)
            okd = dist <= 20
            for yy, xx, k in zip(ry[okd], rx[okd], idx[okd]):
                lab[yy, xx] = lab[ys[k], xs[k]]
        left = int(((lab < 0) & gap).sum())
    if left:
        print(f"  {left}px 离所有部件都太远, 留着")

    changed = 0
    for i, p in enumerate(parts):
        sel = (lab == i) & gap
        cnt = int(sel.sum())
        if cnt == 0:
            continue
        changed += cnt
        print(f"  {p['name']:<16} +{cnt}px")
        if a.dry:
            continue
        im = imgs[i]
        h, w = im.shape[:2]
        ys, xs = np.nonzero(sel)
        nx0 = min(p["x"], int(xs.min()))
        ny0 = min(p["y"], int(ys.min()))
        nx1 = max(p["x"] + w, int(xs.max()) + 1)
        ny1 = max(p["y"] + h, int(ys.max()) + 1)
        canvas = np.zeros((ny1 - ny0, nx1 - nx0, 4), np.uint8)
        canvas[p["y"] - ny0:p["y"] - ny0 + h, p["x"] - nx0:p["x"] - nx0 + w] = im
        for yy, xx in zip(ys, xs):
            canvas[yy - ny0, xx - nx0] = src[yy, xx]
        Image.fromarray(canvas).save(os.path.join(D, "parts", p["image"]))
        p["x"], p["y"], p["width"], p["height"] = nx0, ny0, nx1 - nx0, ny1 - ny0

    if not a.dry and changed:
        for f in (pc, wc):
            if os.path.isdir(os.path.dirname(f)):
                json.dump(co, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"补齐 {changed}px" + (" (dry)" if a.dry else ""))

    if a.grow > 0 and not a.dry:
        grow_seams(D, co, src, parts, a.grow)
    return 0


def grow_seams(D, co, src, parts, n):
    """每个部件沿缝往外多吃 n 像素(填源图颜色, alpha 原样)。

    为什么: Spine 网格的边界顶点是**贴着不透明区**摆的, 相邻两块各缩进半像素,
    渲染出来就是一条白虚线(编辑器里特别明显, 用户说的"拆图还是错")。让相邻部件
    在缝里重叠 n 像素, 网格怎么缩都不会露底; 因为填进去的是**源图原像素**,
    合成结果一个像素都不变。
    """
    H, W = src.shape[:2]
    fig = src[:, :, 3] > 60
    total = 0
    for p in parts:
        f = os.path.join(D, "parts", p["image"])
        im = np.array(Image.open(f).convert("RGBA"))
        h, w = im.shape[:2]
        # 先摊到全画布上算, 免得在窗口边缘把扩张截断
        canv = np.zeros((H, W, 4), np.uint8)
        x0, y0 = max(0, p["x"]), max(0, p["y"])
        x1, y1 = min(W, p["x"] + w), min(H, p["y"] + h)
        canv[y0:y1, x0:x1] = im[y0 - p["y"]:y1 - p["y"], x0 - p["x"]:x1 - p["x"]]
        m = canv[..., 3] > 8
        add = np.zeros((H, W), bool)
        cur = m
        for _ in range(n):
            cur = cv2.dilate(cur.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            add |= cur & fig
        add &= ~m
        if not add.any():
            continue
        canv[add] = src[add]
        ys, xs = np.nonzero(canv[..., 3] > 8)
        nx0, ny0 = int(xs.min()), int(ys.min())
        nx1, ny1 = int(xs.max()) + 1, int(ys.max()) + 1
        Image.fromarray(canv[ny0:ny1, nx0:nx1]).save(f)
        p["x"], p["y"], p["width"], p["height"] = nx0, ny0, nx1 - nx0, ny1 - ny0
        total += int(add.sum())
    for fpath in (os.path.join(D, "parts", "coords.json"), os.path.join(D, "coords.json")):
        if os.path.isdir(os.path.dirname(fpath)):
            json.dump(co, open(fpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"缝隙重叠 +{total}px ({n}px)")


if __name__ == "__main__":
    raise SystemExit(main())
