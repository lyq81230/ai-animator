"""用 AI 分层给出「头」的轮廓，用源图给出像素，重建 head 部件。

用法:
    python rebuild_head.py --work output/sprite_boss_dongzhuo_white_336 \
        --src E:/pic/sprite_boss_dongzhuo_white_336.png \
        --layers seethrough_repo/workspace/layerdiff_output/sprite_boss_dongzhuo_white_336 \
        --parts head --masks head.png,face.png [--dry]

思路:
  1. 用 --masks 列的层 alpha 并集(放大到画布尺寸)当头部轮廓; 取最大连通块, 顺带吃掉
     紧贴主体的碎块(被切开的护颊)。
  2. head 部件里落在轮廓外的像素搬给 --spill-to (默认 topwear)。之前用椭圆硬切,
     会把下巴切走。
  3. --erase-parts 里列的部件, 在轮廓内的像素抹掉(head 后画, 但这些是重复内容)。
     五官(eyewhite/nose/... )不能抹, 它们本来就压在脸上。
  4. head 部件用源图像素 + 轮廓 alpha 重写 -> 清晰且形状正确。
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reassign_region import move_region  # noqa: E402


def read_part(work, coords, name):
    p = next(q for q in coords["parts"] if q["name"] == name)
    a = np.array(Image.open(os.path.join(work, "parts", p["image"])).convert("RGBA"))
    return p, a


def part_canvas_mask(coords, name, W, H):
    p, a = read_part(None if False else "", coords, name) if False else (None, None)
    return None


def layer_mask(layers, name, W, H, thr=40):
    im = Image.open(os.path.join(layers, name)).convert("RGBA").resize((W, H), Image.LANCZOS)
    return np.array(im)[..., 3] > thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--parts", default="head")
    ap.add_argument("--masks", default="head.png,face.png")
    ap.add_argument("--spill-to", default="topwear")
    ap.add_argument("--erase-parts", default="topwear")
    ap.add_argument("--close", type=int, default=3)
    ap.add_argument("--grow", type=int, default=1)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    work = args.work
    cpath = os.path.join(work, "coords.json")
    coords = json.load(open(cpath, encoding="utf-8"))
    W, H = coords["canvas"]["w"], coords["canvas"]["h"]

    m = None
    for nm in args.masks.split(","):
        nm = nm.strip()
        if not nm:
            continue
        mm = layer_mask(args.layers, nm, W, H)
        m = mm if m is None else (m | mm)
    if m is None or not m.any():
        sys.exit("轮廓为空, 检查 --masks")
    if args.close > 1:
        m = cv2.morphologyEx(m.astype(np.uint8),
                             cv2.MORPH_CLOSE, np.ones((args.close, args.close), np.uint8)) > 0
    if args.grow:
        m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8),
                       iterations=args.grow) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n > 1:
        big = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA])
        core = cv2.dilate((lab == big).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        m = core & m
    ys, xs = np.nonzero(m)
    print(f"头轮廓 {int(m.sum())}px bbox({xs.min()},{ys.min()})-({xs.max()},{ys.max()})")

    src = np.array(Image.open(args.src).convert("RGBA"))
    if src.shape[1] != W:
        src = np.array(Image.fromarray(src).resize((W, H), Image.NEAREST))

    for pname in [s.strip() for s in args.parts.split(",") if s.strip()]:
        coords = json.load(open(cpath, encoding="utf-8"))
        p, a = read_part(work, coords, pname)
        cur = np.zeros((H, W, 4), np.uint8)
        cur[p["y"]:p["y"] + a.shape[0], p["x"]:p["x"] + a.shape[1]] = a
        inside = cur[..., 3] > 12
        print(f"{pname}: 现有 {int(inside.sum())}px, 轮廓内 {int((inside & m).sum())}px")

        out_mask = inside & ~m
        if out_mask.any():
            print(f"   -> 搬给 {args.spill_to}: {int(out_mask.sum())}px")
            if not args.dry:
                move_region(work, pname, args.spill_to, out_mask)

        if args.dry:
            continue

        # 抹掉 --erase-parts 在轮廓内的重复像素
        coords = json.load(open(cpath, encoding="utf-8"))
        for q in coords["parts"]:
            if q["name"] not in [s.strip() for s in args.erase_parts.split(",")]:
                continue
            fp = os.path.join(work, "parts", q["image"])
            qa = np.array(Image.open(fp).convert("RGBA"))
            sl = (slice(q["y"], q["y"] + qa.shape[0]), slice(q["x"], q["x"] + qa.shape[1]))
            hit = m[sl] & (qa[..., 3] > 0)
            if hit.any():
                qa[hit, 3] = 0
                Image.fromarray(qa).save(fp)
                print(f"   从 {q['name']} 抹掉 {int(hit.sum())}px")

        # 用源图重写 head
        coords = json.load(open(cpath, encoding="utf-8"))
        p = next(q for q in coords["parts"] if q["name"] == pname)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        crop = src[y0:y1, x0:x1].copy()
        mc = m[y0:y1, x0:x1]
        crop[..., 3] = np.where(mc, np.maximum(crop[..., 3], 220), 0)
        Image.fromarray(crop).save(os.path.join(work, "parts", p["image"]))
        p["x"], p["y"] = x0, y0
        p["width"], p["height"] = x1 - x0, y1 - y0
        json.dump(coords, open(cpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"   {pname} 重写为 {p['width']}x{p['height']} @({p['x']},{p['y']})")


if __name__ == "__main__":
    main()
