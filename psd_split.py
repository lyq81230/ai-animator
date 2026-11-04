 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psd_split.py

读 PSD -> 导出叶子图层 PNG, 生成 coords.json。
支持:
  --group NAME      只导出某个组(如女生组 nv)
  --align-video V   自动算 PSD->视频 相似变换, 把坐标/尺寸烘到【视频像素空间】,
                    canvas 设为视频分辨率。对齐时排除鸟层(它在 f0 不存在)。
  --bird-prefix     标记"后出现"的部件(鸟), 打 "bird": true, 供后续做透明+按帧追踪。

输出 coords.json 的 x,y,width,height 已是目标空间坐标。
"""

import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image

try:
    from psd_tools import PSDImage
except ImportError:
    sys.exit("缺少 psd-tools: pip install psd-tools")

try:
    import cv2
except ImportError:
    cv2 = None


def sanitize(name):
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", name.strip())
    return s.strip("_") or "layer"


def find_group(psd, name):
    for x in psd:
        if x.is_group() and x.name == name:
            return x
    # 深层找
    for x in psd.descendants():
        if x.is_group() and x.name == name:
            return x
    return None


def compute_transform(psd, root, video_path, bird_prefixes):
    """
    算相似变换(均匀缩放+平移): PSD画布坐标 -> 视频像素。
    对齐用 root 组去掉鸟层后的内容 bbox, 对齐到视频 f0 去绿幕后的角色 bbox。
    返回 (s, tx, ty, vw, vh)
    """
    cap = cv2.VideoCapture(video_path)
    ok, f0 = cap.read()
    vw_res, vh_res = int(cap.get(3)), int(cap.get(4))
    cap.release()
    if not ok:
        sys.exit("读不到视频首帧")

    hsv = cv2.cvtColor(f0, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 60, 60), (90, 255, 255))
    char = cv2.bitwise_not(green)
    char = cv2.morphologyEx(char, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.where(char > 0)
    vy0, vy1 = ys.min(), ys.max()
    vx0, vx1 = xs.min(), xs.max()
    char_h = vy1 - vy0
    vcx, vcy = (vx0 + vx1) / 2, (vy0 + vy1) / 2

    # PSD 去鸟内容 bbox(整画布坐标)
    hidden = []
    for lyr in root.descendants():
        if (not lyr.is_group() and lyr.visible
                and lyr.name.startswith(bird_prefixes)):
            lyr.visible = False
            hidden.append(lyr)
    comp = np.array(root.composite(viewport=psd.viewbox))
    for lyr in hidden:
        lyr.visible = True
    alpha = comp[:, :, 3]
    ys, xs = np.where(alpha > 10)
    py0, py1 = ys.min(), ys.max()
    px0, px1 = xs.min(), xs.max()
    ph = py1 - py0
    pcx, pcy = (px0 + px1) / 2, (py0 + py1) / 2

    s = char_h / ph
    tx = vcx - s * pcx
    ty = vcy - s * pcy
    print(f"[align] scale={s:.4f} tx={tx:.1f} ty={ty:.1f} "
          f"video={vw_res}x{vh_res}")
    return s, tx, ty, vw_res, vh_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd", required=True)
    ap.add_argument("--parts", default="psd_parts")
    ap.add_argument("--coords", default="coords.json")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--group", default="", help="只导出某组")
    ap.add_argument("--align-video", default="", help="按此视频算变换并烘坐标")
    ap.add_argument("--max-grid", type=int, default=10, help="最大网格密度")
    ap.add_argument("--max-canvas", type=int, default=2048,
                    help="无视频时画布最长边上限(px); 超大 PSD 缩到此内防崩, 0=不缩")
    ap.add_argument("--px-per-cell", type=int, default=60,
                    help="网格每格目标像素(视频空间), 越小网格越密")
    ap.add_argument("--bird-prefix", default="niao,zhua,yanjing",
                    help="后出现部件(鸟)的图层名前缀, 逗号分隔")
    ap.add_argument("--include-hidden", action="store_true", default=True,
                    help="包含隐藏图层(默认开启)")
    ap.add_argument("--skip-hidden", dest="include_hidden",
                    action="store_false",
                    help="跳过隐藏图层(旧行为)")
    args = ap.parse_args()

    psd = PSDImage.open(args.psd)
    print(f"PSD 画布: {psd.width} x {psd.height}")

    bird_prefixes = tuple(p for p in args.bird_prefix.split(",") if p)

    root = psd
    if args.group:
        g = find_group(psd, args.group)
        if g is None:
            sys.exit(f"找不到组: {args.group}")
        root = g
        print(f"只导出组: {args.group}")

    # 变换
    s, tx, ty = 1.0, 0.0, 0.0
    canvas = [psd.width, psd.height]
    if args.align_video:
        if cv2 is None:
            sys.exit("--align-video 需要 opencv-python")
        topgrp = root if root is not psd else psd
        s, tx, ty, vw, vh = compute_transform(
            psd, topgrp, args.align_video, bird_prefixes)
        canvas = [vw, vh]
    else:
        # 无视频(Idle): 超大 PSD 画布按比例缩到 max_canvas 内, 否则下游
        # 合成/网格/预览处理 8192×4872 这类巨图 -> 内存爆炸/原生崩溃(App 闪退)。
        mc = args.max_canvas
        big = max(psd.width, psd.height)
        if mc > 0 and big > mc:
            s = mc / float(big)
            canvas = [max(1, round(psd.width * s)), max(1, round(psd.height * s))]
            print(f"画布超 {mc}px, 缩放 s={s:.4f} -> {canvas[0]}x{canvas[1]}")

    os.makedirs(args.parts, exist_ok=True)

    iterable = root.descendants() if root is not psd else psd.descendants()

    parts = []
    used = set()
    n_hidden = 0
    for layer in iterable:
        if layer.is_group():
            continue
        hidden = not layer.visible
        if hidden and not args.include_hidden:
            continue
        # 隐藏层 composite 可能返回空 -> 临时置可见再合成
        if hidden:
            try:
                layer.visible = True
                img = layer.composite()
            finally:
                layer.visible = False
            n_hidden += 1
        else:
            img = layer.composite()
        if img is None or img.width == 0 or img.height == 0:
            continue

        base = sanitize(layer.name)
        fname = base
        k = 1
        while fname in used:
            fname = f"{base}_{k}"
            k += 1
        used.add(fname)

        png = fname + ".png"
        img_rgba = img.convert("RGBA")
        # 画布缩放时同步缩放部件 PNG, 否则导出图仍是 PSD 原始巨尺寸
        if s != 1.0:
            nw = max(1, round(img_rgba.width * s))
            nh = max(1, round(img_rgba.height * s))
            img_rgba = img_rgba.resize((nw, nh), Image.LANCZOS)
        img_rgba.save(os.path.join(args.parts, png))

        # PSD 画布坐标 -> 目标空间
        vx = s * layer.left + tx
        vy = s * layer.top + ty
        vw_ = s * img.width
        vh_ = s * img.height

        gx = max(2, min(args.max_grid, round(vw_ / args.px_per_cell)))
        gy = max(2, min(args.max_grid, round(vh_ / args.px_per_cell)))

        part = {
            "name": fname,
            "image": png,
            "x": round(vx, 2),
            "y": round(vy, 2),
            "width": round(vw_, 2),
            "height": round(vh_, 2),
            "grid": [gx, gy],
        }
        if layer.name.startswith(bird_prefixes):
            part["bird"] = True
        parts.append(part)

    coords = {
        "fps": args.fps,
        "canvas": canvas,
        "transform": {"scale": s, "tx": tx, "ty": ty},
        "parts": parts,
    }
    with open(args.coords, "w", encoding="utf-8") as f:
        json.dump(coords, f, ensure_ascii=False, indent=2)

    n_bird = sum(1 for p in parts if p.get("bird"))
    print(f"完成: {len(parts)} 部件 ({n_bird} 鸟层, {n_hidden} 隐藏层已包含) -> {args.parts}/")
    print(f"坐标表(目标空间) -> {args.coords}")


if __name__ == "__main__":
    main()
