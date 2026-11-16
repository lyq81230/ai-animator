#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""补全被切平的部件底边（拆图 bbox 裁切留下的直线切口）。

拆图时每个部件的 PNG 是按"分配给该 slot 的像素 bbox"裁出来的。如果部件的
真实轮廓伸出了这个 bbox（典型：手的下缘、脚的脚尖），PNG 底边就会是一条
几十像素宽的水平直线——透明背景直接贴着皮肤像素，没有收边描边。

本脚本检测到这种"底边直切"后，用一条圆润的下缘把它收口：

1. 判定切口行：从最底往上找"跨度接近满宽"的最后一行（真轮廓越往下越窄，
   被裁的行则整行同宽）。
2. 按 --curve 给出的下探量生成收口曲线（两端归零、中间最深），只用
   该列最底部几行的平均色铺底，并按深度渐暗成下缘阴影，不复制上方褶皱
   （镜像填充会做出"多一道褶"的假手指）。
3. 沿新增轮廓补 2px 深色描边，和原画的粗描边风格一致。
4. 同步改写 coords.json 里的 height（output/<名>/coords.json 与
   parts/coords.json 两份都改），避免 w/h 与实际 PNG 不一致。

用法：
    python fix_part_cut.py --parts output/fa/parts --name arm_hand_L --dry-run
    python fix_part_cut.py --parts output/fa/parts --name arm_hand_L
    python fix_part_cut.py --parts output/fa/parts --name arm_hand_L --curve 0,4,9,7,3,0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image

OUTLINE = (13, 12, 15)


def load_rgba(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA")).astype(np.float32)


def find_cut_row(alpha: np.ndarray, min_ratio: float = 0.75) -> int | None:
    """从底往上找最后一行"跨度接近满宽"的行，即被裁切的边界。"""
    h, w = alpha.shape
    for y in range(h - 1, -1, -1):
        idx = np.nonzero(alpha[y])[0]
        if len(idx) == 0:
            continue
        if (idx.max() - idx.min() + 1) >= min_ratio * w:
            return y
    return None


def smooth_curve(width: int, depths: list[float]) -> np.ndarray:
    """把 --curve 的分段深度插值成逐列下探量。"""
    n = len(depths)
    xs = np.linspace(0, width - 1, n)
    return np.interp(np.arange(width), xs, np.array(depths, float)).clip(0)


def close_bottom(img: np.ndarray, cut: int, depths: list[float],
                 dark: float) -> tuple[np.ndarray, int]:
    """在切口下方生成圆润收口，返回 (新图, 新增行数)。"""
    base = img[: cut + 1].copy()
    h, w = base.shape[:2]
    depth = smooth_curve(w, depths)
    extra = int(np.ceil(depth.max())) + 2
    nh = h + extra

    mask = np.zeros((nh, w), bool)
    for x in range(w):
        if depth[x] <= 0:
            continue
        # 该列底部必须有真实像素，否则会在空白列上造出一条悬空的细线
        if not (base[max(0, h - 3):h, x, 3] > 20).any():
            continue
        ye = min(nh - 1, int(round(h + depth[x])))
        mask[h:ye + 1, x] = True

    out = np.concatenate([base, np.zeros((extra, w, 4), np.float32)], 0)
    # 每列用最底部几行的平均色铺底（保住这一列的肤色倾向）
    for x in range(w):
        col = np.nonzero(base[:, x, 3] > 20)[0]
        if len(col) == 0:
            continue
        color = base[max(0, col.max() - 3): col.max() + 1, x, :3].mean(0)
        ys = np.nonzero(mask[:, x])[0]
        if len(ys) == 0:
            continue
        span = max(1.0, depth[x])
        for y in ys:
            out[y, x, :3] = color * (1.0 - dark * min(1.0, (y - h) / span))
            out[y, x, 3] = 255

    # 新区域轻度横向平滑，去掉逐列硬跳
    rgb = out[:, :, :3]
    blur = cv2.blur(rgb.astype(np.uint8), (3, 3)).astype(np.float32)
    for y in range(h, nh):
        m = mask[y]
        if m.any():
            rgb[y, m] = 0.45 * rgb[y, m] + 0.55 * blur[y, m]

    # 2px 描边，只画在新增部分，避免把原有轮廓再压黑一遍
    full = out[:, :, 3] > 20
    inner = full.copy()
    for _ in range(2):
        er = inner.copy()
        er[1:, :] &= inner[:-1, :]
        er[:-1, :] &= inner[1:, :]
        er[:, 1:] &= inner[:, :-1]
        er[:, :-1] &= inner[:, 1:]
        inner = er
    edge = full & ~inner
    edge[:h] = False
    out[edge] = [*OUTLINE, 255]
    return out, extra


def update_coords(parts_dir: str, name: str, new_h: int) -> list[str]:
    """同步 height 字段（parts/ 与上层各一份 coords.json）。"""
    changed = []
    for path in (os.path.join(parts_dir, "coords.json"),
                 os.path.join(os.path.dirname(parts_dir.rstrip("/\\")), "coords.json")):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        hit = False
        for p in data.get("parts", []):
            if p.get("name") == name:
                if p.get("height") != new_h:
                    p["height"] = new_h
                    hit = True
        if hit:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            changed.append(path)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="补全被切平的部件底边")
    ap.add_argument("--parts", required=True, help="部件目录，如 output/fa/parts")
    ap.add_argument("--name", required=True, help="部件名，如 arm_hand_L")
    ap.add_argument("--curve", default="0,4,9,7,3,0",
                    help="从左到右各点下探像素数，逗号分隔（默认 0,4,9,7,3,0）")
    ap.add_argument("--dark", type=float, default=0.32, help="下缘渐暗强度（默认 0.32）")
    ap.add_argument("--min-ratio", type=float, default=0.75,
                    help="判定直切的最小跨度占比（默认 0.75）")
    ap.add_argument("--dry-run", action="store_true", help="只出预览图，不改文件")
    ap.add_argument("--preview", default="_cut_fix_preview.png", help="预览图路径")
    args = ap.parse_args()

    png = os.path.join(args.parts, args.name + ".png")
    if not os.path.isfile(png):
        print(f"[ERROR] 找不到 {png}")
        return 1

    img = load_rgba(png)
    alpha = img[:, :, 3] > 20
    cut = find_cut_row(alpha, args.min_ratio)
    if cut is None:
        print("[SKIP] 没有检测到底边直切，无需处理")
        return 0
    if cut >= img.shape[0] - 3:
        print(f"[INFO] 切口行 y={cut}/{img.shape[0] - 1}，按此收口")

    depths = [float(v) for v in args.curve.split(",") if v.strip() != ""]
    out, extra = close_bottom(img, cut, depths, args.dark)

    # 预览：原图 / 新图并排，放大 6 倍
    s = 6
    def to_img(a):
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        bg = Image.new("RGBA", im.size, (250, 250, 250, 255))
        bg.alpha_composite(im)
        return bg.resize((im.width * s, im.height * s), Image.NEAREST)

    a_im, b_im = to_img(img), to_img(out)
    sheet = Image.new("RGBA", (a_im.width + b_im.width + 30,
                               max(a_im.height, b_im.height) + 20), (235, 235, 235, 255))
    sheet.alpha_composite(a_im, (10, 10))
    sheet.alpha_composite(b_im, (a_im.width + 20, 10))
    sheet.save(args.preview)
    print(f"[OK] 切口 y={cut}，向下补 {extra} 行，尺寸 "
          f"{img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]}")
    print(f"[OK] 预览 {args.preview}（左原图 / 右补全）")

    if args.dry_run:
        print("[DRY-RUN] 未写入")
        return 0

    bak = png + ".orig"
    if not os.path.exists(bak):
        shutil.copy2(png, bak)
        print(f"[OK] 备份 {bak}")
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(png)
    print(f"[OK] 已写入 {png}")
    for p in update_coords(args.parts, args.name, out.shape[0]):
        print(f"[OK] 更新 height -> {out.shape[0]}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
