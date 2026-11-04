#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_preview.py — 拆图结果预览图。

拆完图先出一张"体检单": 左边是部件按坐标合起来的成品(棋盘格背景),
每个部件带编号框; 右边是部件缩略图 + 名字 + 程序判定出的角色。
用来确认"拆图是否正确", 正确了再去生成待机/行走/受击/死亡。

用法:
    python split_preview.py --parts output/xxx/parts \
        --coords output/xxx/coords.json --out output/xxx/_split_preview.png
"""
import argparse
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

BG = (30, 31, 35)
FG = (236, 238, 242)
DIM = (150, 156, 166)
ACCENT = (73, 199, 255)
WARN = (255, 176, 82)
CHK1 = (58, 60, 66)          # 棋盘格亮格
CHK2 = (43, 45, 50)          # 棋盘格暗格

VMAX = 880                   # 合成图最大边长
CELL = 172                   # 缩略图格子边长
PAD = 8
GCOLS = 3                    # 缩略图列数
M = 24                       # 页边距
HEAD = 78
FOOT = 40

_ROLE_CN = {
    "hair": "头发(风吹)", "ribbon": "飘带/裙摆(风吹)", "eye": "眼睛(眨眼)",
    "blink_eye": "闭眼图", "brow": "眉毛", "lash": "睫毛",
    "arm_upper": "大臂", "arm_fore": "小臂", "arm_hand": "手",
    "face": "脸部/五官", "body": "躯干/衣服", "bird": "小鸟(隐藏)",
    "headwear_rigid": "头巾/头盔(随头)", "headgear": "头巾/头盔(随头)",
    "cloth": "头巾(下摆可飘)", "prop": "道具(跟手)", "leg": "腿(髋/膝/踝)",
    "eye_closed": "闭眼补丁(死亡用)",
}

# 这些名字即使被写成 cls=face(无头发模式), 也要在体检单上标成"头巾/头盔"
_HAIR_NAME_KW = ("hair", "headwear", "toushi", "maozi", "hat", "headgear")

try:                          # 复用生成器的语义分类, 保证"预览判定"就是"实际挂骨"
    from gen_idle2 import classify as _classify
except Exception:             # noqa: BLE001  — 缺依赖时预览仍要能出
    _classify = None


def _font(size):
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
              r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _role(part):
    """生效的角色: coords 里的 cls 优先(生成器就是这么用的), 否则按名字分类。"""
    cls = part.get("cls")
    name = (part.get("name") or "").lower()
    if cls == "headgear" or (cls == "face" and any(k in name for k in _HAIR_NAME_KW)):
        return "headgear"
    if cls:
        return cls
    if _classify is None:
        return "?"
    try:
        return _classify(part.get("name", ""), bool(part.get("bird")))
    except Exception:         # noqa: BLE001
        return "?"


def _checker(w, h, step=16):
    """深色棋盘格, 用来看清透明区域与部件边缘。"""
    img = Image.new("RGB", (max(1, w), max(1, h)), CHK2)
    d = ImageDraw.Draw(img)
    for yy in range(0, h, step):
        for xx in range(0, w, step):
            if ((xx // step) + (yy // step)) % 2 == 0:
                d.rectangle([xx, yy, xx + step - 1, yy + step - 1], fill=CHK1)
    return img


def _paste_rgba(base, im, x, y):
    """带裁剪的 alpha 合成(部件可以超出画布)。"""
    x0, y0 = max(0, x), max(0, y)
    x1 = min(base.width, x + im.width)
    y1 = min(base.height, y + im.height)
    if x1 <= x0 or y1 <= y0:
        return
    base.alpha_composite(im.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))


def _fit(w, h, box):
    s = min(box / float(w), box / float(h))
    return max(1, round(w * s)), max(1, round(h * s))


def _cut(s, n):
    return s if len(s) <= n else s[:n - 1] + "…"


def make_preview(parts_dir, coords_path, out_png, log=print):
    """生成预览图, 返回 {n, roles:{}, missing:[], out:path}。"""
    coords = json.load(open(coords_path, encoding="utf-8"))
    canvas = coords.get("canvas") or {}
    cw = int(canvas.get("w") or 1024)
    ch = int(canvas.get("h") or 1024)
    parts = coords.get("parts") or []
    if not parts:
        raise RuntimeError("coords.json 里没有部件")

    # ---- 合成原图空间 ----
    comp = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    missing = []
    roles = {}
    for p in parts:
        path = os.path.join(parts_dir, p.get("image", ""))
        if not p.get("image") or not os.path.isfile(path):
            missing.append(p.get("name", "?"))
            continue
        im = Image.open(path).convert("RGBA")
        if im.size != (int(p.get("width", im.width)), int(p.get("height", im.height))):
            im = im.resize((int(p["width"]), int(p["height"])), Image.LANCZOS)
        # 闭眼补丁是"死亡才显示"的贴片, 平时在生成器里是隐藏的。体检图上不能合成
        # 上去 —— 否则脸上永远挂着两道锯齿黑线, 看着像五官拆错了。
        if str(p.get("cls")) != "eye_closed":
            _paste_rgba(comp, im, int(round(p.get("x", 0))), int(round(p.get("y", 0))))
        r = _role(p)
        roles[r] = roles.get(r, 0) + 1

    # ---- 版式 ----
    n = len(parts)
    cols = 3 if n <= 9 else (4 if n <= 20 else 5)
    cell = CELL if n <= 20 else 150
    scale = min(VMAX / float(cw), VMAX / float(ch), 2.0)
    dw, dh = max(1, round(cw * scale)), max(1, round(ch * scale))
    rows = max(1, math.ceil(n / float(cols)))
    cell_h = cell + 42
    gall_w = cols * cell + (cols - 1) * PAD
    body_h = max(dh, rows * cell_h)
    comp_y = HEAD + (body_h - dh) // 2      # 合成图垂直居中, 不留大片空白
    W = M + dw + 26 + gall_w + M
    H = HEAD + body_h + FOOT + M

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_title, f_sub = _font(25), _font(15)
    f_name, f_meta, f_num = _font(15), _font(13), _font(12)

    # ---- 标题 ----
    d.text((M, 20), f"拆图结果 · {len(parts)} 个部件", font=f_title, fill=FG)
    d.text((M, 52), f"画布 {cw}×{ch}   ·   生成动作前请先确认：部件是否独立、"
                    f"手臂是否分段、有无该有的部件缺失",
           font=f_sub, fill=DIM)

    # ---- 左: 合成图 ----
    view = _checker(dw, dh).convert("RGBA")
    view.alpha_composite(comp.resize((dw, dh), Image.LANCZOS), (0, 0))
    img.paste(view.convert("RGB"), (M, comp_y))
    d.rectangle([M, comp_y, M + dw - 1, comp_y + dh - 1], outline=(86, 90, 98))

    # 部件框 + 编号
    for i, p in enumerate(parts, 1):
        if p.get("name") in missing:
            continue
        x0 = M + p.get("x", 0) * scale
        y0 = comp_y + p.get("y", 0) * scale
        x1 = x0 + p.get("width", 0) * scale
        y1 = y0 + p.get("height", 0) * scale
        x0, y0 = max(M, x0), max(comp_y, y0)
        x1 = min(M + dw, x1)
        y1 = min(comp_y + dh, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        d.rectangle([x0, y0, x1, y1], outline=ACCENT)
        tag = str(i)
        tw = d.textlength(tag, font=f_num)
        bx, by = x0, y0
        d.rectangle([bx, by, bx + tw + 8, by + 16], fill=ACCENT)
        d.text((bx + 4, by + 1), tag, font=f_num, fill=(12, 18, 22))

    # ---- 右: 部件清单 ----
    gx0 = M + dw + 26
    for i, p in enumerate(parts, 1):
        cx = gx0 + ((i - 1) % cols) * (cell + PAD)
        cy = HEAD + ((i - 1) // cols) * cell_h
        name = p.get("name", "?")
        d.rectangle([cx, cy, cx + cell, cy + cell], fill=CHK2, outline=(72, 76, 84))
        path = os.path.join(parts_dir, p.get("image", ""))
        if os.path.isfile(path):
            im = Image.open(path).convert("RGBA")
            tw_, th_ = _fit(im.width, im.height, cell - 8)
            thumb = Image.new("RGB", (tw_, th_), CHK2)
            thumb.paste(im.resize((tw_, th_), Image.LANCZOS),
                        (0, 0), im.resize((tw_, th_), Image.LANCZOS))
            img.paste(thumb, (cx + (cell - tw_) // 2, cy + (cell - th_) // 2))
        role = _role(p)
        col = WARN if role in ("body", "?") and "body" not in name.lower() else FG
        d.text((cx, cy + cell + 4), _cut(f"{i}. {name}", 22), font=f_name, fill=col)
        d.text((cx, cy + cell + 21),
               f"{_ROLE_CN.get(role, role)} · {int(p.get('width', 0))}×{int(p.get('height', 0))}",
               font=f_meta, fill=DIM)
        if p.get("name") in missing:
            d.text((cx + 4, cy + 4), "缺图", font=f_num, fill=(255, 92, 92))

    # ---- 页脚 ----
    dist = "  ".join(f"{_ROLE_CN.get(k, k)}×{v}" for k, v in
                     sorted(roles.items(), key=lambda t: -t[1]))
    d.text((M, HEAD + body_h + 12), "角色: " + dist, font=f_meta, fill=DIM)
    if missing:
        d.text((M, HEAD + body_h + 12),
               "  缺少图片: " + "、".join(missing), font=f_meta, fill=(255, 92, 92))

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    img.save(out_png)

    log(f"拆图预览: {out_png}  ({len(parts)} 部件, {cw}×{ch})")
    for i, p in enumerate(parts, 1):
        log(f"  #{i:<2} {p.get('name', '?'):<18} {_role(p):<10} "
            f"{int(p.get('width', 0))}×{int(p.get('height', 0))} "
            f"@({int(p.get('x', 0))},{int(p.get('y', 0))})")
    log("角色: " + dist)
    if missing:
        log("缺少图片: " + "、".join(missing))
    return {"n": len(parts), "roles": roles, "missing": missing, "out": out_png}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--coords", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.coords)),
                                "_split_preview.png")
    make_preview(a.parts, a.coords, out)


if __name__ == "__main__":
    main()
