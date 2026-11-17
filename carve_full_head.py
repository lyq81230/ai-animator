#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把散落在躯干/手臂部件里的整颗头并回 head, 并从身体层挖掉。

适用: see-through 把头盔/帽檐/金发连同脸一起判给了 topwear(有时 arm_upper_L
      里还有一份副本), 静止合成看不出问题(head 盖在上面), 但头一转,
      躯干上就留下一顶盔。

做法: 用一条按列变化的切割线 y_cut(x) 定义"头部区域"——中间(脸/下巴)切到
      下巴下方, 两侧(肩甲上方)提前收口, 避免把肩甲一起带走; 落在区域内的
      身体像素并回 head(原 head 已有的位置不覆盖, 保持脸的颜色), 其余从
      身体部件里清零。颜色一律取自部件本身, 不重新画。

用法:
    python carve_full_head.py <角色目录> [--curve 60,0,75,85,...]
                              [--parts topwear,arm_upper_L,arm_upper_R]
                              [--no-yellow] [--apply] [--preview 对比图.png]

默认只预览不落盘; 确认预览后加 --apply 才正式写入。
被改的 png 备份到 <角色目录>/_carve_backup/, 并同步 parts/ 与上层的
两份 coords.json。

拆图自动生效: 在 <角色目录>/carve_head.json(parts 的上级, 重新拆图不会删)
或 config/<角色名>/carve_head.json(随仓库走, output 被清空也还在) 放:
    {"enabled": true, "curve": [...], "parts": [...]}
build_from_seethrough.py 收尾会读它并调用 apply(), 这样重新拆图不会把修正冲掉。
"""
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image
import cv2

# (x, 该列的头部下边界 y)。两端给 0 表示不切, 中间切到下巴(y=117)。
DEFAULT_CURVE = [60, 0, 75, 85, 95, 96, 108, 117, 175, 117,
                 188, 100, 210, 84, 230, 0]
DEFAULT_PARTS = ("topwear", "arm_upper_L", "arm_upper_R")
KEEP = {"head", "headwear", "headscarf", "eyewhite", "irides",
        "eyes_closed", "nose", "mouth", "brows", "lashes",
        "topwear_overlay", "objects"}


def read_parts(D):
    parts_dir = os.path.join(D, "parts")
    cp = os.path.join(parts_dir, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    W, H = co["canvas"]["w"], co["canvas"]["h"]
    canv = {}
    for p in co["parts"]:
        f = os.path.join(parts_dir, p.get("image", p["name"] + ".png"))
        a = np.array(Image.open(f).convert("RGBA"))
        c = np.zeros((H, W, 4), np.uint8)
        x, y = int(p["x"]), int(p["y"])
        h, w = a.shape[:2]
        yy0, yy1 = max(0, y), min(H, y + h)
        xx0, xx1 = max(0, x), min(W, x + w)
        if yy1 > yy0 and xx1 > xx0:
            c[yy0:yy1, xx0:xx1] = a[yy0 - y:yy1 - y, xx0 - x:xx1 - x]
        canv[p["name"]] = c
    return parts_dir, cp, co, W, H, canv


def zone(H, W, curve):
    pts = np.array(curve, float).reshape(-1, 2)
    xs = np.arange(W, dtype=float)
    yc = np.interp(xs, pts[:, 0], pts[:, 1], left=0, right=0)
    yy = np.arange(H)[:, None]
    return np.broadcast_to(yy < yc[None, :], (H, W)).copy()


def head_mask(c, Z, H, W, yellow_extra=True):
    """该部件里属于头的像素 = 切割线以上 + (可选)头部高度内的黄布/金发。

    单靠切割线会在两侧肩甲上方留几撮金发(切割线为避开肩甲提前收了口)。
    这些黄的像素按颜色补挖, 灰色肩甲不受影响。
    """
    al = c[..., 3] > 20
    m = al & Z
    if yellow_extra:
        hsv = cv2.cvtColor(c[..., :3], cv2.COLOR_RGB2HSV)
        yel = ((hsv[..., 0] >= 16) & (hsv[..., 0] <= 45)
               & (hsv[..., 1] > 70) & (hsv[..., 2] > 80))
        yy = np.arange(H)[:, None]
        xx = np.arange(W)[None, :]
        band = (yy < 116) & ((xx < 115) | (xx > 175))
        m |= al & yel & band
    return m


def crop_write(parts_dir, p, c, co, bak_dir=None):
    """把画布 RGBA 裁回部件窗口, 落盘并更新 coords 记录。"""
    al = c[..., 3] > 20
    if not al.any():
        return False
    ys, xs = np.nonzero(al)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    sub = c[y0:y1, x0:x1]
    f = os.path.join(parts_dir, p.get("image", p["name"] + ".png"))
    bak_dir = bak_dir or parts_dir
    os.makedirs(bak_dir, exist_ok=True)
    bak = os.path.join(bak_dir, os.path.basename(f))
    if not os.path.isfile(bak) and os.path.isfile(f):
        shutil.copyfile(f, bak)
    Image.fromarray(sub).save(f)
    old = (p["x"], p["y"], p["width"], p["height"])
    p["x"], p["y"], p["width"], p["height"] = x0, y0, x1 - x0, y1 - y0
    new = (p["x"], p["y"], p["width"], p["height"])
    if old != new:
        print(f"    {p['name']}: {old} -> {new}")
    return True


def _migrate_old_backups(parts_dir, bak_dir):
    """把老版本留在 parts/ 里的 *.png.bak_head 挪到 _carve_backup/。"""
    for _f in sorted(os.listdir(parts_dir)):
        if not _f.endswith(".bak_head"):
            continue
        src = os.path.join(parts_dir, _f)
        dst = os.path.join(bak_dir, _f[:-len(".bak_head")])
        if not os.path.isfile(dst):
            shutil.move(src, dst)
        else:
            os.remove(src)


def apply(D, curve=None, targets=None, yellow_band=True,
          preview=None, dry=False):
    """执行一次头部归位。dry=True 只算不写盘(命令行入口默认如此)。

    返回本次从身体层挖掉的像素总数。重跑已修好的目录会返回 0, 不会破坏结果。
    """
    curve = list(curve) if curve else list(DEFAULT_CURVE)
    targets = list(targets) if targets else list(DEFAULT_PARTS)
    parts_dir, cp, co, W, H, canv = read_parts(D)
    Z = zone(H, W, curve)
    pmap = {p["name"]: p for p in co["parts"]}
    if "head" not in pmap:
        print("没有 head 部件, 跳过")
        return 0

    # 1) 挖掉身体部件里落在头部区域的像素。
    #    taken 只用来决定"这一格的颜色由谁并回 head"(避免重复叠加),
    #    但每个身体部件自己那一层都必须清零 —— 重叠的副本留着就是拖影。
    taken = np.zeros((H, W), bool)
    outs = {}
    for n in targets:
        if n not in canv or n in KEEP:
            continue
        c = canv[n].copy()
        clear = head_mask(c, Z, H, W, yellow_extra=yellow_band)
        m = clear & ~taken          # 该部件负责并回 head 的那部分
        taken |= clear
        c[clear] = 0                # 整块挖掉, 含与别人重叠的副本
        outs[n] = (c, m)
        print(f"  {n}: 挖掉 {int(clear.sum())}px (其中 {int(m.sum())}px 并回 head)")

    # 2) head 已有像素优先保留(脸的颜色不能变), 只补空缺
    head = canv["head"].copy()
    hm = head[..., 3] > 20
    for n, (_c, m) in outs.items():
        sel = m & ~hm
        head[sel] = canv[n][sel]     # 颜色取原部件
        hm |= sel
    added = int((taken & (canv["head"][..., 3] <= 20)).sum())
    print(f"  head: 补入 {added}px, 共 {int((head[..., 3] > 20).sum())}px")

    if not dry:
        bak_dir = os.path.join(D, "_carve_backup")
        os.makedirs(bak_dir, exist_ok=True)
        _migrate_old_backups(parts_dir, bak_dir)
        for n, (c, _m) in outs.items():
            crop_write(parts_dir, pmap[n], c, co, bak_dir)
        crop_write(parts_dir, pmap["head"], head, co, bak_dir)
        for path in (cp, os.path.join(D, "coords.json")):
            if os.path.isfile(path):
                json.dump(co, open(path, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2)
        print("  已落盘, coords 两份已同步")
    else:
        print("  未指定 --apply: 仅预览, 未修改任何文件")

    if preview:
        build_preview(co, W, H, canv, outs, head, preview)
    return int(taken.sum())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 1
    D = args[0]
    flags = sys.argv[1:]
    # 破坏性脚本: 默认只预览, 必须显式 --apply 才写盘
    dry = "--apply" not in flags
    curve = targets = None
    yellow_band = True
    preview = None
    for i, f in enumerate(flags):
        if f.startswith("--curve="):
            curve = [float(v) for v in f.split("=", 1)[1].split(",")]
        elif f.startswith("--parts="):
            targets = [v.strip() for v in f.split("=", 1)[1].split(",")]
        elif f == "--no-yellow":
            yellow_band = False
        elif f == "--preview" and i + 1 < len(flags):
            preview = flags[i + 1]
    apply(D, curve, targets, yellow_band, preview, dry)
    return 0


def build_preview(co, W, H, canv, outs, head, path):
    """左边=原合成, 中=新合成, 右=挖完的 topwear 单图。"""
    from PIL import ImageDraw
    order = [p["name"] for p in co["parts"]]

    def composite(use_new):
        cv = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        for n in order:
            c = canv[n]
            if use_new:
                if n == "head":
                    c = head
                elif n in outs:
                    c = outs[n][0]
            cv.alpha_composite(Image.fromarray(c), (0, 0))
        return cv

    a, b = composite(False), composite(True)
    tw = outs.get("topwear", (canv.get("topwear"), None))[0]
    if tw is None:
        tw = np.zeros((H, W, 4), np.uint8)
    twbg = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    twbg.alpha_composite(Image.fromarray(tw))
    s = 2
    box = (55, 10, 240, 215)
    tiles = [("before", a), ("after", b), ("topwear only", twbg)]
    tw_, th_ = (box[2] - box[0]) * s, (box[3] - box[1]) * s
    sheet = Image.new("RGBA", (tw_ * 3 + 80, th_ + 45), (235, 235, 235, 255))
    dr = ImageDraw.Draw(sheet)
    for i, (tag, im) in enumerate(tiles):
        c = im.crop(box).resize((tw_, th_), Image.LANCZOS)
        x = 20 + i * (tw_ + 20)
        sheet.alpha_composite(c, (x, 35))
        dr.text((x, 12), tag, fill=(0, 0, 0))
    sheet.save(path)
    print(f"  预览 -> {path}")


if __name__ == "__main__":
    sys.exit(main())
