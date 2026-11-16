#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把扇子(扇子)周边散落在其它部件里的三缕紫色烟雾合并进唯一的 objects(扇子) 部件。

做法:
 1) 从原图重建烟雾掩码(暗紫/藕色: G 为最小通道, R/B 相近且明显高于 G)。
 2) 重建 objects.png = 扇子本体(原 objects 内容) + 全部烟雾, 画布扩展到能容纳所有烟雾。
 3) 从其它所有部件里摘掉烟雾像素(置透明), 并重裁 bbox、更新 coords。
 4) 合成校验: 新 parts 合成图与原图在烟雾覆盖区一致。
"""
import os
import json
import cv2
import numpy as np
from PIL import Image

PARTS_DIR = "output/zombie_fan_hero_338/parts"
SRC = ""      # 留空 -> 按 PARTS_DIR 的角色名去 E:/pic 找源图


def smoke_mask(src):
    """返回布尔烟雾掩码 (H,W)。"""
    a0 = src[..., 3] > 0
    R, G, B = src[..., 0].astype(int), src[..., 1].astype(int), src[..., 2].astype(int)
    # 暗紫/藕色烟雾: G 最小, R/B 都明显高于 G, 且 B≈R(容差内)
    sm = a0 & (B >= G + 6) & (R >= G + 6) & (B >= R - 18) & (B <= R + 18) & (B >= 35)
    sm = cv2.dilate(sm.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & a0
    return sm


def alpha_bbox(a, thresh=110):
    ys, xs = np.where(a > thresh)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def main(parts_dir=None, src_path=None):
    global PARTS_DIR, SRC
    if parts_dir:
        PARTS_DIR = parts_dir
    if src_path:
        SRC = src_path
    if not src_path or not os.path.isfile(src_path):
        # 按角色名去 E:/pic 找源图(parts 的上一级就是角色名)
        name = os.path.basename(os.path.normpath(os.path.join(PARTS_DIR, "..")))
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            cand = os.path.join("E:/pic", name + ext)
            if os.path.isfile(cand):
                src_path = cand
                break
    if not src_path or not os.path.isfile(src_path):
        raise SystemExit(f"找不到源图: {src_path}")
    src = np.asarray(Image.open(src_path).convert("RGBA"))
    H, W, _ = src.shape
    sm = smoke_mask(src)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sm.astype(np.uint8), 8)
    comps = [(int(st[k, cv2.CC_STAT_AREA]),
              int(st[k, cv2.CC_STAT_LEFT]), int(st[k, cv2.CC_STAT_TOP]),
              int(st[k, cv2.CC_STAT_WIDTH]), int(st[k, cv2.CC_STAT_HEIGHT]))
             for k in range(1, n) if int(st[k, cv2.CC_STAT_AREA]) >= 15]
    comps.sort(reverse=True)
    print("烟雾连通块(>=15px):", len(comps), " 烟雾像素", int(sm.sum()))
    for c in comps:
        print("  area=%d bbox=(%d,%d,%dx%d)" % (c[0], c[1], c[2], c[3], c[4]))

    coords = json.load(open(os.path.join(PARTS_DIR, "coords.json"), encoding="utf-8"))
    parts = coords["parts"]
    obj = next(p for p in parts if p["name"] == "objects")

    # 扇子本体掩码 = 当前 objects.png 的 alpha(映射到全局)
    ofp = os.path.join(PARTS_DIR, obj["image"])
    oim = np.asarray(Image.open(ofp).convert("RGBA"))
    oy0, ox0 = obj["y"], obj["x"]
    fan = np.zeros((H, W), bool)
    fan[oy0:oy0 + oim.shape[0], ox0:ox0 + oim.shape[1]][:oim.shape[0], :oim.shape[1]] = (oim[..., 3] > 8)

    group = fan | sm  # 扇子 + 全部烟雾
    ys, xs = np.where(group)
    ny0, ny1 = int(ys.min()), int(ys.max() + 1)
    nx0, nx1 = int(xs.min()), int(xs.max() + 1)
    nh, nw = ny1 - ny0, nx1 - nx0

    # 重建 objects: 取原图在 group 掩码下的像素
    new = np.zeros((nh, nw, 4), np.uint8)
    sub_src = src[ny0:ny1, nx0:nx1]
    sub_grp = group[ny0:ny1, nx0:nx1]
    new[sub_grp] = sub_src[sub_grp]
    Image.fromarray(new).save(ofp)
    obj.update({"x": nx0, "y": ny0, "width": nw, "height": nh})
    print("objects 重建: %dx%d @(%d,%d), 含烟雾像素 %d" %
          (nw, nh, nx0, ny0, int((sm[ny0:ny1, nx0:nx1] & (new[..., 3] > 0)).sum())))

    # 从其它部件摘烟雾
    for p in parts:
        if p is obj:
            continue
        fp = os.path.join(PARTS_DIR, p["image"])
        if not os.path.isfile(fp):
            continue
        im = np.array(Image.open(fp).convert("RGBA"))
        ph, pw = im.shape[:2]
        y0, x0 = p["y"], p["x"]
        y1, x1 = min(H, y0 + ph), min(W, x0 + pw)
        local_sm = sm[y0:y1, x0:x1][:ph, :pw]
        pa = im[..., 3] > 8
        rm = pa & local_sm
        if int(rm.sum()) == 0:
            continue
        im[rm] = 0
        bb = alpha_bbox(im[..., 3])
        if bb is None or (bb[2] - bb[0]) * (bb[3] - bb[1]) < 50:
            # 整片清空 -> 删除部件
            os.remove(fp)
            p["_delete"] = True
            print(" %s: 烟雾占满, 删除" % p["name"])
            continue
        im = im[bb[1]:bb[3], bb[0]:bb[2]]
        Image.fromarray(im).save(fp)
        p.update({"x": int(x0 + bb[0]), "y": int(y0 + bb[1]),
                  "width": int(im.shape[1]), "height": int(im.shape[0])})
        print(" %s: 摘掉 %d 烟雾像素, 裁到 %dx%d @(%d,%d)" %
              (p["name"], int(rm.sum()), im.shape[1], im.shape[0], p["x"], p["y"]))

    # 清理删除标记
    coords["parts"] = [p for p in parts if not p.get("_delete")]
    json.dump(coords, open(os.path.join(PARTS_DIR, "coords.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("coords 更新完成, 剩余部件数:", len(coords["parts"]))

    # 合成校验
    comp = np.zeros((H, W, 4), np.uint8)
    for p in coords["parts"]:
        fp = os.path.join(PARTS_DIR, p["image"])
        if not os.path.isfile(fp):
            continue
        im = np.asarray(Image.open(fp).convert("RGBA"))
        y0, x0 = p["y"], p["x"]
        y1, x1 = min(H, y0 + im.shape[0]), min(W, x0 + im.shape[1])
        sub = comp[y0:y1, x0:x1][:im.shape[0], :im.shape[1]]
        ov = im[..., 3] > 0
        sub[ov] = im[:y1 - y0, :x1 - x0][ov]
    diff = (np.any(src[..., :3] != comp[..., :3], axis=2)) & (src[..., 3] > 0) & (comp[..., 3] > 0)
    print("合成差异像素(应为边缘/透明差异):", int(diff.sum()))
    Image.fromarray(comp).save("_composite_fan.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("parts_dir", nargs="?", default=PARTS_DIR)
    ap.add_argument("--src", default=SRC)
    _a = ap.parse_args()
    main(_a.parts_dir, _a.src)
