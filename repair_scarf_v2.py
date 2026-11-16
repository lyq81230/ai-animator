# -*- coding: utf-8 -*-
"""把 zombie_fan_hero_338 的黄色头巾合并成【唯一一个整体部件】headwear_overlay。

相对上一版的关键改进:
  - 头巾掩码用「亮黄种子 + 洪水填充」, 把整条头巾(包头布 + 左右垂布 +
    额头带 + 发髻结 + 流苏 + 阴影/黑描边)连成一张完整连通掩码, 不再漏掉
    头部(尤其左侧垂布贴着脸的那一缕);
  - 其余部件(尤其 head)把属于头巾的像素彻底摘掉 -> 反向 inpaint 补成邻近
    肤色/衣色(保持不透明, 不露洞), 并验证摘干净(残余为 0);
  - 重建 headwear_overlay = 原图[完整掩码], 单部件覆盖全部头巾区域。

用法:
    python repair_scarf_v2.py <parts目录> [--src <原图.png>] [--dry]
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np
from PIL import Image


def build_scarf_mask(src):
    """返回布尔头巾掩码 (H,W): 亮黄种子 + 仅沿【黄色核心】洪水填充。

    这样既能把整条头巾(包头布 + 左右垂布 + 额头带 + 发髻结 + 流苏, 含阴影黄)
    连成完整掩码, 又不会越界吞掉棕袍/武器(棕 H<14)或绿皮肤(H>50)——因为它们
    不属于黄核心, 且与头巾不连通。
    """
    H, W, _ = src.shape
    a0 = src[..., 3] > 0
    hsv = cv2.cvtColor(src[..., :3], cv2.COLOR_RGB2HSV)
    Hh, Ss, Vv = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)

    # 种子: 明显是头巾的亮黄布
    narrow = a0 & (Hh >= 15) & (Hh <= 47) & (Ss >= 80) & (Vv >= 120)
    n, lab, st, _ = cv2.connectedComponentsWithStats(narrow.astype(np.uint8), 8)
    seed = np.zeros_like(narrow)
    for k in range(1, n):
        if int(st[k, cv2.CC_STAT_AREA]) >= 150:
            seed |= (lab == k)

    # 黄色核心生长区(仅黄, 排除棕 H<14 与绿皮肤 H>50): 把阴影黄 + 近白高光垂布也连进来
    core = a0 & (Hh >= 14) & (Hh <= 50) & (Ss >= 30) & (Vv >= 25)
    flood = seed.copy()
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(80):
        d = cv2.dilate(flood.astype(np.uint8), k3).astype(bool)
        nw = flood | (d & core)
        if nw.sum() == flood.sum():
            break
        flood = nw

    # 收 1-2px 黑描边/暗影边缘, 再闭运算填小缝
    ring = cv2.dilate(flood.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool) & a0 & (Vv < 120) & (Ss < 60) & ~flood
    scarf = flood | ring
    scarf = cv2.morphologyEx(scarf.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((3, 3), np.uint8)).astype(bool)
    # 去掉 < 120px 的小噪点(确保只剩头巾本体)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(scarf.astype(np.uint8), 8)
    clean = np.zeros_like(scarf)
    for k in range(1, n2):
        if int(st2[k, cv2.CC_STAT_AREA]) >= 120:
            clean |= (lab2 == k)
    # 填补头巾内部 < 40px 的小洞(闭合垂布上的 1px 缝隙, 不影响外部独立部件)
    inv = (~clean).astype(np.uint8) * 255
    n3, lab3, st3, _ = cv2.connectedComponentsWithStats(inv, 8)
    for k in range(1, n3):
        if int(st3[k, cv2.CC_STAT_AREA]) <= 40:
            clean |= (lab3 == k)
    # 收 2px 黄边: 把垂布边缘的黄色抗锯齿像素也并入(棕袍/绿皮已被 H 范围排除)
    fringe = cv2.dilate(clean.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool) & core & ~clean
    clean = clean | fringe
    return clean


def _alpha_bbox(a, thresh=110):
    ys, xs = np.where(a > thresh)
    if len(xs) == 0:
        ys, xs = np.where(a > 8)
        if len(xs) == 0:
            return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _remove_and_inpaint(im, sub_mask, fill="inpaint"):
    """im: (ph,pw,4); sub_mask: (ph,pw) 头巾掩码(已对齐窗口)。
    返回 (new_im, deleted_bool, offx, offy)。
    fill='inpaint': 反向 inpaint(适合小洞); fill='skinmed': 用部件自身肤色中位数填实
    (适合大块头巾洞, 避免 inpaint 从残余黄色采样而留黄)。"""
    pa = im[..., 3] > 8
    hole = pa & sub_mask
    if int(hole.sum()) == 0:
        return im, False, 0, 0
    solid = pa & ~hole
    dil = cv2.dilate(solid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    interior = hole & dil
    boundary = hole & ~dil
    new = im.copy()
    if fill == "skinmed":
        # 用部件非黄区域的肤色中位数填实(无黄、实心)
        ih = cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2HSV)
        Hh_, Ss_, Vv_ = ih[..., 0].astype(int), ih[..., 1].astype(int), ih[..., 2].astype(int)
        skin = pa & ~sub_mask & ~((Hh_ >= 14) & (Hh_ <= 50) & (Ss_ >= 25) & (Vv_ >= 25))
        if int(skin.sum()) == 0:
            skin = pa & ~sub_mask
        med = np.median(im[skin][..., :3], axis=0).astype(np.uint8)
        new[..., :3][hole] = med
        new[..., 3][hole] = 255
    else:
        rgb = im[..., :3].astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _hh = int(hole.sum())
        _rad = int(min(28, max(5, round((_hh ** 0.5) / 2))))
        inp = cv2.inpaint(rgb_bgr, (hole.astype(np.uint8) * 255), _rad, cv2.INPAINT_TELEA)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        new[..., :3][interior] = inp[interior]
        new[..., 3][interior] = 255
        new[..., 3][boundary] = 0
    bb = _alpha_bbox(new[..., 3])
    if bb is None or (bb[2] - bb[0]) * (bb[3] - bb[1]) < 50:
        return new, True, 0, 0
    new = new[bb[1]:bb[3], bb[0]:bb[2]]
    return new, False, bb[0], bb[1]


def main(parts_dir, src_path=None, dry=False):
    cpath = os.path.join(parts_dir, "coords.json")
    coords = json.load(open(cpath, encoding="utf-8"))
    W = int(coords["canvas"]["w"]); H = int(coords["canvas"]["h"])
    if src_path is None or not os.path.isfile(src_path):
        # 按角色名去 E:/pic 找源图(parts_dir 的上一级就是角色名)
        name = os.path.basename(os.path.normpath(os.path.join(parts_dir, "..")))
        for cand in (name, "fashu_hero_338"):
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                p = os.path.join("E:/pic", cand + ext)
                if os.path.isfile(p):
                    src_path = p
                    break
            if src_path and os.path.isfile(src_path):
                break
    if not src_path or not os.path.isfile(src_path):
        raise SystemExit(f"找不到源图: {src_path}")
    src = np.asarray(Image.open(src_path).convert("RGBA"))

    mask = build_scarf_mask(src)
    ys, xs = np.where(mask)
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    print(f"头巾掩码: {int(mask.sum())}px, bbox=({bx0},{by0},{bx1-bx0}x{by1-by0})")

    # 1) 重建唯一头巾部件(完整)
    scarf_img = src[by0:by1, bx0:bx1].copy()
    scarf_img[..., 3] = np.where(mask[by0:by1, bx0:bx1], scarf_img[..., 3], 0)
    owner = next((p for p in coords["parts"] if p["name"] == "headwear_overlay"), None)
    if owner is None:
        owner = {"name": "headwear_overlay", "image": "headwear_overlay.png"}
        coords["parts"].append(owner)
    owner.update({
        "x": bx0, "y": by0, "width": bx1 - bx0, "height": by1 - by0,
        "cls": "cloth", "grid": [6, 6],
    })
    owner.pop("follow_part", None)
    owner.pop("cls0", None)
    print(f"  headwear_overlay -> {bx1-bx0}x{by1-by0} @({bx0},{by0}) cls=cloth")

    # 2) 其余部件摘头巾
    to_delete = []
    for p in coords["parts"]:
        if p is owner:
            continue
        fp = os.path.join(parts_dir, p["image"])
        if not os.path.isfile(fp):
            to_delete.append(p)
            print(f"  {p['name']}: 文件缺失, 删除引用"); continue
        im = np.asarray(Image.open(fp).convert("RGBA"))
        ph, pw = im.shape[:2]
        wy0, wx0 = p["y"], p["x"]
        wy1, wx1 = min(H, wy0 + ph), min(W, wx0 + pw)
        # head 单独用「宽黄」摘除以彻底清空(头部窗口只有头巾+绿皮, 无袍/带),
        # 其余部件用保守的洪水头巾掩码(避免误删棕袍腰带黄)
        if p["name"] == "head":
            ih2 = cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2HSV)
            H2, S2, V2 = ih2[..., 0].astype(int), ih2[..., 1].astype(int), ih2[..., 2].astype(int)
            ysub = (im[..., 3] > 8) & (H2 >= 14) & (H2 <= 50) & (S2 >= 30) & (V2 >= 25)
            # 膨胀 5px: 把整条垂布(含淡边/抗锯齿)整片摘掉, 使洞边界落在绿皮上,
            # 反向 inpaint 才能填成肤色而非黄色
            sub_mask = cv2.dilate(ysub.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & (im[..., 3] > 8)
        else:
            sub_mask = mask[wy0:wy1, wx0:wx1][:ph, :pw]
        pa = im[..., 3] > 8
        hole = int((pa & sub_mask).sum())
        if hole == 0:
            continue
        print(f"  {p['name']}: 摘头巾 {hole}px", end="")
        if dry:
            print("  [dry]"); continue
        new, deleted, offx, offy = _remove_and_inpaint(im, sub_mask, fill="skinmed" if p["name"] == "head" else "inpaint")
        if deleted:
            to_delete.append(p)
            print(" -> 整片清空, 删除"); continue
        Image.fromarray(new).save(fp)
        p.update({"x": int(p["x"] + offx), "y": int(p["y"] + offy),
                  "width": int(new.shape[1]), "height": int(new.shape[0])})
        print(f" -> 补实并裁到 {new.shape[1]}x{new.shape[0]} @({p['x']},{p['y']})")

    if to_delete and not dry:
        names = {id(p) for p in to_delete}
        for p in to_delete:
            fp = os.path.join(parts_dir, p["image"])
            if os.path.isfile(fp):
                os.remove(fp)
        coords["parts"] = [p for p in coords["parts"] if id(p) not in names]
        for p in coords["parts"]:
            if p.get("follow_part") in {q["name"] for q in to_delete}:
                p.pop("follow_part", None)
        print(f"  删除部件: {sorted(q['name'] for q in to_delete)}")

    if dry:
        print("  [dry] 未写入"); return 0

    # 3) 写回
    shutil.copyfile(cpath, cpath + ".bak_v2")
    json.dump(coords, open(cpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    Image.fromarray(scarf_img).save(os.path.join(parts_dir, owner["image"]))

    # 3b) 头巾层里混进来的"非头巾"小块(纯黑描边残片、棕袍角)会跟着头巾一起飘,
    #     看着像凭空多出几笔。按"整块里有没有金色"判断, 丢掉不含金色的 <150px 小块。
    _ow_fp = os.path.join(parts_dir, owner["image"])
    _ow = np.asarray(Image.open(_ow_fp).convert("RGBA")).copy()
    _a0 = (_ow[..., 3] > 20).astype(np.uint8)
    _nn, _ll, _ss, _ = cv2.connectedComponentsWithStats(_a0, 8)
    _R, _G, _B = (int_col.astype(int) for int_col in
                  (_ow[..., 0], _ow[..., 1], _ow[..., 2]))
    _gold = (_R > 120) & (_G > 95) & (_B < 130) & (_R - _B > 40)
    _drop = np.zeros_like(_a0, bool)
    for _k in range(1, _nn):
        _ar = int(_ss[_k, 4])
        if _ar >= 150:
            continue
        _sel = (_ll == _k)
        if int((_gold & _sel).sum()) < _ar * 0.25:
            _drop |= _sel
    if _drop.any():
        _ow[_drop] = 0
        Image.fromarray(_ow).save(_ow_fp)
        print(f"  头巾层清掉非头巾碎块 {int(_drop.sum())}px")

    # 4) 校验: 除 headwear_overlay 外, 其余部件不得再含【可见黄色】头巾像素
    #    (inpaint 后空洞位置变肤色 alpha 仍 255, 必须用颜色判定而非 source 掩码)
    residual = 0
    for p in coords["parts"]:
        if p is owner:
            continue
        fp = os.path.join(parts_dir, p["image"])
        if not os.path.isfile(fp):
            continue
        im = np.asarray(Image.open(fp).convert("RGBA"))
        pa = im[..., 3] > 8
        if int(pa.sum()) == 0:
            continue
        ih = cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2HSV)
        Hh_, Ss_, Vv_ = ih[..., 0].astype(int), ih[..., 1].astype(int), ih[..., 2].astype(int)
        yellow = pa & (Hh_ >= 15) & (Hh_ <= 47) & (Ss_ >= 80) & (Vv_ >= 120)
        r = int(yellow.sum())
        if r:
            print(f"  !! 残留黄色: {p['name']} 仍有 {r}px 可见黄色头巾")
            residual += r
    print(f"校验可见黄色残留: {residual}px (应为 0)")

    # 5) 合成校验
    composite = np.zeros((H, W, 4), np.uint8)
    for p in coords["parts"]:
        fp = os.path.join(parts_dir, p["image"])
        if not os.path.isfile(fp):
            continue
        im = np.asarray(Image.open(fp).convert("RGBA"))
        yy0, xx0 = p["y"], p["x"]
        yy1, xx1 = min(H, yy0 + im.shape[0]), min(W, xx0 + im.shape[1])
        sub = composite[yy0:yy1, xx0:xx1]
        ov = im[..., 3] > 0
        sub[ov] = im[:yy1 - yy0, :xx1 - xx0][ov]
    diff = np.any(composite[..., :3] != src[..., :3], axis=2) & (src[..., 3] > 0)
    print(f"合成校验: 与原图差异像素 {int(diff.sum())} / 前景 {int((src[...,3] > 0).sum())}")
    Image.fromarray(composite).save(os.path.join(parts_dir, "..", "_composite_check.png"))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts_dir")
    ap.add_argument("--src", default="")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.parts_dir, a.src, a.dry))
