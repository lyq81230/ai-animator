# -*- coding: utf-8 -*-
"""把「原图有、但没有任何部件覆盖」的像素补回来。

AI 分层偶尔会把整块东西漏掉(皮甲的盾牌、全甲的膝上段), 之后无论怎么调部件都补不出来,
因为那些像素根本不在任何一层里。但 `layerdiff_output/<name>/src_img.png`(原图)还在,
所以可以用「原图 alpha 减去 部件并集」求出缺口, 再从原图按画布比例抠回来。

用法:
    # 只体检, 不改数据
    python fill_missing.py --work output/sprite_pi_white_336 --list

    # 补齐: >=800px 的缺口单独成件(--prop-name/--prop-bind), 其余并进最近的部件
    python fill_missing.py --work output/sprite_pi_white_336 --apply \
        --prop-name shield --prop-cls prop --prop-bind arm_hand_R
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER_ROOT = os.path.join(HERE, "seethrough_repo", "workspace", "layerdiff_output")


def load_parts(work):
    coords_path = os.path.join(work, "coords.json")
    c = json.load(open(coords_path, encoding="utf-8"))
    return c, coords_path, os.path.join(work, "parts")


def coverage(c, parts_dir):
    W, H = c["canvas"]["w"], c["canvas"]["h"]
    cov = np.zeros((H, W), bool)
    for p in c["parts"]:
        fp = os.path.join(parts_dir, p["image"])
        if not os.path.isfile(fp):
            continue
        im = np.array(Image.open(fp).convert("RGBA"))
        m = im[..., 3] > 12
        cov[p["y"]:p["y"] + m.shape[0], p["x"]:p["x"] + m.shape[1]] |= m
    return cov, W, H


def src_alpha(layers_dir, W, H):
    """原图 alpha, 缩到画布尺寸(最近邻, 只用来定形状)。"""
    src = os.path.join(layers_dir, "src_img.png")
    if not os.path.isfile(src):
        sys.exit(f"找不到原图: {src}")
    a = np.array(Image.open(src).convert("RGBA"))[..., 3]
    return np.array(Image.fromarray(a).resize((W, H), Image.NEAREST)) > 12


def bg_mask(rgba):
    """近白/浅灰的低饱和像素 = 残留背景, 不能当"缺失内容"补进去。

    原图背景是白的时候, AI 分层常在"手臂与身体之间"这种窄缝里留下一条
    高 alpha 的白色 —— 它的 alpha 不是 0, 会被 `src_alpha` 当成前景。
    上一次就是这样把两条白条补进了 arm_fore_L(用户: "白色背景未去除")。
    """
    rgb = rgba[..., :3].astype(int)
    luma = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    return (luma > 205) & (sat < 22)


def src_rgba(layers_dir, W, H):
    """原图缩到画布尺寸。先做预乘再缩放, 避免边缘发黑。"""
    src = os.path.join(layers_dir, "src_img.png")
    im = np.array(Image.open(src).convert("RGBA")).astype(np.float32) / 255.0
    rgb, a = im[..., :3] * im[..., 3:4], im[..., 3:4]
    rgb = np.dstack([cv2.resize(rgb[..., k], (W, H), interpolation=cv2.INTER_AREA)
                     for k in range(3)])
    a = cv2.resize(a[..., 0], (W, H), interpolation=cv2.INTER_AREA)[..., None]
    out = np.zeros((H, W, 4), np.float32)
    out[..., :3] = np.where(a > 1e-6, rgb / np.maximum(a, 1e-6), 0)
    out[..., 3:4] = a
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def find_gaps(work, layers_dir, min_area=150):
    c, coords_path, parts_dir = load_parts(work)
    cov, W, H = coverage(c, parts_dir)
    sa = src_alpha(layers_dir, W, H)
    # 原图里"近白低饱和"的像素算背景, 不算遗漏内容
    bg = bg_mask(src_rgba(layers_dir, W, H))
    gap = sa & ~cov & ~bg
    n, lab, st, cent = cv2.connectedComponentsWithStats(gap.astype(np.uint8), 8)
    comps = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h = (int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP]),
                      int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT]))
        comps.append({"idx": i, "area": area, "box": (x, y, w, h),
                      "cx": float(cent[i][0]), "cy": float(cent[i][1])})
    comps.sort(key=lambda d: -d["area"])
    return c, coords_path, parts_dir, cov, sa, lab, comps, W, H


def nearest_part(c, p):
    """缺口块 -> 离它最近的部件(按 bbox 距离, 同等距离取面积大的)。"""
    x, y, w, h = p["box"]
    best, bestd = None, 1e18
    for q in c["parts"]:
        qx0, qy0 = q["x"], q["y"]
        qx1, qy1 = q["x"] + q["width"], q["y"] + q["height"]
        dx = max(qx0 - x, 0, x - qx1)
        dy = max(qy0 - y, 0, y - qy1)
        d = (dx * dx + dy * dy) ** 0.5
        if d < bestd - 1e-6 or (abs(d - bestd) < 1e-6 and
                                best and q["width"] * q["height"] > best["width"] * best["height"]):
            best, bestd = q, d
    return best, bestd


def blit_into(part, add_rgba, parts_dir, W, H):
    """把 add_rgba(整画布) over 到 part 上, 写回并更新 bbox。"""
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cv.alpha_composite(Image.open(os.path.join(parts_dir, part["image"])).convert("RGBA"),
                       (part["x"], part["y"]))
    cv.alpha_composite(Image.fromarray(add_rgba), (0, 0))
    arr = np.array(cv)
    ys, xs = np.nonzero(arr[..., 3] > 8)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    crop = cv.crop((x0, y0, x1, y1))
    crop.save(os.path.join(parts_dir, part["image"]))
    part.update({"x": x0, "y": y0, "width": crop.width, "height": crop.height})
    return crop.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--layers", default=None, help="默认按 work 目录名去 layerdiff_output 下找")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-area", type=int, default=150)
    ap.add_argument("--prop-name", default=None, help="大缺口单独成件的名字(如 shield)")
    ap.add_argument("--prop-cls", default="prop")
    ap.add_argument("--prop-bind", default=None, help="绑到哪根骨(如 arm_hand_R)")
    ap.add_argument("--prop-min", type=int, default=800, help="多大算'独立道具'")
    ap.add_argument("--map", default=None,
                    help="按面积从大到小指定每个缺口块的归属, 例: "
                         "'0=new:shield:prop:arm_hand_R,1=headwear,2=arm_fore_L'。"
                         "target 是已有部件名, 或 new:名字:cls:跟随部件")
    args = ap.parse_args()

    work = args.work.rstrip("/\\")
    layers = args.layers or os.path.join(LAYER_ROOT, os.path.basename(work))
    if not os.path.isdir(layers):
        sys.exit(f"分层目录不存在: {layers}")

    c, coords_path, parts_dir, cov, sa, lab, comps, W, H = find_gaps(
        work, layers, args.min_area)
    print(f"{os.path.basename(work)}: 原图 {int(sa.sum())}px, 已覆盖 {int(cov.sum())}px, "
          f"缺口块 {len(comps)} 个")
    for p in comps:
        tgt, d = nearest_part(c, p)
        print(f"   面积{p['area']:6d} bbox{tuple(p['box'])} 中心({p['cx']:.0f},{p['cy']:.0f})"
              f"  最近部件={tgt['name']}(距离{d:.0f})")
    if not args.apply:
        return

    src = src_rgba(layers, W, H)
    if args.apply and not os.path.isfile(os.path.join(work, "coords.json.bak")):
        shutil.copy(coords_path, coords_path + ".bak")

    plan = {}
    if args.map:
        for item in args.map.split(","):
            if not item.strip():
                continue
            k, v = item.split("=", 1)
            plan[int(k)] = v.strip()

    # 缺口像素稍微膨胀 1px, 把抗锯齿边缘也带上, 免得补出来有硬边
    new_parts = []
    bg_src = bg_mask(src)          # 膨胀时别把背景白边带进来
    for ci, p in enumerate(comps):
        m = (lab == p["idx"]).astype(np.uint8)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8)) & sa & (~bg_src)
        add = np.zeros((H, W, 4), np.uint8)
        add[m.astype(bool)] = src[m.astype(bool)]

        tgt_name = plan.get(ci)
        if tgt_name and tgt_name.startswith("new:"):
            _, nm, cls, follow = tgt_name.split(":")
            ys, xs = np.nonzero(add[..., 3] > 8)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            crop = Image.fromarray(add[y0:y1, x0:x1])
            crop.save(os.path.join(parts_dir, nm + ".png"))
            np_ = {"name": nm, "image": nm + ".png", "x": x0, "y": y0,
                   "width": crop.width, "height": crop.height,
                   "grid": [3, 3], "cls": cls}
            if follow and follow != "-":
                np_["follow_part"] = follow
            new_parts.append(np_)
            print(f"   新部件 {nm}: {crop.width}x{crop.height} @({x0},{y0}) follow={follow}")
            continue

        if tgt_name:
            tgt = next((q for q in c["parts"] if q["name"] == tgt_name), None)
            if tgt is None:
                print(f"   !! 找不到部件 {tgt_name}, 跳过")
                continue
        elif args.prop_name and p["area"] >= args.prop_min:
            ys, xs = np.nonzero(add[..., 3] > 8)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            crop = Image.fromarray(add[y0:y1, x0:x1])
            name = args.prop_name
            crop.save(os.path.join(parts_dir, name + ".png"))
            np_ = {"name": name, "image": name + ".png", "x": x0, "y": y0,
                   "width": crop.width, "height": crop.height, "grid": [3, 3],
                   "cls": args.prop_cls}
            if args.prop_bind:
                np_["follow_part"] = args.prop_bind
            new_parts.append(np_)
            print(f"   新部件 {name}: {crop.width}x{crop.height} @({x0},{y0})")
            continue
        else:
            tgt, _d = nearest_part(c, p)

        sz = blit_into(tgt, add, parts_dir, W, H)
        print(f"   并入 {tgt['name']} -> {sz[0]}x{sz[1]} @({tgt['x']},{tgt['y']})")

    if new_parts:
        # 新部件插到它跟随的那只手的 slot 前面(盾牌/武器压在手背上, 但不该盖住拳头)
        for np_ in new_parts:
            ins = len(c["parts"])
            fol = np_.get("follow_part")
            if fol:
                for i, q in enumerate(c["parts"]):
                    if q["name"] == fol:
                        ins = i
                        break
            c["parts"].insert(ins, np_)

    json.dump(c, open(coords_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("部件:", [q["name"] for q in c["parts"]])


if __name__ == "__main__":
    main()
