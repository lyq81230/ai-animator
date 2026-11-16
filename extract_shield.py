# -*- coding: utf-8 -*-
"""把被切进右臂的盾牌(皮甲)还原成独立的一块 objects_R。

现象: 拆图把盾牌劈进了 arm_upper_R / arm_fore_R / arm_hand_R 三块里(共 ~3700px),
一弯肘盾就跟着三段各自走, 看着像被切成两半。

做法:
    1. 拿盾的像素模板(见 template(): config 母本 -> 上次渲染的 images 副本 ->
       老的 _bak_parts_* 备份), 用源图做模板匹配定位它在画布上的位置;
    2. 从当前右臂三块里把落在盾上的像素剪掉(手臂恢复成干净的手臂);
    3. objects_R 插到 arm_upper_R 之前(盾在手臂后面), 刚性跟小臂。

用法: python extract_shield.py [角色目录]

为什么改成这样
--------------
以前这里硬依赖 `_bak_parts_1931/<角色>/coords.json` —— 一次性的手工备份。
那份被清掉之后整条重建链就崩在 FileNotFoundError 上(用户报的那个),
而且没人知道该去哪恢复。盾的模板其实一直在 <work>/<动作>/images/ 里躺着
(gen 每次会把用到的部件复制一份), 现在把它固化成 config/<角色>/objects_R.png,
坐标记在 shield.json, 脚本三处依次找, 缺一份也能跑。
"""
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else "output/sprite_pi_white_336"
NAME = os.path.basename(os.path.normpath(D.rstrip("\\/")))
ARM_PARTS = ["arm_upper_R", "arm_fore_R", "arm_hand_R"]
SHIELD = "objects_R"
ACTS = ("walk", "hit", "death")
ALPHA = 40


def src_path():
    """源图: 和 fill_gaps 一样按角色名去 E:/pic 找。"""
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = os.path.join("E:/pic", NAME + ext)
        if os.path.isfile(p):
            return p
    return ""


def template():
    """返回 (模板路径, 记录坐标 dict)。按可靠度依次找三处。"""
    cfg = os.path.join(HERE, "config", NAME, "shield")
    p = os.path.join(cfg, "objects_R.png")
    if os.path.isfile(p):
        rec = {}
        jp = os.path.join(cfg, "shield.json")
        if os.path.isfile(jp):
            try:
                rec = json.load(open(jp, encoding="utf-8"))
            except ValueError:
                rec = {}
        return p, rec
    for a in ACTS:
        p = os.path.join(D, a, "images", "objects_R.png")
        if os.path.isfile(p):
            return p, {}
    for bak in sorted(glob.glob(os.path.join(HERE, "_bak_parts_*"))):
        p = os.path.join(bak, NAME, "parts", "objects_R.png")
        if os.path.isfile(p):
            rec = {}
            jp = os.path.join(bak, NAME, "coords.json")
            if os.path.isfile(jp):
                co = json.load(open(jp, encoding="utf-8"))
                for q in co.get("parts", []):
                    if q.get("name") == SHIELD:
                        rec = dict(q)
            return p, rec
    return "", {}


def locate(tpl_path, rec):
    """模板在画布上的 (x, y, score)。优先用源图匹配, 匹配不上退回记录坐标。"""
    a = np.array(Image.open(tpl_path).convert("RGBA"))
    m = (a[..., 3] > ALPHA).astype(np.float32)
    sp = src_path()
    if sp and m.any():
        src = np.array(Image.open(sp).convert("RGBA"))
        h, w = a.shape[:2]
        if src.shape[1] >= w and src.shape[0] >= h:
            # TM_CCORR_NORMED 支持 mask, 能忽略模板里的透明区(只按不透明的盾面对齐)
            res = cv2.matchTemplate(src[..., :3], a[..., :3], cv2.TM_CCORR_NORMED,
                                    mask=m)
            res = np.nan_to_num(res)
            _, mx, _, mloc = cv2.minMaxLoc(res)
            if mx >= 0.85:
                return int(mloc[0]), int(mloc[1]), float(mx)
            print(f"  模板匹配分数偏低 {mx:.2f}, 用记录坐标")
    if "x" in rec and "y" in rec:
        return int(rec["x"]), int(rec["y"]), 0.0
    return None


def canvas_of(part, D_, W, H):
    a = np.array(Image.open(os.path.join(D_, "parts", part["image"])).convert("RGBA"))
    m = np.zeros((H, W, 4), np.uint8)
    m[part["y"]:part["y"] + a.shape[0], part["x"]:part["x"] + a.shape[1]] = a
    return m


def main():
    cp = os.path.join(D, "coords.json")
    if not os.path.isfile(cp):
        raise SystemExit(f"没有 coords.json: {cp}")
    co = json.load(open(cp, encoding="utf-8"))
    by = {p["name"]: p for p in co["parts"]}
    if SHIELD in by:
        print(f"已经有 {SHIELD}, 跳过")
        return

    tpl, rec = template()
    if not tpl:
        raise SystemExit(
            "找不到盾模板。需要 config/<角色>/objects_R.png "
            "(或 <工作目录>/<动作>/images/objects_R.png)。")
    loc = locate(tpl, rec)
    if loc is None:
        raise SystemExit("盾在源图上定位失败, config/<角色>/shield.json 里也没有 x/y")
    x, y, score = loc

    a = np.array(Image.open(tpl).convert("RGBA"))
    h, w = a.shape[:2]
    dst = os.path.join(D, "parts", SHIELD + ".png")
    shutil.copy2(tpl, dst)
    shield = {
        "name": SHIELD, "image": SHIELD + ".png",
        "x": x, "y": y, "width": w, "height": h,
        "grid": list(rec.get("grid") or [4, 4]),
        "cls": rec.get("cls") or "prop",
        # 盾是绑在小臂上的, 不该跟手腕一起甩(挂手上时走路会左右摇)
        "follow_part": "arm_fore_R",
    }
    shield.pop("z", None)
    print(f"盾: ({x},{y}) {w}x{h}  匹配分数 {score:.2f}")

    mask = np.zeros((co["canvas"]["h"], co["canvas"]["w"]), bool)
    m = a[..., 3] > ALPHA
    mask[y:y + h, x:x + w] = m
    print(f"  盾的可见像素 {int(mask.sum())}")

    # 从右臂里剪掉盾的像素
    for nm in ARM_PARTS:
        p = by.get(nm)
        if not p:
            continue
        f = os.path.join(D, "parts", p["image"])
        arr = np.array(Image.open(f).convert("RGBA"))
        hh, ww = arr.shape[:2]
        sub = mask[p["y"]:p["y"] + hh, p["x"]:p["x"] + ww]
        cut = int(((arr[..., 3] > ALPHA) & sub).sum())
        if cut:
            arr[..., 3][sub] = 0
            Image.fromarray(arr).save(f)
            print(f"  {nm}: 剪掉盾的 {cut}px")

    idx = [i for i, q in enumerate(co["parts"]) if q["name"] == "arm_upper_R"]
    co["parts"].insert(idx[0] if idx else len(co["parts"]), shield)
    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pc = os.path.join(D, "parts", "coords.json")
    if os.path.isfile(pc):
        json.dump(co, open(pc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("已插入 objects_R:", [q["name"] for q in co["parts"]])


if __name__ == "__main__":
    main()
