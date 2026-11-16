# -*- coding: utf-8 -*-
"""从**源图**重建 head / irides, 并补齐头巾 overlay 边缘的锯齿洞。

问题背景
--------
这个角色的 head 层 AI 只给了一半脸, 眼睛处还被挖成两个大洞。之前几轮修
(rebuild_head_v*.py / clean5) 都是**从坏掉的 head.png 出发**反复去棕、裁边,
越修越小: 脸上沿、右颊、下巴各被切掉一圈。而 headwear_overlay 中间给脸留的洞
是按更早那张更大的脸挖的 —— 两块缝合不上, 合成时露出一圈 1~3px 白边。
用户看到的"拆图错误 / app 拆图失败"就是这条白边(**不是**脚本报错)。

做法
----
不再猜脸的形状, 改用**占位反推**: 
    头部区域 = 源图有像素 ∧ 其它所有部件都没占的位置
因为头巾/衣服/手臂已经把各自的范围占掉了, 剩下的最大连通块就是脸
(含眼窝、裂缝、自己的黑描边), 天然和邻居严丝合缝, 不可能再出白边。
红瞳从 head 里挖出来放进 irides(= 它本来就在 head 下层, 靠洞透出来),
洞和 irides 的 alpha 严格相等 → 既不漏白也不压黑。

用法
----
    python fix_zombie_head.py                 # 改 output/zombie_fan_hero_338
    python fix_zombie_head.py --check         # 只体检
    python fix_zombie_head.py --src a.png --out output/xxx
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_OUT = os.path.join(HERE, "output", "zombie_fan_hero_338")
SRC_DIR = "E:/pic"

# 头部搜索区(源图坐标): 只有落在这里面的"无主像素"才可能属于脸
ZONE = (114, 234, 58, 148)      # x0, x1, y0, y1
MIN_FACE = 1200                 # 最大的无主连通块要够大才认
# 红瞳: 眼睛所在的一条横带, 避免把扇子上的红字也算进来
EYE_BAND = (138, 205, 86, 101)
CHIN_Y = 148                    # 下巴最低到这一行(源图坐标), 再往下是脖子


def log(*a):
    print(*a, flush=True)


def load(p):
    return np.array(Image.open(p).convert("RGBA"))


def fill_holes(mask):
    m = mask.astype(np.uint8)
    h, w = m.shape
    ff = m.copy()
    mm = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mm, (0, 0), 2)
    return m.astype(bool) | ((ff != 2) & (m == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", nargs="?", default=DEF_OUT,
                    help="角色输出目录, 如 output/zombie_fan_hero_338")
    ap.add_argument("--src", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(a.out or a.work)
    parts = os.path.join(out, "parts")
    cp = os.path.join(parts, "coords.json")
    if not os.path.isfile(cp):
        raise SystemExit(f"没有 coords.json: {cp}")
    # 源图: 没有 --src 就按角色名去 E:/pic 找
    name = os.path.basename(os.path.normpath(out))
    src_path = a.src or os.path.join(SRC_DIR, name + ".png")
    if not os.path.isfile(src_path):
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            cand = os.path.join(SRC_DIR, name + ext)
            if os.path.isfile(cand):
                src_path = cand
                break
    if not os.path.isfile(src_path):
        raise SystemExit(f"找不到源图: {src_path}\n(用 --src 指定)")
    a.src = src_path
    src = load(a.src)
    H, W = src.shape[:2]
    coords = json.load(open(cp, encoding="utf-8"))
    pmap = {p["name"]: p for p in coords["parts"]}
    for need in ("head", "irides", "headwear_overlay"):
        if need not in pmap:
            raise SystemExit(f"coords 里缺部件 {need}")
    log(f"源图 {a.src}  {W}x{H}   部件 {len(coords['parts'])}")

    # ---------- 0. 先删掉源图里根本不存在的部件 ----------
    # 这个角色整颗头包在头巾里, AI 却编了一条 113x42 的 ears 横带(内容就是额头
    # 那圈头巾+皮肤)。留着它会占住脸的位置, 让下面的"无主像素"少一块, 所以先删。
    for nm in ("ears",):
        p = pmap.get(nm)
        if p is None:
            continue
        coords["parts"] = [q for q in coords["parts"] if q is not p]
        f = os.path.join(parts, p["image"])
        if os.path.isfile(f):
            os.remove(f)
        pmap.pop(nm, None)
        log(f"删除无源头部件: {nm} ({p.get('width')}x{p.get('height')})")

    # ---------- 1. 其它部件的占位图 ----------
    occ = np.zeros((H, W), bool)
    for p in coords["parts"]:
        if p["name"] == "head":
            continue
        al = load(os.path.join(parts, p["image"]))[:, :, 3]
        # irides 是半透明光晕, 只用它实心的部分占位, 否则会把眼窝挖空
        thr = 150 if p["name"] in ("irides", "eyes") else 20
        hh, ww = al.shape
        x0, y0 = max(0, p["x"]), max(0, p["y"])
        x1, y1 = min(W, p["x"] + ww), min(H, p["y"] + hh)
        if x1 > x0 and y1 > y0:
            occ[y0:y1, x0:x1] |= (al[y0 - p["y"]:y1 - p["y"],
                                     x0 - p["x"]:x1 - p["x"]] > thr)

    fig = src[:, :, 3] > 20
    free = fig & ~occ
    zx0, zx1, zy0, zy1 = ZONE
    zone = np.zeros((H, W), bool)
    zone[zy0:zy1, zx0:zx1] = True
    cand = free & zone
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    if n < 2:
        raise SystemExit("头部区里没有无主像素, 检查 ZONE")
    areas = [(int(st[k, cv2.CC_STAT_AREA]), k) for k in range(1, n)]
    areas.sort(reverse=True)
    log("无主连通块(前5): " + ", ".join(f"{ar}px" for ar, _ in areas[:5]))
    if areas[0][0] < MIN_FACE:
        raise SystemExit(f"最大的无主块只有 {areas[0][0]}px, 不像脸")
    head_m = (lab == areas[0][1])
    head_m = fill_holes(head_m)
    # 下巴被 topwear 占了, "无主像素"够不到 -> 按源图颜色往下长。
    # 只吃**亮肤色**(暗脖子/袍子排除), 且只长到 y<CHIN_Y。
    r0, g0, b0 = (src[:, :, i].astype(np.int16) for i in range(3))
    vv = src[:, :, :3].max(2)
    sat = vv - src[:, :, :3].min(2)
    skin = (vv > 100) & (vv < 220) & (sat < 60) & (g0 >= r0 - 12)
    dark = (vv <= 55) & fig
    for _ in range(12):
        grow = cv2.dilate(head_m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        add = grow & (~head_m) & skin
        add[:zy0] = False
        add[CHIN_Y:] = False
        add[:, :zx0] = False
        add[:, zx1:] = False
        if not add.any():
            break
        head_m = head_m | add
    # 逐像素长到够不到的地方就停(离脸 12px 以上、或者被暗描边隔开的) —— 下巴右角
    # 就被"暗影 + 袍领"夹出一块咬缺。改成: 直接拿**整块脸肤色**补进来(闭运算补裂缝、
    # 取包含脸心的那块, 天然被暗影切断, 不会顺带吃脖子/手臂)。
    sk = (skin & zone).astype(np.uint8)
    sk = cv2.morphologyEx(sk, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ns, ls, ss, _ = cv2.connectedComponentsWithStats(sk, 8)
    face_c = ls[121, 175] if ns > 1 else 0
    if face_c == 0:                       # 脸心落在暗处时, 取和 head_m 重叠最多的那块
        best, face_c = -1, 0
        for k in range(1, ns):
            ov = int(((ls == k) & head_m).sum())
            if ov > best:
                best, face_c = ov, k
    if face_c:
        face_sk = (ls == face_c) & zone & fig
        face_sk[:zy0] = False
        face_sk[CHIN_Y:] = False
        extra = face_sk & ~head_m
        if extra.any():
            log(f"脸部肤色补全: +{int(extra.sum())}px")
            head_m = head_m | face_sk
    # 描边: 只补离脸 2px 以内的黑线。*不能*让黑线一直"顺着长" ——
    # 脸边界上常接到头巾褶皱/衣领的细线, 顺着长会拖出几条黑刺(用户说的"太散")。
    band = cv2.dilate(head_m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    head_m = head_m | (dark & band)
    # 再瘦一次: 3x3 开运算去掉 1px 毛刺, 但保留贴着脸的那圈
    core = cv2.morphologyEx(head_m.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8)).astype(bool)
    head_m = head_m & cv2.dilate(core.astype(np.uint8),
                                 np.ones((3, 3), np.uint8)).astype(bool)
    # 剪"刺": 又细(离背景 1.4px 内)又只有一个邻居的端点, 反复剪几轮。
    # 脸自己那圈描边有 2~3px 厚, 剪不到; 顺着衣领/褶皱拖出去的黑刺会没。
    for _ in range(6):
        u8 = head_m.astype(np.uint8)
        dt = cv2.distanceTransform(u8, cv2.DIST_L2, 3)
        nb = cv2.filter2D(u8, -1, np.ones((3, 3), np.uint8)) - u8
        spur = head_m & (dt < 1.4) & (nb <= 2)
        if not spur.any():
            break
        head_m = head_m & ~spur
    head_m = fill_holes(head_m)

    ys, xs = np.nonzero(head_m)
    log(f"脸: {int(head_m.sum())}px  x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")

    # ---------- 2. 红瞳 ----------
    r, g, b = (src[:, :, i].astype(np.int16) for i in range(3))
    ex0, ex1, ey0, ey1 = EYE_BAND
    band = np.zeros((H, W), bool)
    band[ey0:ey1, ex0:ex1] = True
    iris = head_m & band & (r > 60) & (r >= g + 25) & (r >= b + 20)
    iris = fill_holes(iris) if iris.sum() else iris
    if iris.sum() == 0:
        log("⚠ 没找到红瞳, irides 保持原样")
    else:
        ys, xs = np.nonzero(iris)
        log(f"红瞳: {int(iris.sum())}px  x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")

    if a.check:
        vis = src.copy()
        vis[head_m] = [0, 255, 0, 255]
        vis[iris] = [255, 0, 255, 255]
        Image.fromarray(vis).resize((W * 2, H * 2), Image.NEAREST).save(
            os.path.join(HERE, "_head_mask_preview.png"))
        log("预览: _head_mask_preview.png")
        return

    # ---------- 3. 按需要扩窗, 写 head / irides ----------
    def fit_window(p, mask):
        """窗口收到 mask 的包围盒(+1px), 不要沿用拆图给的大窗口。
        大窗口本身不错, 但网格/槽位尺寸会跟着虚胖, 编辑器里看不出部件边界。"""
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        left = max(0, x0 - 1)
        top = max(0, y0 - 1)
        right = min(W, x1 + 2)
        bot = min(H, y1 + 2)
        if (left, top, right, bot) != (p["x"], p["y"],
                                       p["x"] + p["width"], p["y"] + p["height"]):
            log(f"  {p['name']} 窗口 {p['x']},{p['y']},{p['width']},{p['height']} -> "
                f"{left},{top},{right-left},{bot-top}")
        p["x"], p["y"] = left, top
        p["width"], p["height"] = right - left, bot - top
        return p

    def write_part(p, mask):
        p = fit_window(p, mask)
        x, y, w, h = p["x"], p["y"], p["width"], p["height"]
        canvas = np.zeros((h, w, 4), np.uint8)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        sub = mask[y0:y1, x0:x1]
        dst = canvas[y0 - y:y1 - y, x0 - x:x1 - x]
        dst[...] = np.where(sub[..., None], src[y0:y1, x0:x1], 0)
        Image.fromarray(canvas, "RGBA").save(os.path.join(parts, p["image"]))
        yy, xx = np.nonzero(canvas[:, :, 3] > 20)
        log(f"  {p['name']:18s} {w}x{h} @({x},{y}) px={int((canvas[:,:,3]>20).sum())} "
            f"local x[{xx.min()},{xx.max()}] y[{yy.min()},{yy.max()}]")

    write_part(pmap["head"], head_m & ~iris)
    if iris.sum():
        write_part(pmap["irides"], iris)

    # ---------- 4. 头巾 overlay 边缘的锯齿洞 ----------
    acc = np.zeros((H, W), np.uint8)
    for p in coords["parts"]:
        al = load(os.path.join(parts, p["image"]))[:, :, 3]
        hh, ww = al.shape
        x0, y0 = max(0, p["x"]), max(0, p["y"])
        x1, y1 = min(W, p["x"] + ww), min(H, p["y"] + hh)
        if x1 > x0 and y1 > y0:
            acc[y0:y1, x0:x1] = np.maximum(
                acc[y0:y1, x0:x1], al[y0 - p["y"]:y1 - p["y"], x0 - p["x"]:x1 - p["x"]])
    gap = (src[:, :, 3] > 120) & (acc < 20)
    log(f"未覆盖缺口: {int(gap.sum())}px")
    if gap.sum():
        ow = pmap["headwear_overlay"]
        ov = load(os.path.join(parts, ow["image"]))
        ox, oy, oww, ohh = ow["x"], ow["y"], ow["width"], ow["height"]
        oc = np.zeros((H, W), np.uint8)
        oc[oy:oy + ohh, ox:ox + oww] = ov[:, :, 3]
        # 脸现在整块在 head 里, 头巾窗口里剩下的空缺只可能是头巾自己的边/流苏,
        # 直接全补给它 —— 只要还在窗口内。(以前只补"贴着边"的, 流苏尖上留了个洞)
        win = np.zeros((H, W), bool)
        win[oy:oy + ohh, ox:ox + oww] = True
        add = gap & (oc < 20) & win
        if add.sum():
            for yy, xx in zip(*np.nonzero(add)):
                lx, ly = xx - ox, yy - oy
                if 0 <= lx < oww and 0 <= ly < ohh:
                    ov[ly, lx] = src[yy, xx]
            Image.fromarray(ov, "RGBA").save(os.path.join(parts, ow["image"]))
            log(f"  补进头巾: {int(add.sum())}px")
        left = gap & ~add
        if left.sum():
            n2, _, st2, _ = cv2.connectedComponentsWithStats(left.astype(np.uint8), 8)
            for k in range(1, n2):
                ar = int(st2[k, 4])
                if ar >= 30:
                    x, y, w2, h2 = (int(v) for v in st2[k, :4])
                    log(f"  仍有缺口 {ar}px x[{x},{x+w2}] y[{y},{y+h2}]")

    # ---------- 5. 落盘 coords ----------
    json.dump(coords, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    wc = os.path.join(out, "coords.json")
    if os.path.isfile(wc):
        json.dump(coords, open(wc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log("完成")


if __name__ == "__main__":
    main()
