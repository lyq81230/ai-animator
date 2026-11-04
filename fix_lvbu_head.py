# -*- coding: utf-8 -*-
"""重建吕布的头部/头饰, 并清掉 AI 蒙错的五官碎块。

背景: seethrough 对这张图给出的是"整头"和一堆错位的肤色块(ears/nose/mouth/
eyelash/eyebrow 全是椭圆色块, 位置也不在脸上), 直接信它 → 拆图里 head 变成
"头盔 + 两肩 + 整个胸甲", 脸跑到五官碎块里, 五官碎块又装着发髻和盔顶。

做法(不依赖 AI 五官层):
    head      = 源图 ∧ 头部多边形(盔 + 脸 + 侧护颊)
    headwear  = 源图 ∧ (冠+发髻多边形 ∪ 红缨多边形)
    ears/nose/mouth/eyelash/eyebrow 直接删除, 像素归到 head/headwear
    其余部位里落在上述 mask 内的像素清掉(避免重影)

用法:
    python fix_lvbu_head.py            # 直接改 output/sprite_boss_lvbu_white_336
    python fix_lvbu_head.py --preview  # 只出候选 mask 预览, 不动文件
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

D = "output/sprite_boss_lvbu_white_336"
SRC = "E:/pic/sprite_boss_lvbu_white_336.png"

# 头部(盔+脸+护颊), 顺时针, 画布坐标
HEAD_POLY = [
    (165, 49), (172, 47), (183, 46), (194, 48), (200, 51),
    (203, 57), (206, 65), (208, 75), (207, 85), (203, 93),
    (197, 98), (188, 100), (176, 100), (169, 97), (165, 92),
    (160, 86), (158, 76), (159, 65), (161, 56),
]

# 冠 + 发髻
CROWN_POLY = [
    (166, 12), (198, 12), (201, 30), (202, 44), (196, 49),
    (184, 49), (172, 47), (166, 40), (164, 30),
]

# 红缨
PLUME_POLY = [
    (194, 38), (204, 42), (214, 46), (224, 49), (232, 53),
    (238, 57), (238, 64), (233, 66), (229, 72), (226, 70),
    (222, 66), (218, 68), (215, 74), (212, 78), (210, 74),
    (208, 68), (205, 60), (200, 52), (196, 45),
]

FACE_PARTS = ["ears", "nose", "mouth", "eyelash", "eyebrow"]


def poly_mask(poly, w, h):
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon(poly, fill=255)
    return np.array(im) > 0


def tight(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()


def build_masks(w, h):
    head = poly_mask(HEAD_POLY, w, h)
    orn = poly_mask(CROWN_POLY, w, h) | poly_mask(PLUME_POLY, w, h)
    # 边界上有一条条浅灰"发丝"横穿(原图 y≈46~52), 多边形会把它切断成虚线,
    # 所以两个 mask 都往外胀一点, 把贴边的发丝/描边一起带走。
    from scipy import ndimage as ndi
    head = ndi.binary_dilation(head, np.ones((3, 3)), iterations=1)
    orn = ndi.binary_dilation(orn, np.ones((3, 3)), iterations=2)
    orn &= ~head
    return head, orn


def preview(c):
    W, H = c["canvas"]["w"], c["canvas"]["h"]
    src = np.array(Image.open(SRC).convert("RGBA"))
    head, orn = build_masks(W, H)
    alpha = src[..., 3] > 8
    out = []
    for nm, m in (("head", head & alpha), ("headwear", orn & alpha)):
        buf = np.zeros_like(src)
        buf[m] = src[m]
        x0, y0, x1, y1 = tight(m)
        crop = Image.fromarray(buf[y0:y1 + 1, x0:x1 + 1])
        z = max(1, min(8, 380 // max(crop.size)))
        crop = crop.resize((crop.size[0] * z, crop.size[1] * z), Image.NEAREST)
        bg = Image.new("RGBA", crop.size, (70, 70, 78, 255))
        bg.alpha_composite(crop)
        d = ImageDraw.Draw(bg)
        d.text((3, 3), f"{nm} {x1-x0+1}x{y1-y0+1}", fill=(255, 255, 0, 255))
        out.append(bg.convert("RGB"))
        print(f"{nm}: box=({x0},{y0},{x1},{y1}) {x1-x0+1}x{y1-y0+1} "
              f"{int(m.sum())}px")
    Wm = sum(i.size[0] for i in out) + 12
    Hm = max(i.size[1] for i in out)
    M = Image.new("RGB", (Wm, Hm), (20, 20, 20))
    x = 0
    for i in out:
        M.paste(i, (x, 0))
        x += i.size[0] + 12
    M.save("_lvbu_head_cand.png")
    print("预览: _lvbu_head_cand.png")


def apply(c, D):
    W, H = c["canvas"]["w"], c["canvas"]["h"]
    src = np.array(Image.open(SRC).convert("RGBA"))
    alpha = src[..., 3] > 8
    head, orn = build_masks(W, H)
    head &= alpha
    orn &= alpha

    parts = {p["name"]: p for p in c["parts"]}

    # 0) 记下旧 head 的覆盖范围: 它多占的肩甲/胸甲像素要在后面还给 topwear,
    #    否则这些像素会凭空消失, 拼起来是空洞。
    ph = parts["head"]
    old_im = np.array(Image.open(os.path.join(D, "parts", ph["image"])).convert("RGBA"))
    old_head = np.zeros((H, W), bool)
    oh, ow = old_im.shape[:2]
    old_head[ph["y"]:ph["y"] + oh, ph["x"]:ph["x"] + ow] = old_im[..., 3] > 8

    # 1) 别的部位里, 落在新 mask 内的像素清掉(这些像素归 head/headwear)
    take = head | orn
    for p in c["parts"]:
        if p["name"] in ("head", "headwear") or p["name"] in FACE_PARTS:
            continue
        im = Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA")
        a = np.array(im)
        h, w = a.shape[:2]
        x0, y0 = p["x"], p["y"]
        sub = take[y0:y0 + h, x0:x0 + w]
        if sub.any():
            a[..., 3][sub] = 0
            Image.fromarray(a).save(os.path.join(D, "parts", p["image"]))

    # 2) 写 head / headwear
    for nm, m in (("head", head), ("headwear", orn)):
        x0, y0, x1, y1 = tight(m)
        buf = np.zeros((y1 - y0 + 1, x1 - x0 + 1, 4), np.uint8)
        buf[m[y0:y1 + 1, x0:x1 + 1]] = src[y0:y1 + 1, x0:x1 + 1][m[y0:y1 + 1, x0:x1 + 1]]
        Image.fromarray(buf).save(os.path.join(D, "parts", nm + ".png"))
        p = parts[nm]
        p.update({"image": nm + ".png", "x": int(x0), "y": int(y0),
                  "width": int(x1 - x0 + 1), "height": int(y1 - y0 + 1)})
        p.pop("grid", None)
        print(f"{nm}: ({x0},{y0}) {x1-x0+1}x{y1-y0+1} {int(m.sum())}px")

    # 3) 删除错位的五官碎块
    c["parts"] = [p for p in c["parts"] if p["name"] not in FACE_PARTS]
    for nm in FACE_PARTS:
        f = os.path.join(D, "parts", nm + ".png")
        if os.path.isfile(f):
            os.remove(f)
    print("已删除:", ", ".join(FACE_PARTS))

    # 3.5) 旧 head 多占的肩/胸像素还给 topwear(扩大它的包围盒)
    leftover = old_head & ~(head | orn)
    if leftover.any():
        pt = parts["topwear"]
        tim = np.array(Image.open(os.path.join(D, "parts", pt["image"])).convert("RGBA"))
        tx, ty, th, tw = pt["x"], pt["y"], tim.shape[0], tim.shape[1]
        ys, xs = np.nonzero(leftover)
        nx0, ny0 = min(tx, int(xs.min())), min(ty, int(ys.min()))
        nx1 = max(tx + tw - 1, int(xs.max()))
        ny1 = max(ty + th - 1, int(ys.max()))
        big = np.zeros((ny1 - ny0 + 1, nx1 - nx0 + 1, 4), np.uint8)
        big[ty - ny0:ty - ny0 + th, tx - nx0:tx - nx0 + tw] = tim
        big[ys - ny0, xs - nx0] = src[ys, xs]
        Image.fromarray(big).save(os.path.join(D, "parts", pt["image"]))
        pt.update({"image": pt["image"], "x": int(nx0), "y": int(ny0),
                   "width": int(nx1 - nx0 + 1), "height": int(ny1 - ny0 + 1)})
        print(f"topwear 扩到 ({nx0},{ny0}) {nx1-nx0+1}x{ny1-ny0+1}, "
              f"接收 {int(leftover.sum())}px 肩/胸")

    json.dump(c, open(os.path.join(D, "coords.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 4) 覆盖率自检 + 补洞
    src_a = alpha
    def coverage():
        cov = np.zeros((H, W), np.int32)
        for p in c["parts"]:
            a = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))[..., 3]
            h, w = a.shape
            cov[p["y"]:p["y"] + h, p["x"]:p["x"] + w] += (a > 8)
        return cov
    cov = coverage()
    hole = src_a & (cov == 0)
    if hole.any():
        # 洞按"离哪块最近就归哪块"补回去(多边形边缘的抗锯齿像素会漏几个)
        from scipy import ndimage as ndi
        lab = np.zeros((H, W), np.int32)
        for i, p in enumerate(c["parts"], 1):
            a = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))[..., 3]
            h, w = a.shape
            lab[p["y"]:p["y"] + h, p["x"]:p["x"] + w][a > 8] = i
        _d, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        ys, xs = np.nonzero(hole)
        owner = lab[iy[ys, xs], ix[ys, xs]]
        for i, p in enumerate(c["parts"], 1):
            sel = owner == i
            if not sel.any():
                continue
            px, py = xs[sel], ys[sel]
            im = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))
            x0, y0 = min(p["x"], int(px.min())), min(p["y"], int(py.min()))
            x1 = max(p["x"] + im.shape[1] - 1, int(px.max()))
            y1 = max(p["y"] + im.shape[0] - 1, int(py.max()))
            big = np.zeros((y1 - y0 + 1, x1 - x0 + 1, 4), np.uint8)
            big[p["y"] - y0:p["y"] - y0 + im.shape[0],
                p["x"] - x0:p["x"] - x0 + im.shape[1]] = im
            big[py - y0, px - x0] = src[py, px]
            Image.fromarray(big).save(os.path.join(D, "parts", p["image"]))
            p.update({"x": int(x0), "y": int(y0),
                      "width": int(x1 - x0 + 1), "height": int(y1 - y0 + 1)})
        print(f"补洞 {int(hole.sum())}px")
        cov = coverage()
        hole = src_a & (cov == 0)
    print(f"覆盖: 空缺 {int(hole.sum())}px, 重叠 {int((cov>1).sum())}px")
    if hole.any():
        ys, xs = np.nonzero(hole)
        print("  空缺范围", (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))


def main():
    c = json.load(open(os.path.join(D, "coords.json"), encoding="utf-8"))
    if "--preview" in sys.argv:
        preview(c)
        return
    apply(c, D)


if __name__ == "__main__":
    main()
