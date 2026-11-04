# -*- coding: utf-8 -*-
"""把部件里残留的"纯白背景"抠掉。

为什么需要单独一步:
  AI 分层是按"这块像什么"分的, 白底会跟着衣服一起被分进 topwear/arm_upper 里 ——
  尤其是手臂和身体之间那道缝、腿之间、烟雾围出来的洞里。原图背景已经是透明的,
  所以 build_from_seethrough 里那套"底色贴边才删"的规则管不到它们(它们被角色包在
  中间, 不碰边界), 于是导出后角色腋下/烟雾里挂着一块块死白。

为什么在**画布**上算, 而不是逐部件算:
  同一块白往往被劈在好几个部件里(topwear / arm_upper_L / head 各有一份), 逐部件
  判断时白块不完整, 边缘那圈抗锯齿浅灰还会漏在别的部件里 —— 合成后就是洞周围
  一道白边。所以先按 coords 把部件拼成整张画布, 在画布上把"白底 + 它外围的浅灰边"
  一次算成一个 mask, 再回切到每个部件。

判据(很硬, 不会误伤):
  1) 三个通道都 >= 252 且 alpha > 200 —— 这套画的白色高光都是带灰的(<=250);
  2) 连通域面积 >= 40, 且内颜色标准差 < 2 —— 背景是死平的一片, 高光有渐变;
  3) 从这些"死白"往外扩 3 圈, 只吃 luma>=190 且 sat<=45 的像素(烟雾是紫的,
     饱和度会挡住), 把抗锯齿白边一起带走。
"""
import json
import os
import sys

import numpy as np
import cv2
from PIL import Image

PURE = 252
ALPHA_ON = 200
MIN_AREA = 40
FRINGE_LUM = 190
FRINGE_SAT = 45
FRINGE_RINGS = 3
OUTLINE_DIST = 5.0   # 距外轮廓小于这个的白 = 画的白描边, 不动
OUTLINE_DIST = 5.0   # 距外轮廓小于这个的白 = 画的白描边, 不动


def tight_bbox(im):
    a = im[..., 3] > 8
    if not a.any():
        return None
    ys, xs = np.nonzero(a)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def grid_for(w, h):
    g = 3
    if w * h > 20000:
        g = 4
    if w * h > 60000:
        g = 6
    return [g, g]


def canvas_white_mask(canvas):
    """在整张画布上算出"被角色包住的白底 + 它外围的白边"的 mask。

    关键: **只删离外轮廓足够远的白**。
    这套画法本身带白描边(剑刃、头巾顶边都是纯白的一圈), 那圈白离透明区只有 2~3px;
    真正要删的白底是"烟雾/手臂/腿围出来的洞", 离透明区都在 8px 以上。
    实测: 董卓的白洞 min 距离 8.0 / 17.6 / 18.6, 华雄剑刃 2.2, 法术甲头巾 2.0 ——
    用 5px 一刀切, 两边分得很干净。
    """
    rgb = canvas[..., :3].astype(np.int16)
    a = canvas[..., 3]
    pure = (rgb[..., 0] >= PURE) & (rgb[..., 1] >= PURE) & (rgb[..., 2] >= PURE) \
        & (a > ALPHA_ON)
    if not pure.any():
        return None

    clear = a <= 12
    dist = cv2.distanceTransform((~clear).astype(np.uint8), cv2.DIST_L2, 5)

    n, lab, st, _ = cv2.connectedComponentsWithStats(pure.astype(np.uint8), 8)
    kill = np.zeros(pure.shape, bool)
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) < MIN_AREA:
            continue
        comp = lab == i
        vals = canvas[comp][:, :3].astype(np.float32)
        if vals.std() >= 2.0:            # 有渐变 -> 角色的高光
            continue
        if dist[comp].min() < OUTLINE_DIST:   # 贴着外轮廓 -> 是画的白描边
            continue
        kill |= comp
    if not kill.any():
        return None

    lum = canvas[..., :3].mean(2)
    sat = canvas[..., :3].max(2) - canvas[..., :3].min(2)
    fringe = (a > 0) & (lum >= FRINGE_LUM) & (sat <= FRINGE_SAT) \
        & (dist >= OUTLINE_DIST - 2)
    for _ in range(FRINGE_RINGS):
        ring = cv2.dilate(kill.astype(np.uint8),
                          np.ones((3, 3), np.uint8)).astype(bool) & ~kill
        add = ring & fringe
        if not add.any():
            break
        kill |= add
    return kill



def source_white_mask(src_path, W, H, outline_dist=4.0, grow=2):
    """用**原图**算"残留白底"的 mask(画布坐标)。

    为什么用原图: 分层把那块白劈到好几个部件里, 有的部件还把它涂成了灰填充
    (画布合成出来既不是白也不是洞) —— 只看部件会漏。原图里它是完整一块纯白,
    位置/形状都最准。
    """
    src = Image.open(src_path).convert("RGBA").resize((W, H), Image.NEAREST)
    a = np.array(src).astype(int)
    al = a[..., 3]
    rgb = a[..., :3]
    pure = ((rgb >= 250).all(2)) & (al > 12)
    if not pure.any():
        return None
    clear = al <= 12
    dist = cv2.distanceTransform((~clear).astype(np.uint8), cv2.DIST_L2, 5)
    n, lab, st, _ = cv2.connectedComponentsWithStats(pure.astype(np.uint8), 8)
    kill = np.zeros(pure.shape, bool)
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) < MIN_AREA:
            continue
        comp = lab == i
        if a[comp][:, :3].astype(np.float32).std() >= 2.0:
            continue
        if dist[comp].min() < outline_dist:   # 贴着剪影 -> 是画的白描边
            continue
        kill |= comp
    if not kill.any():
        return None
    if grow:
        kill = cv2.dilate(kill.astype(np.uint8),
                          np.ones((3, 3), np.uint8), iterations=grow).astype(bool)
    return kill

def clean_dir(work, dry=False, src_img=None):
    cpath = os.path.join(work, "coords.json")
    coords = json.load(open(cpath, encoding="utf-8"))
    parts = coords["parts"]
    W, H = coords["canvas"]["w"], coords["canvas"]["h"]

    imgs = {}
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for p in parts:
        fp = os.path.join(work, "parts", p["image"])
        if not os.path.isfile(fp):
            continue
        im = Image.open(fp).convert("RGBA")
        imgs[p["name"]] = im
        canvas.alpha_composite(im, (p["x"], p["y"]))

    mask = None
    if src_img and os.path.isfile(src_img):
        mask = source_white_mask(src_img, W, H)
        if mask is not None:
            print(f"   按原图算白底: {int(mask.sum())}px")
    if mask is None:
        mask = canvas_white_mask(np.array(canvas))
    if mask is None:
        print("   没有纯白残留")
        return 0
    total = int(mask.sum())
    print(f"   画布上找到白底 {total}px")
    if dry:
        vis = Image.fromarray((mask * 255).astype(np.uint8))
        vis.save(os.path.join(work, "_white_mask.png"))
        return total

    nfix = 0
    for p in parts:
        im = imgs.get(p["name"])
        if im is None:
            continue
        a = np.array(im)
        px, py = p["x"], p["y"]
        h, w = a.shape[0], a.shape[1]
        sub = mask[py:py + h, px:px + w]
        if sub.shape != (h, w) or not sub.any():
            continue
        n = int((sub & (a[..., 3] > 0)).sum())
        if n == 0:
            continue
        a[..., 3] = np.where(sub, 0, a[..., 3])
        nb = tight_bbox(a)
        if nb is None:
            print(f"   {p['name']:16s} 整块被清空, 跳过")
            continue
        x0, y0, x1, y1 = nb
        Image.fromarray(a[y0:y1, x0:x1]).save(
            os.path.join(work, "parts", p["image"]))
        p.update({"x": int(px + x0), "y": int(py + y0),
                  "width": x1 - x0, "height": y1 - y0,
                  "grid": grid_for(x1 - x0, y1 - y0)})
        nfix += n
        print(f"   {p['name']:16s} -{n:5d}px -> {x1 - x0}x{y1 - y0} "
              f"@({p['x']},{p['y']})")

    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(coords, f, ensure_ascii=False, indent=2)
    return nfix


if __name__ == "__main__":
    ap = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry" in sys.argv[1:]
    src = ""
    if "--src" in sys.argv:
        src = sys.argv[sys.argv.index("--src") + 1]
        ap = [a for a in ap if a != src]
    for w in ap:
        print(f"== {w}")
        n = clean_dir(w, dry=dry, src_img=src)
        print(f"   合计 {n}px")


def bleed_colors(im, rounds=8):
    """把不透明像素的颜色"渗"进透明区(RGB 扩散, alpha 不变)。

    导出/渲染时贴图会被双线性采样: 边缘那圈透明像素如果 RGB 还是白的
    (从白底抠出来的图几乎都是), 采样就会把白带进边缘, 画面上一圈白边/白点。
    这里只改 RGB, 不动 alpha, 所以形状一点没变。
    """
    a = im[..., 3]
    rgb = im[..., :3].astype(np.float32)
    known = a > 0
    if known.all():
        return im
    for _ in range(rounds):
        w = known.astype(np.float32)
        num = cv2.boxFilter(rgb * w[..., None], -1, (3, 3), normalize=False)
        den = cv2.boxFilter(w, -1, (3, 3), normalize=False)
        add = (~known) & (den > 0)
        if not add.any():
            break
        rgb[add] = num[add] / den[add][:, None]
        known = known | add
    out = im.copy()
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def bleed_dir(work, rounds=8):
    cpath = os.path.join(work, "coords.json")
    coords = json.load(open(cpath, encoding="utf-8"))
    n = 0
    for p in coords["parts"]:
        fp = os.path.join(work, "parts", p["image"])
        if not os.path.isfile(fp):
            continue
        im = np.array(Image.open(fp).convert("RGBA"))
        out = bleed_colors(im, rounds)
        if np.array_equal(out, im):
            continue
        Image.fromarray(out).save(fp)
        n += 1
    return n


def decontaminate_edges(im, rounds=2, diff=60, min_lum=140):
    """把轮廓上残留的"白边像素"染成它内侧的颜色。

    白底抠图会在剪影边缘留一圈"半白"的抗锯齿像素: 它们本身不透明、比内侧亮一大截、
    且孤立地贴在透明边上。渲染出来就是一圈点状白边。
    判据只认"比内侧亮 60 以上且低饱和", 所以:
      - 剑刃/头巾那种**连成片的**白描边, 内侧也是白的, 差值小 -> 不动;
      - 盔甲亮边同理, 不动;
      - 只有"外面白、里面暗"的那一圈会被替换。
    只改 RGB, alpha 不动, 形状不变。
    """
    a = im[..., 3]
    rgb = im[..., :3].astype(np.float32)
    for _ in range(rounds):
        clear = a <= 12
        if not clear.any():
            break
        edge = (a > 12) & cv2.dilate(clear.astype(np.uint8),
                                     np.ones((3, 3), np.uint8)).astype(bool)
        inner = (a > 12) & ~edge
        if not inner.any() or not edge.any():
            break
        w = inner.astype(np.float32)
        num = cv2.boxFilter(rgb * w[..., None], -1, (3, 3), normalize=False)
        den = cv2.boxFilter(w, -1, (3, 3), normalize=False)
        ok = edge & (den > 0)
        if not ok.any():
            break
        near = num[ok] / den[ok][:, None]
        cur = rgb[ok]
        lum_c = cur.mean(1)
        lum_n = near.mean(1)
        sat_c = cur.max(1) - cur.min(1)
        hit = (lum_c - lum_n > diff) & (lum_c >= min_lum) & (sat_c <= 60)
        if not hit.any():
            break
        idx = np.nonzero(ok)
        sel = (idx[0][hit], idx[1][hit])
        rgb[sel] = near[hit]
    out = im.copy()
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def decon_dir(work, rounds=2):
    cpath = os.path.join(work, "coords.json")
    coords = json.load(open(cpath, encoding="utf-8"))
    n = 0
    for p in coords["parts"]:
        fp = os.path.join(work, "parts", p["image"])
        if not os.path.isfile(fp):
            continue
        im = np.array(Image.open(fp).convert("RGBA"))
        out = decontaminate_edges(im, rounds)
        if np.array_equal(out, im):
            continue
        Image.fromarray(out).save(fp)
        n += 1
    return n
