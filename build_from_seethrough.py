#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_from_seethrough.py

把 see-through 输出的 768x768 分层 PNG 反映射回原图空间, 紧裁成部件 PNG,
生成 video2spine 用的 parts/ 目录 + coords.json。

see-through 预处理: 原图(W,H) -> center_square_pad(S,S, S=max(W,H)) -> resize 768。
反向: 768 层 resize 回 S, 裁去 pad 还原 (W,H)。
  pad_x = (S - W)//2, pad_y = (S - H)//2
正方形输入(W==H): pad=0, 直接映射, 不裁切 (修复非视频/正方形素材变形)。
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

from arm_split import split_handwear
from leg_split import split_legs


def split_leg_halves(rgba, knee_ratio=0.45, overlap=4):
    """把一条腿在"膝高"处切成上/下两片(带 overlap 重叠), 用于刚性蒙皮也要屈膝。

    两片交界正好落在膝骨位置上: 上片跟髋骨、下片跟膝骨, 转动时以膝为铰链,
    接缝只在膝点处相连, 重叠几像素可以盖住张开的细缝。
    """
    a = rgba[..., 3] > 12
    ys = np.where(a.any(axis=1))[0]
    if len(ys) == 0:
        return [(rgba, "up")]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    yk = int(round(y0 + knee_ratio * (y1 - y0)))
    up = np.zeros_like(rgba)
    lo = np.zeros_like(rgba)
    u_end = min(y1, yk + overlap)
    l_beg = max(y0, yk - overlap)
    up[y0:u_end] = rgba[y0:u_end]
    lo[l_beg:y1] = rgba[l_beg:y1]
    return [(up, "up"), (lo, "low")]


def _split_feet_at_ankle(parts, out_dir, ankle_ratio=0.88):
    """把腿的下段按【踝线】切成 小腿 + 脚, 脚片 100% 绑 leg_?_foot。

    为什么必须切: 腿只有"上/下"两段时, 下段(膝~脚底)整块 100% 绑在膝骨上
    (bind_bone=leg_?_fore) —— 踝骨就成了彻底的空骨。gen_idle_bones 里明明给
    踝骨写了"迈步时脚尖下垂、落地踩平", 却没有任何像素跟着它转, 用户看到的就是
    "双脚末端没绑到脚上": 腿在迈, 靴子永远保持同一个角度。

    踝线取"腿顶往下 0.88 腿长", 与 gen_idle_bones 建 leg_?_foot 用的
    `topY - 0.88 * hgt` 完全一致 —— 切口正好落在踝骨原点, 脚片绕踝转才对。
    """
    out = []
    for p in parts:
        out.append(p)
        nm = str(p["name"])
        if not (nm.startswith("leg_") and nm.endswith("_low")):
            continue
        side = nm.split("_")[1]
        ups = [q for q in parts if q["name"] == f"leg_{side}_up"]
        lows = [q for q in parts if q["name"] == f"leg_{side}_low"]
        if not ups or not lows:
            continue
        top = min(q["y"] for q in ups)
        bot = max(q["y"] + q["height"] for q in lows)
        ay = int(round(top + ankle_ratio * (bot - top)))
        cut = ay - p["y"]
        if not (3 < cut < p["height"] - 3):
            continue                      # 脚太短(靴子并进小腿), 保持原样
        fp = os.path.join(out_dir, p["image"])
        im = Image.open(fp).convert("RGBA")
        calf = im.crop((0, 0, im.width, cut))
        # 脚片往上多吃 2px(overlap): 踝关节转起来时, 水平切口的两端会张开一个
        # 楔形缝(实测 11° 时 2.2px)。脚片在 z 上层, 多留的这 2px 正好把缝盖住。
        foot = im.crop((0, max(0, cut - 2), im.width, im.height))
        cols = np.nonzero((np.array(foot)[..., 3] > 8).any(0))[0]
        rows = np.nonzero((np.array(foot)[..., 3] > 8).any(1))[0]
        fx, fy = p["x"], ay
        if len(cols) and len(rows):
            cx0, cx1 = int(cols[0]), int(cols[-1]) + 1
            ry0, ry1 = int(rows[0]), int(rows[-1]) + 1
            foot = foot.crop((cx0, ry0, cx1, ry1))
            fx, fy = p["x"] + cx0, ay + ry0
        calf.save(fp)
        p["height"] = calf.height
        fname = f"foot_{side}"
        foot.save(os.path.join(out_dir, fname + ".png"))
        nfoot = {"name": fname, "image": fname + ".png",
                 "x": int(fx), "y": int(fy),
                 "width": foot.width, "height": foot.height,
                 "grid": grid_for(foot.width, foot.height),
                 "cls": "foot", "bind_bone": f"leg_{side}_foot"}
        if "_z_arm" in p:
            nfoot["_z_arm"] = p["_z_arm"]
        out.append(nfoot)
        print(f"  脚部拆分: {nm} -> 小腿 {calf.height}px + {fname} {foot.height}px"
              f"(100% 绑 leg_{side}_foot, 踝线 y={ay})")
    return out


def _strip_cloth_from_props(parts, out_dir):
    """把武器上夹带的"布料块"剔掉。

    `_cut_blades` 按连通域从 topwear 里切刀, 但披风/裙摆的一角常常和刀挨得很近,
    连膨胀后就被并进同一块 —— 用户看到的是"武器上附带了一部分衣服"。
    布料是**粉紫色**(R 明显高于 G, B 也高于 G), 刀是黑/银(低饱和), 按这个分得开。
    """
    dropped = 0
    for p in parts:
        if p.get("cls") != "prop":
            continue
        fp = os.path.join(out_dir, p["image"])
        if not os.path.isfile(fp):
            continue
        im = np.array(Image.open(fp).convert("RGBA")).copy()
        m = (im[..., 3] > 12).astype(np.uint8)
        if not m.any():
            continue
        k, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        hit = False
        for i in range(1, k):
            if stats[i, cv2.CC_STAT_AREA] < 80:
                continue
            rgb = im[..., :3][lab == i].astype(int)
            r, g, b = rgb[:, 0].mean(), rgb[:, 1].mean(), rgb[:, 2].mean()
            if (r - g) > 10 and (b - g) > 10:
                im[lab == i, 3] = 0
                dropped += 1
                hit = True
        if hit:
            a = im[..., 3]
            ys, xs = np.nonzero(a > 10)
            crop = Image.fromarray(im).crop((int(xs.min()), int(ys.min()),
                                             int(xs.max()) + 1, int(ys.max()) + 1))
            crop.save(fp)
            p.update({"x": p["x"] + int(xs.min()), "y": p["y"] + int(ys.min()),
                      "width": crop.width, "height": crop.height,
                      "grid": grid_for(crop.width, crop.height)})
    if dropped:
        print(f"  武器剔除布料块: {dropped} 处(粉紫色, 不是刀)")
    return parts


def _back_layer_to_bottom(parts):
    """披风/翅膀(cls=="body", 名字含 wing/back/cape/cloak)必须排在最底层。

    它们在角色【背后】。之前为了消掉"头盔与披风之间的硬边"把 wings 提到 topwear
    之上, 结果是披风盖在胸甲前面 —— 层序反了。硬边要靠"把 head 挖出身体层"解决,
    不能靠把背后的东西提到前面。
    """
    keys = ("wing", "back", "cape", "cloak", "beiying")
    back = [p for p in parts
            if p.get("cls") == "body" and any(k in str(p["name"]).lower() for k in keys)]
    if not back or all(parts.index(q) < min(parts.index(r) for r in parts if r not in back)
                       for q in back):
        return parts
    rest = [p for p in parts if p not in back]
    if back != parts[:len(back)]:
        print(f"  披风/翅膀放回最底层: {', '.join(q['name'] for q in back)}")
    return back + rest


def _carve_head_feet_from_body(parts, out_dir, cw, ch, carve_head=False):
    """把"被包在身体层里"的头和脚挖出来, 让它们各自成件。

    see-through 遇到"头被头盔整个包住""脚被胫甲/靴筒包住"的角色时, 会把整颗头
    (含头盔、角)和整只脚判成 body 的一部分吐进 topwear。后果:
      - 头挂在躯干骨上: 转头时头不动, 只转一小块脸皮;
      - 脚挂在躯干骨上: 迈步时脚像钉在地上, 只有小腿在动(用户: "身体包含了脚部");
      - 脚还会**横向超出腿的 bbox**(魔神左脚在 x111~148, 而 leg_low 只有 120~170),
        切腿时按腿的范围切, 超出那截就永远留在身体层里。
    判据: topwear 在"头邻域"或"踝线以下的腿邻域"里还有足够多的不透明像素。
    """
    tw = next((p for p in parts if p["name"] in ("topwear", "body")), None)
    if tw is None:
        return parts
    arr = np.array(Image.open(os.path.join(out_dir, tw["image"])).convert("RGBA"))
    work = arr.copy()

    def take(box, need=80):
        x0, y0, x1, y1 = [int(v) for v in box]
        sx0, sy0 = max(0, x0 - tw["x"]), max(0, y0 - tw["y"])
        sx1 = min(work.shape[1], x1 - tw["x"])
        sy1 = min(work.shape[0], y1 - tw["y"])
        if sx1 <= sx0 or sy1 <= sy0:
            return None
        sub = work[sy0:sy1, sx0:sx1].copy()
        if int((sub[..., 3] > 12).sum()) < need:
            return None
        return sub, tw["x"] + sx0, tw["y"] + sy0

    def merge_into(part, got, clear=True):
        """把 take 出来的块并进 part, 取并集 bbox 写回, 并从 topwear 挖掉。"""
        nonlocal work
        sub, gx, gy = got
        cv = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        cv.alpha_composite(
            Image.open(os.path.join(out_dir, part["image"])).convert("RGBA"),
            (part["x"], part["y"]))
        cv.alpha_composite(Image.fromarray(sub), (gx, gy))
        a = np.array(cv)[..., 3]
        ys, xs = np.nonzero(a > 10)
        bx0, by0 = int(xs.min()), int(ys.min())
        crop = cv.crop((bx0, by0, int(xs.max()) + 1, int(ys.max()) + 1))
        crop.save(os.path.join(out_dir, part["image"]))
        part.update({"x": bx0, "y": by0,
                     "width": crop.width, "height": crop.height,
                     "grid": grid_for(crop.width, crop.height)})
        if clear:
            work[gy - tw["y"]:gy - tw["y"] + sub.shape[0],
                 gx - tw["x"]:gx - tw["x"] + sub.shape[1], 3] = 0
        return crop.size

    added = []
    # ---- 头: 只有"小脸"才认为头没拆出来(大整头说明已经重铺过了) ----
    hd = next((p for p in parts if p["name"] == "head"), None)
    if not carve_head:
        pass      # 默认不做"从身体层挖头": 见下方说明
    elif hd and hd["width"] * hd["height"] < 2400:
        # 先验一件事: 头的框里 topwear 到底占了多少。
        # 头真的还埋在身体层里 -> 这块应该是一片实心(头盔/脸都在 topwear);
        # 头已经单独拆出来了 -> 这块只剩背景或披风边缘, 填充率很低。
        # 以前不看这个, 只要"脸小"就往 2.5 倍脸宽的大方框里抓, 结果把两侧肩甲和
        # 整个胸甲都并进 head(董卓 144x144、吕布 81x105 就是这么来的)。
        ix0 = max(0, int(hd["x"]) - tw["x"])
        iy0 = max(0, int(hd["y"]) - tw["y"])
        ix1 = min(arr.shape[1], int(hd["x"] + hd["width"]) - tw["x"])
        iy1 = min(arr.shape[0], int(hd["y"] + hd["height"]) - tw["y"])
        fill = 0.0
        if ix1 > ix0 and iy1 > iy0:
            fill = float((arr[iy0:iy1, ix0:ix1, 3] > 12).mean())
        if fill < 0.35:
            print(f"  头部不再从身体层挖: 头框内 topwear 只占 {fill * 100:.0f}%"
                  f"(头已经独立成件)")
        else:
            # 上 2 倍脸高: 角比脸高出近两倍。下 1.5 倍: 到下巴+脖子, 再多就带进肩甲
            box = (hd["x"] - 1.1 * hd["width"], hd["y"] - 2.0 * hd["height"],
                   hd["x"] + 2.5 * hd["width"], hd["y"] + 2.5 * hd["height"])
            got = take(box, need=400)
            if got:
                sz = merge_into(hd, got)
                print(f"  头部从身体层挖出: head {sz[0]}x{sz[1]}"
                  f"(脸只占 {hd['width'] * hd['height']}px, 头和头盔原来在 topwear 里)")
    # ---- 脚: 从身体层挖回来, 但**并进腿**, 不单独成件 ----
    # 单独切一块 foot_? 出来试过三轮, 每次都是同一个病根: 靴子是**一整块**造型,
    # 在踝线横切一刀, 上下两块各自刚性旋转, 抬脚时切缝张开 / 靴尖被甩出去
    # (用户: "足部切图不完整, 且未绑定到腿部")。整条腿 + 顶点权重渐变
    # (髋->膝->踝)本来就能让靴尖跟踝骨, 所以脚的像素直接并进腿里。
    for side in ("L", "R"):
        cand = [q for q in parts
                if q.get("cls") == "leg" and str(q["name"]).startswith(f"leg_{side}")]
        if not cand:
            continue
        p = next((q for q in cand if str(q["name"]).endswith("_low")), cand[0])
        # 踝线在小腿高的 0.72 处起(往上留一点余量, 中间像素重合无害);
        # 横向放宽 40% 兜住"超出腿 bbox 的靴尖"
        box = (p["x"] - 0.4 * p["width"], p["y"] + 0.72 * p["height"],
               p["x"] + 1.4 * p["width"], p["y"] + p["height"] + 22)
        got = take(box, need=200)
        if not got:
            continue
        sz = merge_into(p, got)
        print(f"  脚部并回腿: {p['name']} {sz[0]}x{sz[1]}"
              f"(踝线以下原来在 topwear 里, 单独成件会被切缝撕开)")
    if added:
        parts.extend(added)
    if (np.array(work)[..., 3] > 12).sum() != int((arr[..., 3] > 12).sum()):
        # 只把挖掉后的图写回, 不动它的 x/y/尺寸 —— 图像本身没裁, 坐标还是原来的
        Image.fromarray(work).save(os.path.join(out_dir, tw["image"]))
    return parts


def _rescue_crushed_headwear(parts, out_dir):
    """头饰被拆碎时, 从"后发/披风"那层把头救回来。

    模型的 back hair 层经常把整个头部(兜帽 + 两只角 + 后颈)一起吐出来, 于是:
      - headwear 只剩几片盔沿碎屑(填充率十几%) —— 用户看到"头盔的分解不正确";
      - 真正的兜帽被当成"翅膀/披风"排到最底层(z 最低), 被衣服盖住 ——
        用户看到"头盔被遮盖"。
    判据: headwear 不透明像素 / 外框面积 < 0.30(碎片), 且有一个 body 类部件
    盖住了头部区域。这时以 head 为锚点, 从那个部件里裁一块"头部邻域"并进 headwear。

    区域按 [脸] 的尺寸外扩(实测魔神: 脸 21x13 -> 需要 72x59 才包住角和兜帽):
      左 1.2 倍脸宽、右 1.7 倍(角多在一侧)、上 2 倍脸高、下 2.5 倍脸高。
    原部件**不做挖洞** —— 头只转几度, 重合部分被 headwear 盖着看不出来;
    硬挖会留下矩形缺口, 头一转就露馅。
    """
    hws = [p for p in parts if p.get("cls") == "headgear"]
    hd = next((p for p in parts if p["name"] == "head"), None)
    if not hws or not hd:
        return parts
    tot = 0.0
    for q in hws:
        f = os.path.join(out_dir, q["image"])
        if not os.path.isfile(f):
            return parts
        a = np.array(Image.open(f).convert("RGBA"))[..., 3]
        tot += (a > 12).sum() / max(1.0, q["width"] * q["height"])
    fill = tot / len(hws)
    if fill >= 0.30:
        return parts                     # 正常头盔, 不用救
    hx, hy, hw_, hh_ = hd["x"], hd["y"], hd["width"], hd["height"]
    rx0 = int(hx - 1.2 * hw_)
    rx1 = int(hx + hw_ + 1.7 * hw_)
    ry0 = int(hy - 2.0 * hh_)
    # 下沿只到下巴下方一点点: 再往下就是胸前的披风, 带进来会跟着头一起转
    ry1 = int(hy + hh_ + max(8.0, 0.55 * hh_))
    # 找"盖住这块区域最多"的躯干/披风类部件
    best, best_n = None, 0
    for q in parts:
        if q in hws or q["name"] == "head" or q.get("cls") in ("leg", "arm_upper",
                                                               "arm_fore", "arm_hand",
                                                               "foot", "prop"):
            continue
        f = os.path.join(out_dir, q["image"])
        if not os.path.isfile(f):
            continue
        im = np.array(Image.open(f).convert("RGBA"))[..., 3]
        sx0, sy0 = max(0, rx0 - q["x"]), max(0, ry0 - q["y"])
        sx1 = min(im.shape[1], rx1 - q["x"])
        sy1 = min(im.shape[0], ry1 - q["y"])
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        n = int((im[sy0:sy1, sx0:sx1] > 12).sum())
        if n > best_n:
            best, best_n = q, n
    if best is None or best_n < 0.25 * (rx1 - rx0) * (ry1 - ry0):
        return parts
    # 裁这块, 并与现有头饰合并(切片要 clamp 到源图内 —— 头部区域常常超出部件上沿)
    src = np.array(Image.open(os.path.join(out_dir, best["image"])).convert("RGBA"))
    yy0 = max(0, ry0 - best["y"])
    yy1 = min(src.shape[0], ry1 - best["y"])
    xx0 = max(0, rx0 - best["x"])
    xx1 = min(src.shape[1], rx1 - best["x"])
    sub = src[yy0:yy1, xx0:xx1].copy()
    px, py = best["x"] + xx0, best["y"] + yy0
    if sub.size == 0:
        return parts
    canvas = Image.new("RGBA", (int(max(parts[0]["x"] + 400, rx1 + 40)),
                                int(max(parts[0]["y"] + 400, ry1 + 40))), (0, 0, 0, 0))
    for q in hws:
        canvas.alpha_composite(
            Image.open(os.path.join(out_dir, q["image"])).convert("RGBA"),
            (q["x"], q["y"]))
    canvas.alpha_composite(Image.fromarray(sub), (px, py))
    a = np.array(canvas)[..., 3]
    ys, xs = np.nonzero(a > 10)
    merged = canvas.crop((int(xs.min()), int(ys.min()),
                          int(xs.max()) + 1, int(ys.max()) + 1))
    # 下沿羽化: 兜帽和下面的披风是同色, 硬边会像"被切了一刀"
    mp = np.array(merged).astype(float)
    for i in range(5):
        mp[mp.shape[0] - 5 + i, :, 3] *= (1.0 - i / 4.0 * 0.85)
    merged = Image.fromarray(mp.clip(0, 255).astype(np.uint8))
    merged.save(os.path.join(out_dir, "headwear.png"))
    keep = hws[0]
    for q in hws[1:]:
        parts.remove(q)
    keep.update({"image": "headwear.png", "x": int(xs.min()), "y": int(ys.min()),
                 "width": merged.width, "height": merged.height,
                 "grid": grid_for(merged.width, merged.height)})
    print(f"  头饰救援: headwear 填充率仅 {fill * 100:.0f}%(碎片), "
          f"从 {best['name']} 裁出头部 {merged.width}x{merged.height} 并进头饰")
    return parts


def _face_over_headwear(parts, ratio=0.35):
    """小脸 + 大盔壳的角色: 把 headwear 排到 head 之前(头盔在外壳层, 脸压在上)。

    判据只看面积比: head / headwear < 0.35 说明"头饰是围着脸的框", 原画里脸是
    完整的一块、四周才是盔沿与角(魔神这类)。这时把头饰留在最上层就是"头盔盖脸",
    只能靠 punch_head_wear 把脸按头盔轮廓啃掉一圈, 露出来的反而是残脸。
    反过来 head 是大整头(华雄那种把头发/五官重铺进 head 的), 面积比会 Close to 1,
    保持头饰在上层才对 —— 头发不能跑到脸后面去。
    """
    hd = next((p for p in parts if p["name"] == "head"), None)
    hws = [p for p in parts if p.get("cls") == "headgear"]
    if not hd or not hws:
        return parts
    ha = hd["width"] * hd["height"]
    wa = sum(q["width"] * q["height"] for q in hws)
    if ha / max(1, wa) >= ratio:
        return parts
    if all(parts.index(q) > parts.index(hd) for q in hws):
        out = [q for q in parts if q not in hws]
        i = out.index(hd)
        for q in reversed(hws):
            out.insert(i, q)
        print(f"  头部 z 序: head 只有头饰面积的 {ha / max(1, wa) * 100:.0f}%, "
              f"按'脸在头盔开口里'处理 -> 头饰排到 head 之前(脸在上层)")
        return out
    return parts


# 会被"脸窗"检验的五官槽位(neckwear 在脖子下面, 不参与)
_FACE_WINDOW_SLOTS = ("face", "ears", "earwear", "eyewhite", "irides",
                      "eyelash", "eyebrow", "nose", "mouth", "eyewear")


def _canvas_mask(out_dir, p, W, H):
    """部件在整张画布上的不透明 mask。"""
    a = np.array(Image.open(os.path.join(out_dir, p["image"])).convert("RGBA"))[..., 3] > 8
    m = np.zeros((H, W), bool)
    x0, y0 = max(0, int(p["x"])), max(0, int(p["y"]))
    x1 = min(W, int(p["x"]) + a.shape[1])
    y1 = min(H, int(p["y"]) + a.shape[0])
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = a[y0 - p["y"]:y1 - p["y"], x0 - p["x"]:x1 - p["x"]]
    return m


def _put_pixels(out_dir, p, add, src_rgba, W, H, keep=None):
    """把 add 位置的源图像素写进部件; keep 非空时只保留原内容里的这部分。"""
    old = np.zeros((H, W, 4), np.uint8)
    a = np.array(Image.open(os.path.join(out_dir, p["image"])).convert("RGBA"))
    x0, y0 = max(0, int(p["x"])), max(0, int(p["y"]))
    x1, y1 = min(W, x0 + a.shape[1]), min(H, y0 + a.shape[0])
    old[y0:y1, x0:x1] = a[y0 - p["y"]:y1 - p["y"], x0 - p["x"]:x1 - p["x"]]
    if keep is not None:
        old[..., 3][~keep] = 0
    content = (old[..., 3] > 8) | add
    if not content.any():
        return
    ys, xs = np.nonzero(content)
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    buf = old[by0:by1 + 1, bx0:bx1 + 1].copy()
    sub = add[by0:by1 + 1, bx0:bx1 + 1]
    buf[sub] = src_rgba[by0:by1 + 1, bx0:bx1 + 1][sub]
    Image.fromarray(buf).save(os.path.join(out_dir, p["image"]))
    p.update({"x": bx0, "y": by0,
              "width": int(bx1 - bx0 + 1), "height": int(by1 - by0 + 1),
              "grid": grid_for(int(bx1 - bx0 + 1), int(by1 - by0 + 1))})


def _give_to_nearest(parts, out_dir, px_mask, src_rgba, W, H, skip=()):
    """把 px_mask 里的像素按"离哪个部件最近"写进那个部件。"""
    if not px_mask.any():
        return
    from scipy import ndimage as ndi
    cand = [q for q in parts if q["name"] not in skip]
    if not cand:
        return
    lab = np.zeros((H, W), np.int32)
    for i, q in enumerate(cand, 1):
        lab[_canvas_mask(out_dir, q, W, H)] = i
    if not (lab > 0).any():
        return
    _d, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    ys, xs = np.nonzero(px_mask)
    owner = lab[iy[ys, xs], ix[ys, xs]]
    for i, q in enumerate(cand, 1):
        sel = owner == i
        if not sel.any():
            continue
        add = np.zeros((H, W), bool)
        add[ys[sel], xs[sel]] = True
        _put_pixels(out_dir, q, add, src_rgba, W, H)


def _repair_face_split(parts, out_dir, layers_dir, src_rgba, W, H, frame=None):
    """五官层错位时把它丢掉, 必要时按 AI head 层重建整颗头。

    起因(吕布这张图): see-through 把脸拆成一堆错位的肤色块 —— ears 落在额头上、
    nose 盖住盔顶、eyelash/eyebrow 跑到发髻上; 而 head 层本身是完整的一颗头。
    直接信这些层, 拆出来的 head 会被"从身体层挖头"那一步撑成"头盔 + 两肩 + 胸甲",
    五官碎块则是一堆肉色椭圆 —— 用户看到的就是"头面部拆解有问题"。

    判据: 定义"脸窗" = head 外框横向内缩 12%、纵向 25%~95%。真五官一定落在里面
    (实测董卓/华雄/魔神/皮甲/全甲/suo/法术甲 最低也有 33%, 耳朵在头两侧所以最低);
    错位的碎块是 0%~8%。低于 20% 就判为错位。
    两个以上错位 -> 认为这一路 AI 分层整体不可信, 用 AI head 层重铺整颗头,
    多出来的(肩甲/胸甲)还给身体。
    """
    hd = next((p for p in parts if p["name"] == "head"), None)
    if hd is None or src_rgba is None:
        return parts
    hm = _canvas_mask(out_dir, hd, W, H)
    if not hm.any():
        return parts
    ys, xs = np.nonzero(hm)
    hy0, hy1, hx0, hx1 = ys.min(), ys.max(), xs.min(), xs.max()
    hh = max(1, hy1 - hy0 + 1)
    hw = max(1, hx1 - hx0 + 1)
    wx0, wx1 = hx0 + 0.12 * hw, hx1 - 0.12 * hw
    wy0, wy1 = hy0 + 0.25 * hh, hy0 + 0.95 * hh

    bad = []
    for p in list(parts):
        if p["name"] not in _FACE_WINDOW_SLOTS:
            continue
        m = _canvas_mask(out_dir, p, W, H)
        n = int(m.sum())
        if n < 8:
            continue
        py, px = np.nonzero(m)
        share = float(((px >= wx0) & (px <= wx1) &
                       (py >= wy0) & (py <= wy1)).mean())
        if share < 0.20:
            bad.append((p, share))
    if not bad:
        return parts

    names = [p["name"] for p, _s in bad]
    print("  五官层错位: " + ", ".join(
        f"{p['name']}(脸窗内 {s * 100:.0f}%)" for p, s in bad) +
        " -> 丢弃, 像素归头/身体")
    drop = set(names)
    parts = [p for p in parts if p["name"] not in drop]
    for nm in names:
        f = os.path.join(out_dir, nm + ".png")
        if os.path.isfile(f):
            os.remove(f)

    # 两个以上错位: AI 的 head/face 那一路整体不可信 -> 按 AI head 层重建整颗头
    rebuilt = False
    if len(bad) >= 2 and layers_dir:
        hp = os.path.join(layers_dir, "head.png")
        if os.path.isfile(hp):
            try:
                from scipy import ndimage as ndi
                if frame:
                    S, pad_x, pad_y = frame
                    ai = map_to_frame(hp, W, H, S, pad_x, pad_y)[..., 3] > 40
                else:
                    ai = np.array(Image.open(hp).convert("RGBA")
                                  .resize((W, H), Image.LANCZOS))[..., 3] > 40
                ai = ndi.binary_dilation(ai, np.ones((3, 3)), iterations=2)
                new_h = ai & (src_rgba[..., 3] > 8)
                # 只在"确实把身体吃进去了"时才重建: 新头要更小(但不至于碎成一片),
                # 太小说明 AI head 层本身不可靠, 保持原样
                if 250 <= new_h.sum() < 0.95 * hm.sum():
                    extra = hm & ~new_h
                    # 头里多出来的就是肩甲/胸甲/领口, 一律还给躯干。
                    # 不能按"最近部件"分: 肩甲离大臂比离躯干近, 会被劈给手臂,
                    # 一摆臂就把胸口的甲片带着甩出去。
                    body = next((q for q in parts
                                 if q["name"] in ("topwear", "body")), None)
                    if body is not None and extra.any():
                        _put_pixels(out_dir, body, extra, src_rgba, W, H)
                    else:
                        _give_to_nearest(parts, out_dir, extra, src_rgba, W, H,
                                         skip=("head",))
                    _put_pixels(out_dir, hd, new_h, src_rgba, W, H, keep=new_h)
                    # 五官这一路既然已经错位到这个程度, 剩下的五官层也不可信
                    # (吕布剩一个 mouth 是块粉色糊斑) -> 一并丢掉, 头做成整块
                    rest = [q["name"] for q in parts
                            if q["name"] in _FACE_WINDOW_SLOTS]
                    if rest:
                        parts = [q for q in parts if q["name"] not in rest]
                        for nm in rest:
                            f = os.path.join(out_dir, nm + ".png")
                            if os.path.isfile(f):
                                os.remove(f)
                        print("  五官并入整头: " + ", ".join(rest))
                    print(f"  头部重建: {int(hm.sum())}px -> {int(new_h.sum())}px"
                          f"(按 AI head 层, 不再含肩甲/胸甲), "
                          f"多余 {int(extra.sum())}px 还给身体")
                    rebuilt = True
            except Exception as ex:                    # noqa: BLE001
                print(f"  头部重建跳过: {ex}")

    # 收尾: 谁都没覆盖的像素按最近部件补回去(删部件/重建头都会漏几条缝)
    cov = np.zeros((H, W), np.int32)
    for q in parts:
        cov += _canvas_mask(out_dir, q, W, H)
    hole = (src_rgba[..., 3] > 8) & (cov == 0)
    if hole.any():
        _give_to_nearest(parts, out_dir, hole, src_rgba, W, H)
        print(f"  补缝: {int(hole.sum())}px")
    return parts


_REPAINT_SLOTS = ("head", "eyewhite", "irides", "eyelash", "eyebrow",
                  "nose", "mouth", "eyes_closed")


def _repaint_from_source(parts, out_dir, src_rgba, W, H, names=_REPAINT_SLOTS):
    """把指定部件的颜色换成原图的颜色, alpha 保持部件自己的轮廓。

    为什么需要: AI 分层在脸上经常是"糊的"——轮廓对, 但颜色被重建漂移了
    (董卓 head 与源图平均色差 35.8, 渲出来脸发白、眼睛被抹掉)。
    轮廓仍由 AI mask 决定(该挖空的还挖空), 颜色一律取原图, 这样脸是清晰的。
    """
    if src_rgba is None:
        return parts
    for p in parts:
        if p["name"] not in names:
            continue
        f = os.path.join(out_dir, p["image"])
        if not os.path.isfile(f):
            continue
        a = np.array(Image.open(f).convert("RGBA"))
        h, w = a.shape[:2]
        x0, y0 = max(0, int(p["x"])), max(0, int(p["y"]))
        x1, y1 = min(W, x0 + w), min(H, y0 + h)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = src_rgba[y0:y1, x0:x1]
        view = a[y0 - p["y"]:y1 - p["y"], x0 - p["x"]:x1 - p["x"]]
        m = (view[..., 3] > 0) & (sub[..., 3] > 8)
        if int(m.sum()) < 20:
            continue
        before = np.abs(view[..., :3][m].astype(int) - sub[..., :3][m]).mean()
        view[..., :3][m] = sub[..., :3][m]
        Image.fromarray(a).save(f)
        print(f"  按原图重铺颜色: {p['name']} ({int(m.sum())}px, "
              f"与原图色差 {before:.0f} -> 0)")
    return parts


def _clean_prop_junk(parts, out_dir, src_rgba, W, H):
    """武器/道具部件里混着的衣料、碎屑剔掉, 像素还给身体。

    起因(吕布): objects_L 里除了斧头还连着一条 307px 的深色布片(裙摆左边缘),
    被 AI 的 objects 层一起圈了进来。道具是跟手走的, 这块布会跟着斧头飞。
    判据: 道具部件按连通块拆开, 只有一个主块(其余块面积 < 主块 35%)时,
    把副块判为异物, 按最近部件归还。
    """
    if src_rgba is None:
        return parts
    from scipy import ndimage as ndi
    for p in list(parts):
        if p.get("cls") != "prop":
            continue
        a = np.array(Image.open(os.path.join(out_dir, p["image"])).convert("RGBA"))
        al = a[..., 3] > 8
        lab, n = ndi.label(al, np.ones((3, 3)))
        if n < 2:
            continue
        sizes = ndi.sum(al, lab, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        junk = al & (lab != keep)
        if not junk.any() or junk.sum() > 0.35 * sizes[keep - 1]:
            continue
        canvas = np.zeros((H, W), bool)
        x0, y0 = max(0, int(p["x"])), max(0, int(p["y"]))
        x1 = min(W, x0 + a.shape[1])
        y1 = min(H, y0 + a.shape[0])
        canvas[y0:y1, x0:x1] = junk[y0 - p["y"]:y1 - p["y"], x0 - p["x"]:x1 - p["x"]]
        _give_to_nearest(parts, out_dir, canvas, src_rgba, W, H,
                         skip=(p["name"],))
        # 只保留主块: 副块像素已经给了别的部件
        buf = a.copy()
        buf[..., 3][junk] = 0
        ys, xs = np.nonzero(buf[..., 3] > 8)
        bx0, by0 = int(xs.min()), int(ys.min())
        crop = buf[by0:ys.max() + 1, bx0:xs.max() + 1]
        Image.fromarray(crop).save(os.path.join(out_dir, p["image"]))
        p.update({"x": int(p["x"] + bx0), "y": int(p["y"] + by0),
                  "width": int(crop.shape[1]), "height": int(crop.shape[0]),
                  "grid": grid_for(int(crop.shape[1]), int(crop.shape[0]))})
        print(f"  道具去杂: {p['name']} 剔出 {int(junk.sum())}px(非武器内容)")
    return parts


def _prune_headwear_junk(parts, out_dir, W, H):
    """头饰上的"细碎描边"剔掉。

    起因(吕布): AI 的 headwear 层除了冠和红缨, 还把整颗头的轮廓描边当成头饰圈了
    进来 —— 一圈 1~3px 宽的虚线弧, 放大看像一堆碎片挂在头周围。
    做法: 3x3 开运算只留"有厚度"的部分, 再膨胀 1px 与原件取交集(保住外形),
    最后按连通块面积过滤掉 <5% 的碎块。
    保护: 去掉超过 30% 说明这块头饰本身就是细条(飘带/薄纱), 不动它。
    """
    from scipy import ndimage as ndi
    for p in list(parts):
        if p.get("cls") not in ("cloth", "headgear"):
            continue
        f = os.path.join(out_dir, p["image"])
        if not os.path.isfile(f):
            continue
        a = np.array(Image.open(f).convert("RGBA"))
        al = a[..., 3] > 8
        total = int(al.sum())
        if total < 400:
            continue
        op = ndi.binary_opening(al, np.ones((3, 3)))
        kept = ndi.binary_dilation(op, np.ones((3, 3)), iterations=1) & al
        lab, n = ndi.label(kept, np.ones((3, 3)))
        if n:
            sizes = ndi.sum(kept, lab, range(1, n + 1))
            limit = max(40, 0.05 * kept.sum())
            clean = np.zeros_like(kept)
            for i in range(1, n + 1):
                if sizes[i - 1] >= limit:
                    clean |= lab == i
            kept = clean
        gone = total - int(kept.sum())
        if gone <= 0:
            continue
        if gone > 0.30 * total:
            print(f"  头饰去碎边: {p['name']} 会掉 {gone / total * 100:.0f}%, "
                  f"说明它本来就是细条, 跳过")
            continue
        buf = a.copy()
        buf[..., 3][~kept] = 0
        ys, xs = np.nonzero(buf[..., 3] > 8)
        if not len(xs):
            continue
        crop = buf[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        Image.fromarray(crop).save(f)
        p.update({"x": int(p["x"] + xs.min()), "y": int(p["y"] + ys.min()),
                  "width": int(crop.shape[1]), "height": int(crop.shape[0]),
                  "grid": grid_for(int(crop.shape[1]), int(crop.shape[0]))})
        print(f"  头饰去碎边: {p['name']} 去掉 {gone}px"
              f"({gone / total * 100:.0f}%) -> {crop.shape[1]}x{crop.shape[0]}")
    return parts


try:                      # 复用生成器的语义分类: 残影要挂到"对的那个骨"上
    from gen_idle2 import classify as _classify
except Exception:         # noqa: BLE001
    _classify = None

PARTS_SCHEMA = 3       # 拆图结果结构版本(1=最早 16 部件, 2=加腿/道具, 3=切片腿+头巾合并)
MIN_PIXELS = 200       # alpha 非零像素低于此值视为空层丢弃(整块部件)
PX_PER_CELL = 55

# 小特征槽位: 用更低的像素阈值, 否则"红瞳 141px"这种会被当成噪声扔掉
# 小特征的最小像素: 336 画布上眼睛/嘴可能只有 20~30px(董卓的嘴 21px),
# 门槛太高就被当噪声扔掉, 脸上会缺五官。
FEATURE_MIN = {
    "irides": 12, "nose": 15, "eyewhite": 15, "eyelash": 15, "eyebrow": 15,
    "mouth": 12, "ears": 40, "earwear": 30, "eyewear": 30, "neckwear": 30,
}

# 头发槽位关键字(无头发模式: 这些其实是头巾/头盔, 要刚性跟随 head)
_HAIR_NAME_KW = ("hair", "headwear", "toushi", "maozi", "hat", "headgear")
# 无头发模式下会被合并成一整块"头饰"的槽位
_HEAD_WEAR_SLOTS = ("back hair", "front hair", "headwear")
# 五官/耳朵这些"长在脸上"的小部件: 模型的前发 mask 常常把它们的像素也圈进去
# (实测嘴巴 77%、耳朵 54% 落在头巾里), 合并头饰时必须把这部分剔掉,
# 否则头巾一飘就带着嘴和耳朵一起动。
_FACE_FEATURE_SLOTS = ("eyewhite", "irides", "eyelash", "eyebrow", "nose", "mouth", "ears")

# 画布 back->front 绘制顺序 (slots 顺序)
# 必须覆盖模型 VALID_BODY_PARTS_V2 的全部 slot —— 漏一个就等于把该部位丢掉。
# 早期版本只列了 17 个, 漏掉 legwear/bottomwear/footwear/neckwear/tail/wings/objects,
# 结果拆出来的角色【没有腿】。
# 顺序: 腿/裤/鞋在躯干之前(腰甲、衣摆会盖住裤腰), 道具画在手前面, 手在最后(压住刀柄)。
Z_ORDER = [
    "back hair", "tail", "wings",
    "legwear", "bottomwear", "footwear",
    "neck", "topwear", "neckwear",
    "head", "ears", "earwear",
    "face", "eyewhite", "irides", "eyelash", "eyebrow", "nose", "mouth",
    "eyewear", "front hair", "headwear",
    "objects",       # 手持道具(刀/盾等)
    "handwear",
]

# 最终 z 序(数值小=更靠后/更先画)。按名字前缀匹配, xxx_overlay 自动继承 xxxx 的层。
# 关键: 大臂(15)排在躯干(21)之前 —— 手臂是被护肩、衣袖盖住的, 不是盖在盔甲上面;
#      小臂/手(80/81)排在最后 —— 手要压在裙甲、道具之上(握着刀), 否则手会钻进衣摆里。
Z_RANK = [
    ("back hair", 0), ("tail", 3), ("wings", 4),
    ("legwear", 10), ("bottomwear", 11), ("footwear", 12), ("leg", 10),
    ("arm_upper", 15),
    ("neck", 20), ("topwear", 21), ("neckwear", 25),
    ("head", 30), ("ears", 30), ("earwear", 30), ("face", 30),
    ("eyewhite", 34), ("irides", 35), ("eyelash", 36), ("eyes_closed", 36.5),
    ("eyebrow", 37),
    ("nose", 38), ("mouth", 39), ("eyewear", 40),
    ("front hair", 45), ("headscarf", 44),   # 头巾压在脸上、但被头盔压住
    ("headwear", 95),   # 头盔永远在最上层: 人身上没有东西盖住它
    ("objects", 70), ("residual", 70),
    ("arm_fore", 80), ("arm_hand", 81),
]


def strip_bg(rgba, tol=24):
    """原图没有透明通道(纯色背景)时, 从四边泛洪抠掉背景。

    用户丢进来的 png 常常是"带背景的截图"(RGB, alpha 全 255)。不抠的话:
    残影补偿会把【整个背景】当成模型漏掉的部位, 捡回一张铺满画布的大补丁,
    把所有部件盖住; 而且真实漏掉的腿/鞋也埋在里面分不出来。
    返回 (抠好的 rgba, 前景 mask)。已经有透明通道的直接返回。
    """
    a = rgba[..., 3]
    if (a < 12).mean() > 0.02:
        return rgba, (a > 12)
    h, w = a.shape
    rgb = rgba[..., :3].astype(np.int16)
    pts = np.concatenate([rgb[:5, :5].reshape(-1, 3), rgb[:5, -5:].reshape(-1, 3),
                          rgb[-5:, :5].reshape(-1, 3), rgb[-5:, -5:].reshape(-1, 3)])
    bg = np.median(pts, axis=0)
    near = (np.abs(rgb - bg).max(axis=2) <= tol).astype(np.uint8)
    n, lab = cv2.connectedComponents(near, 4)
    edge = set(lab[0, :].tolist()) | set(lab[-1, :].tolist()) \
        | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    edge.discard(0)
    if not edge:
        return rgba, (a > 12)
    bgmask = np.isin(lab, list(edge))
    out = rgba.copy()
    out[bgmask] = 0
    fg = ~bgmask & (a > 12)
    return out, fg


def white_holes(rgba, min_area=300, thr=226, flat=0.45):
    """找出"被当成不透明的背景白"的大块 —— 烟雾/头发丝之间的白底洞。

    原图是白底截图时, 模型还原出来的 src_img 会把"几缕烟之间的那块白"当成前景
    涂成不透明的纯白。这块白会跟着图层被切进部件里: 部件图看着像"贴着一块白背景",
    拼回去在烟里多出两个白方块。

    三条同时满足才算洞(只按"白"会误伤亮色角色的高光):
      1. 连通块面积 ≥ min_area;
      2. 三通道都 > thr;
      3. 块里有 ≥flat 比例的像素是【纯】白(≥254) —— 背景是平涂的, 画上去的高光
         带渐变(实测 fa 的袖口/头巾高光纯白只占 1~10%, 董卓的洞占 65~82%)。
    """
    a = rgba[..., 3] > 12
    wh = ((rgba[..., :3].min(axis=2) > thr) & a).astype(np.uint8)
    if not wh.any():
        return np.zeros(a.shape, bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(wh, 8)
    out = np.zeros(a.shape, bool)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            continue
        m = lab == i
        px = rgba[..., :3][m]
        if float((px.min(axis=1) >= 254).mean()) < flat:
            continue
        out |= m
    return out


def map_to_frame(png_path, W, H, S, pad_x, pad_y):
    """768x768 层 -> 原图(W,H)空间 RGBA。"""
    im = np.array(Image.open(png_path).convert("RGBA"))   # 768x768x4
    big = cv2.resize(im, (S, S), interpolation=cv2.INTER_LINEAR)
    return big[pad_y:pad_y + H, pad_x:pad_x + W]          # (H,W,4)


def main_bbox(rgba, frac=0.01, min_px=30):
    """和 tight_bbox 一样返回 (x0,y0,x1,y1), 但忽略面积很小的孤立噪点。

    模型图层常在离主体很远的地方残留几个像素(alpha>12), tight_bbox 会把整张
    图撑大 —— 部件 png 变巨大、图集被迫开到 2048、道具重叠判断也跟着失真。
    这里按"主块 1% 以上"的连通块取并集, 既裁掉噪点又不会切掉头巾垂布这种
    真正的第二块。
    """
    a = (rgba[..., 3] > 12).astype(np.uint8)
    if not a.any():
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(a, 8)
    if n <= 2:
        return tight_bbox(rgba)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    big = max(areas)
    keep = [i + 1 for i, ar in enumerate(areas) if ar >= max(min_px, big * frac)]
    if not keep:
        return tight_bbox(rgba)
    x0 = min(int(stats[i, cv2.CC_STAT_LEFT]) for i in keep)
    y0 = min(int(stats[i, cv2.CC_STAT_TOP]) for i in keep)
    x1 = max(int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) for i in keep)
    y1 = max(int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) for i in keep)
    return x0, y0, x1, y1


def tight_bbox(rgba):
    a = rgba[..., 3]
    ys, xs = np.where(a > 12)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def grid_for(w, h):
    gx = max(2, min(10, round(w / PX_PER_CELL)))
    gy = max(2, min(10, round(h / PX_PER_CELL)))
    return [gx, gy]


_ZR = [(k.replace(" ", "_"), v) for k, v in Z_RANK]


def _z_rank(name):
    """按名字前缀取 z 序(数值小=先画/更靠后)。xxx_overlay 继承 xxxx 的层。"""
    n = name.lower().replace(" ", "_")
    best, rank = -1, 50
    for key, val in _ZR:
        if n.startswith(key) and len(key) > best:
            best, rank = len(key), val
    return rank


def _attach_props(parts):
    """把武器/盾这类道具绑到最近的手臂部件上 —— 手腕一动, 刀和盾跟着走。

    生成器会给道具打 100% 权重到那只手的骨头上(见 gen_idle_bones 的 cls=="prop")。
    判据: 和某只手/小臂的外框有重叠, 且中心离得够近; 否则不算道具(当普通部件)。
    """
    arms = [p for p in parts if str(p.get("cls", "")).startswith("arm_")]
    hits = 0
    for p in parts:
        prop = p.pop("is_prop", False)
        if not prop or not arms:
            continue
        cx = p["x"] + p["width"] / 2.0
        cy = p["y"] + p["height"] / 2.0
        # 谁的外框和它重叠最多就跟谁: 刀握在手里 -> 手; 盾贴在 小臂 -> 小臂
        # (绑手会绕腕关节甩, 绑小臂才是绕肘转, 盾看起来才稳)
        best, ba, bd = None, 0, 1e18
        for a in arms:
            if a is p:                      # 别跟自己匹配(道具名里带 arm_ 时会发生)
                continue
            ox = min(p["x"] + p["width"], a["x"] + a["width"]) - max(p["x"], a["x"])
            oy = min(p["y"] + p["height"], a["y"] + a["height"]) - max(p["y"], a["y"])
            if ox <= 0 or oy <= 0:          # 外框都不挨着, 不可能握在手里
                continue
            dx = a["x"] + a["width"] / 2.0 - cx
            dy = a["y"] + a["height"] / 2.0 - cy
            d = (dx * dx + dy * dy) ** 0.5
            if ox * oy > ba or (ox * oy == ba and d < bd):
                best, ba, bd = a, ox * oy, d
        if best is not None and bd <= 0.9 * max(p["width"], p["height"]) + 12:
            p["cls"] = "prop"
            p["follow_part"] = best["name"]
            hits += 1
            print(f"  道具 {p['name']:22s} -> 跟随 {best['name']} (重叠{ba}px d={bd:.0f})")
    if hits:
        print(f"道具绑定: {hits} 件武器/盾跟随手臂动作")
    return parts


def _check_face_order(parts, out_dir, src_rgba, W, H):
    """face / head 谁压谁, 按原图实测决定。

    模型给的 face 常常只是一块平整的皮肤底板(有时干脆是白板), 压在 head 上会把
    head 里已经画好的五官整片盖掉 —— 脸变成一块没有眼睛嘴的白板。
    反过来也可能: head 是光头底模, face 才是画好五官的那层。
    两种叠放各合成一次, 取更接近原图的那个; 差不多就不动。
    """
    fp = next((p for p in parts if p["name"] == "face"), None)
    hp = next((p for p in parts if p["name"] == "head"), None)
    if not fp or not hp:
        return parts
    x0 = max(0, min(fp["x"], hp["x"]))
    y0 = max(0, min(fp["y"], hp["y"]))
    x1 = min(W, max(fp["x"] + fp["width"], hp["x"] + hp["width"]))
    y1 = min(H, max(fp["y"] + fp["height"], hp["y"] + hp["height"]))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return parts

    imgs = {p["name"]: Image.open(os.path.join(out_dir, p["image"])).convert("RGBA")
            for p in (fp, hp)}

    def layer(order):
        c = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
        for p in order:
            c.alpha_composite(imgs[p["name"]], (p["x"] - x0, p["y"] - y0))
        return np.array(c).astype(np.int16)

    src = src_rgba[y0:y1, x0:x1].astype(np.int16)
    after = layer([hp, fp])      # face 压 head(默认)
    before = layer([fp, hp])     # face 垫在 head 下
    m = (src[..., 3] > 20) & (after[..., 3] > 20) & (before[..., 3] > 20)
    if int(m.sum()) < 200:
        return parts
    d_after = float(np.abs(after[..., :3] - src[..., :3]).mean(2)[m].mean())
    d_before = float(np.abs(before[..., :3] - src[..., :3]).mean(2)[m].mean())

    cur_before = parts.index(fp) < parts.index(hp)
    cur = d_before if cur_before else d_after
    best_before = d_before <= d_after
    best = d_before if best_before else d_after
    if best_before == cur_before or best > cur * 0.8:
        return parts          # 现状已经够好 / 差别不明显 -> 不动
    rest = [p for p in parts if p["name"] not in ("face", "head")]
    idx = min(parts.index(fp), parts.index(hp))
    rest.insert(idx, fp)
    rest.insert(idx + 1, hp)
    print(f"面部叠放自检: face 与 head 顺序对调 (与原图差异 {cur:.1f} -> {best:.1f}) -> "
          + ("face 垫在 head 之下" if best_before else "face 压在 head 之上(五官在 face 层)"))
    return rest


def _make_closed_eyes(parts, out_dir):
    """合成一张"闭眼"补丁(皮肤色 + 眼线), 供死亡/昏迷时盖住睁着的眼睛。

    为什么需要: 很多素材的 head 层本身就把眼睛画死了(eye/lash 是另外几层叠上去的),
    所以"压扁眼睛层"只能得到一条缝, 底下的眼睛还睁着。唯一可靠的办法是盖住它 ——
    按 eyewhite/irides/eyelash 的位置做一块皮肤补丁, 中间画一条眼线。
    补丁平时被缩到 0(看不见), 死亡落地时放大到 1 倍。
    """
    from PIL import ImageDraw

    by = {p["name"]: p for p in parts}
    srcs = [by[n] for n in ("eyewhite", "irides", "eyelash") if n in by]
    if len(srcs) < 2:
        return parts
    cw = max(int(p["x"] + p["width"]) for p in parts) + 8
    chh = max(int(p["y"] + p["height"]) for p in parts) + 8
    mask = np.zeros((chh, cw), np.uint8)
    eyeball = np.zeros((chh, cw), np.uint8)      # 只有眼球(眼白+瞳孔), 用来定眼线高度
    lash_rgb = (45, 42, 40)
    for p in srcs:
        im = np.array(Image.open(os.path.join(out_dir, p["image"])).convert("RGBA"))
        a = im[..., 3] > 12
        x0, y0 = int(p["x"]), int(p["y"])
        h, w = a.shape
        mask[y0:y0 + h, x0:x0 + w] |= a.astype(np.uint8)
        if p["name"] in ("eyewhite", "irides"):
            eyeball[y0:y0 + h, x0:x0 + w] |= a.astype(np.uint8)
        if p["name"] == "eyelash":             # 用睫毛的深色当眼线色
            px = im[..., :3][a].reshape(-1, 3).astype(np.float32)
            if len(px):
                dark = px[np.argsort(px.sum(1))[:max(1, len(px) // 3)]]
                lash_rgb = tuple(int(v) for v in np.median(dark, axis=0))
    if int(mask.sum()) < 40:
        return parts

    # 皮肤色: 取"眼睛外圈那一环"里 head 层的中位色 —— 这才是这张脸真正的肤色。
    # (不能用 face 底板: 那层常常是一块偏白的通用皮肤色, 和角色本身不一致)
    skin = None
    head_p = by.get("head")
    if head_p is not None:
        him = np.array(Image.open(os.path.join(out_dir, head_p["image"])).convert("RGBA"))
        hx0, hy0 = int(head_p["x"]), int(head_p["y"])
        hh, hw = him.shape[:2]
        canvas = np.zeros((chh, cw, 4), np.uint8)
        canvas[hy0:hy0 + hh, hx0:hx0 + hw] = him
        ring = cv2.dilate(mask, np.ones((13, 13), np.uint8), 1) & (1 - mask)
        sel = (ring > 0) & (canvas[..., 3] > 200)
        if int(sel.sum()) > 30:
            skin = np.median(canvas[..., :3][sel].reshape(-1, 3), axis=0)
    if skin is None:
        if "face" in by:                        # 兜底: 用 face 底板
            fp = by["face"]
            im = np.array(Image.open(os.path.join(out_dir, fp["image"])).convert("RGBA"))
            a = im[..., 3] > 200
            if a.any():
                skin = np.median(im[..., :3][a].reshape(-1, 3), axis=0)
    if skin is None:
        skin = np.array([150, 160, 140], np.float32)

    # 眼线按【膨胀前】的连通域画: 膨胀后两只眼可能并成一块, 一条线横跨鼻梁会很难看
    raw = mask.copy()
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    # 底图: 直接用 head 层的像素, 把眼睛区域 inpaint 修补掉 —— 纯色填充会抹平
    # 脸颊阴影, 看着像贴了一块膏药; inpaint 能延续周围的肤色和明暗。
    base = np.zeros((chh, cw, 3), np.uint8)
    base[:] = skin.astype(np.uint8)
    if head_p is not None:
        him = np.array(Image.open(os.path.join(out_dir, head_p["image"])).convert("RGBA"))
        hh2, hw2 = him.shape[:2]
        hy0, hx0 = int(head_p["y"]), int(head_p["x"])
        region = base[hy0:hy0 + hh2, hx0:hx0 + hw2]
        a2 = him[..., 3] > 200
        region[a2] = him[..., :3][a2]
    pad = 3
    big = cv2.copyMakeBorder(base, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    mbig = cv2.copyMakeBorder((mask * 255), pad, pad, pad, pad, cv2.BORDER_REFLECT)
    healed = cv2.inpaint(big, mbig, 3, cv2.INPAINT_TELEA)
    healed = healed[pad:pad + chh, pad:pad + cw]

    soft = cv2.GaussianBlur((mask * 255).astype(np.float32), (3, 3), 0)
    crop = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    crop[..., :3] = healed[y0:y1, x0:x1]
    crop[..., 3] = np.clip(soft[y0:y1, x0:x1], 0, 255).astype(np.uint8)

    # 每只眼一条微弯的眼线(中段略下沉), 位置在眼高偏下一点
    src_m = eyeball if int(eyeball.sum()) > 40 else raw
    n, lab, stats, _ = cv2.connectedComponentsWithStats(src_m.astype(np.uint8), 8)
    patch = Image.fromarray(crop, "RGBA")
    dr = ImageDraw.Draw(patch)
    for i in range(1, n):
        bx, byy, bw, bh, area = (int(v) for v in stats[i])
        if area < 20:
            continue
        cy = byy + bh * 0.55 - y0                # 眼球偏下一点, 像眼皮压下来的位置
        xa, xb = bx - x0 + 1, bx - x0 + bw - 1
        dr.line([(xa, cy - 1), (xa + (xb - xa) * 0.5, cy + 1.6), (xb, cy - 1)],
                fill=lash_rgb + (255,), width=2)
    patch.save(os.path.join(out_dir, "eyes_closed.png"))

    # 插到 eye 层之后(压住所有眼睛层)
    parts = [p for p in parts if p["name"] != "eyes_closed"]
    part = {"name": "eyes_closed", "image": "eyes_closed.png",
            "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
            "grid": grid_for(x1 - x0, y1 - y0), "cls": "eye_closed"}
    parts.append(part)
    parts.sort(key=lambda p: p.get("_z_arm", _z_rank(p["name"])))
    print(f"  闭眼补丁: {x1-x0}x{y1-y0} (皮肤色 + 眼线, 平时隐藏)")
    return parts


def keep_wear_color(mask, src_rgba, dh_tol=18, s_min=55):
    """只保留"和这块布的主色同色相"的像素。衣服(棕)/皮肤(灰绿)的色相和黄布
    差 20° 以上, 会被摘掉; 布面明暗褶子只变明度不变色相, 留得住。描边单独按
    "够暗且贴着布"保留, 否则轮廓会发毛。
    """
    px = src_rgba[..., :3][mask].reshape(-1, 3).astype(np.float32)
    if len(px) < 200:
        return mask
    q = (px // 32).astype(np.int32)
    key = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    ref = px[key == vals[np.argmax(counts)]].mean(0)
    hsv = cv2.cvtColor(src_rgba[..., :3].astype(np.uint8), cv2.COLOR_RGB2HSV)
    h_ref = int(cv2.cvtColor(np.uint8([[[int(ref[0]), int(ref[1]), int(ref[2])]]]),
                             cv2.COLOR_RGB2HSV)[0, 0, 0])
    h = hsv[..., 0].astype(np.int32)
    s_ = hsv[..., 1].astype(np.int32)
    v = hsv[..., 2].astype(np.int32)
    dh = np.abs(h - h_ref)
    dh = np.minimum(dh, 180 - dh)
    # 色相差 + 明度下限双条件: 棕衣服和黄布的色相只差 8~10°(8 位 H 只有 2°/格),
    # 单看色相分不开; 但布面明显更亮(黄 V≈230, 棕 V≈143), 加个明度下限就干净了。
    v_ref = int(cv2.cvtColor(np.uint8([[[int(ref[0]), int(ref[1]), int(ref[2])]]]),
                             cv2.COLOR_RGB2HSV)[0, 0, 2])
    cloth = (dh < dh_tol) & (s_ > s_min) & (v >= max(60, v_ref - 70))
    dark = src_rgba[..., :3].max(axis=2) < 90
    near = cv2.dilate(cloth.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return mask & (cloth | (dark & near))


def split_wear_parts(crop):
    """把"头盔 + 头巾"连成一整块的头饰按颜色拆开。

    原图里黄头巾是扎在帽檐【下面】的: 顺序是 头盔 > 头巾 > 脸。合成一整块再统一
    置顶的话, 头巾会跑到头盔上面; 统一靠下又会被脸盖。按颜色分开、各自定 z 才准。
    返回 (头盔 mask, 布/头巾 mask)。
    """
    rgb = crop[..., :3].astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.int32)
    s_ = hsv[..., 1].astype(np.int32)
    v = hsv[..., 2].astype(np.int32)
    a = crop[..., 3] > 12
    scarf = a & (s_ > 90) & (h >= 15) & (h <= 38) & (v > 90)
    return (a & ~scarf), scarf


def fill_small_holes(mask, max_area=900):
    """补掉 mask 里的小洞(不碰触边界的空白, 那是外部)。

    头盔/头巾的 mask 常被"下面那层头"啃掉一块, 合成出来帽顶是透明的,
    绿色头皮从洞里透出来, 看着就是"头盔没盖住头"。
    """
    inv = (~mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    H, W = mask.shape
    out = mask.copy()
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) > max_area:
            continue
        ys, xs = np.where(lab == i)
        if ys.size == 0:
            continue
        if ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1:
            continue
        out |= (lab == i)
    return out


def widen_headwear(parts, out_dir, src_rgba, W, H, fg_mask=None):
    """给头盔/头巾补回被"头/脸"那层啃掉的边(帽檐、下垂的布)。

    放在所有部件都生成之后做: 此时能算出"没有被任何部件占用"的前景像素, 把
    颜色又不是皮肤的那些补进 headwear。中途做容易被后续合并覆盖掉。
    """
    hw = next((q for q in parts if q["name"] == "headwear"), None)
    if hw is None or src_rgba is None:
        return parts
    cov = np.zeros((H, W), bool)
    for q in parts:
        if q is hw:
            continue
        im = np.array(Image.open(os.path.join(out_dir, q["image"])).convert("RGBA"))
        a = im[..., 3] > 12
        h, w = a.shape
        cov[q["y"]:q["y"] + h, q["x"]:q["x"] + w] |= a
    skin = None
    fc = next((q for q in parts if q["name"] == "face"), None)
    if fc is not None:
        fim = np.array(Image.open(os.path.join(out_dir, fc["image"])).convert("RGBA"))
        m = fim[..., 3] > 12
        if m.any():
            skin = fim[..., :3][m].reshape(-1, 3).astype(np.float32).mean(0)
    fg = fg_mask if fg_mask is not None else (src_rgba[..., 3] > 12)
    add = fg & ~cov
    if skin is not None:
        d = np.linalg.norm(src_rgba[..., :3].astype(np.float32) - skin, axis=2)
        add &= (d > 30)
    # 区域以【头】为准往外扩: 帽檐比头宽得多, 只按 headwear 自己的 bbox 扩会漏。
    hd = next((q for q in parts if q["name"] == "head"), None)
    if hd is None:
        hd = hw
    reg = np.zeros((H, W), bool)
    x0 = max(0, min(hd["x"], hw["x"]) - 30)
    x1 = min(W, max(hd["x"] + hd["width"], hw["x"] + hw["width"]) + 30)
    y0 = max(0, hw["y"] - 10)
    y1 = min(H, hd["y"] + int(hd["height"] * 0.55))
    reg[y0:y1, x0:x1] = True
    add &= reg
    if int(add.sum()) < 50:
        return parts
    im = np.array(Image.open(os.path.join(out_dir, hw["image"])).convert("RGBA"))
    canvas = np.zeros((H, W, 4), np.uint8)
    canvas[hw["y"]:hw["y"] + im.shape[0], hw["x"]:hw["x"] + im.shape[1]] = im
    canvas[add] = src_rgba[add]
    nb = tight_bbox(canvas)
    if nb is None:
        return parts
    bx0, by0, bx1, by1 = nb
    Image.fromarray(canvas[by0:by1, bx0:bx1]).save(os.path.join(out_dir, hw["image"]))
    hw.update({"x": bx0, "y": by0, "width": bx1 - bx0, "height": by1 - by0,
               "grid": grid_for(bx1 - bx0, by1 - by0)})
    print(f"  头饰补边: +{int(add.sum())}px (帽檐/下垂的布)")
    return parts


def strip_part_bg(parts, out_dir, bg, tol=34):
    """把"原图的底色"从每个部件里抠掉。

    原图是带底色的截图(RGB, 没有 alpha)时, 模型分层偶尔会把背景也算进某一层,
    部件就会拖着一块底色 —— 拼回去时在角色周围露出一块方块。
    只删颜色接近底色的像素: 角色自己的颜色(金属灰/布/皮肤)离得远, 不会误伤;
    删完重算包围盒。
    返回修过的部件数。
    """
    if bg is None:
        return 0
    bg = np.asarray(bg, np.float32)
    n_fix = 0
    for p in parts:
        fp = os.path.join(out_dir, p["image"])
        if not os.path.isfile(fp):
            continue
        im = np.array(Image.open(fp).convert("RGBA"))
        a = im[..., 3] > 12
        if not a.any():
            continue
        d = np.linalg.norm(im[..., :3].astype(np.float32) - bg, axis=2)
        raw = a & (d < tol)
        # 只删"和部件外轮廓连着"的底色块: 角色身上的暗部(阴影/黑描边)颜色也可能
        # 接近底色, 但它们被角色包在中间, 不碰边缘 —— 按连通域挑出贴边的才删,
        # 否则脸上会被啃出一条条白带(踩过)。
        import cv2 as _cv
        n_lab, lab = _cv.connectedComponents(raw.astype(np.uint8), 8)
        edge = set(lab[0, :].tolist()) | set(lab[-1, :].tolist()) \
            | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
        edge.discard(0)
        if not edge:
            continue
        kill = np.isin(lab, list(edge))
        n = int(kill.sum())
        if n < 60 or n > a.sum() * 0.75:
            continue
        im[..., 3] = np.where(kill, 0, im[..., 3])
        nb = tight_bbox(im)
        if nb is None:
            continue
        x0, y0, x1, y1 = nb
        Image.fromarray(im[y0:y1, x0:x1]).save(fp)
        p.update({"x": int(p["x"] + x0), "y": int(p["y"] + y0),
                  "width": x1 - x0, "height": y1 - y0,
                  "grid": grid_for(x1 - x0, y1 - y0)})
        n_fix += 1
        print(f"  去底色: {p['name']:16s} -{n}px")
    return n_fix


def drop_blank_face(parts, out_dir, src_rgba):
    """丢掉"空白皮肤底板"那种 face。

    模型有时把脸分成两层: head 是画好五官的脸, face 是一块没细节的浅色底。
    face 垫在 head 下面本来无害, 但 head 有透明缺口, 露出来的浅色底就成了"白板脸"。
    判据: face 的颜色和原图同一块区域差得远(>45), 而 head 接近原图 -> 删掉 face。
    """
    if src_rgba is None:
        return parts
    fc = next((q for q in parts if q["name"] == "face"), None)
    hd = next((q for q in parts if q["name"] == "head"), None)
    if fc is None or hd is None:
        return parts

    def _diff(q):
        fp = os.path.join(out_dir, q["image"])
        if not os.path.isfile(fp):
            return None
        im = np.array(Image.open(fp).convert("RGBA"))
        a = im[..., 3] > 200
        if not a.any():
            return None
        sub = src_rgba[q["y"]:q["y"] + im.shape[0], q["x"]:q["x"] + im.shape[1]]
        sa = sub[..., 3] > 200
        if not sa.any():
            return None
        col = im[..., :3][a].reshape(-1, 3).astype(np.float32).mean(0)
        src_col = sub[..., :3][sa].reshape(-1, 3).astype(np.float32).mean(0)
        return float(np.linalg.norm(col - src_col))

    d_f, d_h = _diff(fc), _diff(hd)
    if d_f is None or d_h is None:
        return parts
    # face 明显偏离原图、head 更接近 -> face 就是那块空白底
    if d_f > 45 and d_f > d_h + 20:
        parts = [q for q in parts if q is not fc]
        print(f"  白板脸: face 与原图差 {d_f:.0f}(head 只差 {d_h:.0f}), 已丢弃 face")
    return parts


def trim_arm_body(parts, out_dir, W=None, H=None):
    """把手臂部件里混进来的"身体"剔掉。

    handwear 切出来的大臂矩形常把躯干的一部分一起带进来(suo 的左大臂内侧挂着
    一块 660px 的深色胸甲)。判据两条, 满足其一即算身体:
      a) 颜色接近 topwear 主色;  或者
      b) 被 topwear 图层覆盖、并且颜色偏暗(不是金属高光)。
    手臂自己的金属件要么不被 topwear 覆盖, 要么够亮, 不会被误伤。
    """
    tw = next((q for q in parts if q["name"] == "topwear"), None)
    if tw is None:
        return parts
    tw_img = np.array(Image.open(os.path.join(out_dir, tw["image"])).convert("RGBA"))
    ta = tw_img[..., 3] > 12
    if not ta.any():
        return parts
    tcol = tw_img[..., :3][ta].reshape(-1, 3).astype(np.float32).mean(0)
    if W is None or H is None:
        W = max(q["x"] + q["width"] for q in parts)
        H = max(q["y"] + q["height"] for q in parts)
    W, H = int(W), int(H)
    tw_mask = np.zeros((H, W), bool)
    tw_mask[tw["y"]:tw["y"] + tw_img.shape[0], tw["x"]:tw["x"] + tw_img.shape[1]] = ta

    fixed = 0
    for p in parts:
        if not (p["name"].startswith("arm_upper") or p["name"].startswith("arm_fore")):
            continue
        im = np.array(Image.open(os.path.join(out_dir, p["image"])).convert("RGBA"))
        h, w = im.shape[:2]
        px, py = int(p["x"]), int(p["y"])
        sub_tw = np.zeros((h, w), bool)
        sy0, sx0 = max(0, py), max(0, px)
        sy1, sx1 = min(H, py + h), min(W, px + w)
        if sy1 <= sy0 or sx1 <= sx0:
            continue
        sub_tw[sy0 - py:sy1 - py, sx0 - px:sx1 - px] = tw_mask[sy0:sy1, sx0:sx1]
        if not sub_tw.any():
            continue
        a = im[..., 3] > 12
        d = np.linalg.norm(im[..., :3].astype(np.float32) - tcol, axis=2)
        # 只按"颜色接近衣服主色"判: 之前还加了"亮度<95", 结果深色角色(董卓整身都暗)
        # 的护臂被整片当身体删掉, 手臂只剩几个碎点。
        kill = a & sub_tw & (d < 30)
        n_kill = int(kill.sum())
        if n_kill < 40:
            continue
        # 一下删掉超过 1/4 就说明判错了(大片重叠是角色结构, 不是蹭到的身体边),
        # 宁可留着, 也不能把手臂掏空。
        if n_kill > 0.25 * int(a.sum()):
            print(f"  手臂去身体: {p['name']} 重叠过多({n_kill}/{int(a.sum())}), 跳过")
            continue
        im[..., 3] = np.where(kill, 0, im[..., 3])
        nb = tight_bbox(im)
        if nb is None:
            continue
        x0, y0, x1, y1 = nb
        Image.fromarray(im[y0:y1, x0:x1]).save(os.path.join(out_dir, p["image"]))
        p.update({"x": px + x0, "y": py + y0,
                  "width": x1 - x0, "height": y1 - y0,
                  "grid": grid_for(x1 - x0, y1 - y0)})
        fixed += 1
        print(f"  手臂去身体: {p['name']} 剔掉 {int(kill.sum())}px (体块/暗部)")
    if fixed:
        print(f"  手臂去身体: {fixed} 个部件已修")
    return parts


# 肩甲+大臂的取形(相对肘点 / 前臂宽高)。顺序绕一圈:
#   内上 -> 外上 -> 外中 -> 外下 -> 肘外 -> 肘内 -> 内下 -> 内中
# 为什么不是"椭圆/胶囊": 肩甲是个上面带冠、右边贴着胸甲、下面接布袖的方壳子,
# 圆头顶多盖住中段 —— 左肩甲的冠(左上凸起)和右下角都会留在躯干上, 就是
# "左肩膀没拆出来"。这套比例是把肩甲整块框住的形状, 两侧镜像对称。
_ARM_SHAPE = ((0.30, -2.25), (-0.30, -2.10), (-0.36, -1.20), (-0.30, -0.45),
              (-0.55, -0.02), (0.55, -0.02), (0.32, -0.40), (0.60, -0.85))


def _shade_hole(canvas, hole, k=0.55):
    """挖掉手臂后, 把上衣层那个肩形洞里的像素【压暗】当作底下的阴影。

    试过两种补洞都不行:
      · cv2.inpaint 把整块揉成一团灰糊;
      · "边界颜色向内扩散"会把胸甲的灰顺着洞长出一只假肩甲 —— 比洞还难看。
    压暗原像素最稳: 手臂盖着的时候完全看不见; 臂骨走 1~2px 露出来的是一条暗边,
    看起来就是肩甲压在身体上的投影, 不像重影也不像洞。
    """
    out = canvas.copy()
    rgb = out[..., :3].astype(np.float32)
    rgb[hole] *= k
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def split_arm_from_body(parts, out_dir, W, H, force=False):
    """把"肩甲 + 大臂"从 topwear(上衣)层里切出来, 变成 arm_upper_L/R。

    模型只把"肘以下"(铁护臂+手)分进 handwear, 肩甲和布袖留在上衣层里 —— 结果
    大臂只剩几根描边碎线, 肩甲永远钉在躯干上, 手臂一摆就"肩膀不动、胳膊断开"。
    重甲角色几乎都是这样(董卓/吕布这种大护肩)。

    切法(无监督, 只靠前臂当锚点): 肘点 + 前臂宽高 推出一个"肩甲壳 + 布袖梯形"的
    多边形(_ARM_SHAPE), 与上衣层求交。肘比肩外张是立绘通例, 所以所有横向偏移都
    按"内侧/外侧"写, 左右镜像。

    切下来的像素【必须从上衣层挖走】: 只复制不挖的话, 上衣里那张一模一样的肩甲
    留在原地 —— 肩甲的位置被"钉"住了, 看起来就是"左边肩膀没拆出来"。挖完用
    `_shade_hole` 把洞里压暗当投影, 臂骨走 1~2px 时边沿不漏也不出假肩甲。
    """
    tw = next((q for q in parts if q["name"] == "topwear"), None)
    if tw is None:
        return parts
    fp = os.path.join(out_dir, tw["image"])
    if not os.path.isfile(fp):
        return parts
    img = np.array(Image.open(fp).convert("RGBA"))
    tcan = np.zeros((H, W, 4), np.uint8)
    tcan[tw["y"]:tw["y"] + img.shape[0], tw["x"]:tw["x"] + img.shape[1]] = img
    tmask = tcan[..., 3] > 12
    if not tmask.any():
        return parts
    # 紫烟(角色身后的能量/烟雾): 手臂图里要带上(不然肩甲上会缺一块), 但上衣层
    # 【不挖】—— 烟是环境, 不该跟着胳膊走; 而且留着它, 挖洞补边的边沿颜色才不是
    # 一片灰(否则胸甲的灰会顺着洞往上长出一根手指)。
    hsv = cv2.cvtColor(tcan[..., :3].astype(np.uint8), cv2.COLOR_RGB2HSV)
    smoke = ((hsv[..., 0] >= 118) & (hsv[..., 0] <= 168)
             & (hsv[..., 1] > 35) & (hsv[..., 2] < 150))
    made = 0
    take = np.zeros((H, W), bool)          # 从上衣层挖走的总像素
    for side, inward in (("L", +1), ("R", -1)):
        fore = next((q for q in parts if q["name"] == f"arm_fore_{side}"), None)
        if fore is None:                     # 没有前臂当锚点, 位置猜不出来, 不切
            continue
        wf, hf = float(fore["width"]), float(fore["height"])
        cx_e = fore["x"] + wf / 2.0
        cy_e = float(fore["y"])
        band = np.zeros((H, W), np.uint8)
        cv2.fillPoly(band, [np.array([(cx_e + inward * X * wf, cy_e + Y * hf)
                                      for X, Y in _ARM_SHAPE], np.int32)], 255)
        m = tmask & (band > 0)
        # 细碎的描边点不算部件(带上会显得肩甲边缘一排白毛刺)
        _n, _lab, _st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        _keep = np.zeros_like(m)
        for _i in range(1, _n):
            if int(_st[_i, cv2.CC_STAT_AREA]) >= 40:
                _keep |= (_lab == _i)
        m = _keep
        # 紫烟只留贴着肩甲的那一圈(鬃毛边), 大团烟留在上衣层 —— 部件图上不会驮着
        # 一整块紫云, 而肩甲边缘又不会像被刀切过。烟本身不跟着胳膊走(不挖)。
        if smoke.any():
            armor = m & ~smoke
            if armor.any():
                fringe = cv2.dilate(armor.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                m = armor | (m & smoke & fringe)
        if int(m.sum()) < 400:
            continue

        # 已经拆干净了就别再切(否则正常角色的大臂会被上衣挖掉一块)
        old = next((q for q in parts if q["name"] == f"arm_upper_{side}"), None)
        old_mask = np.zeros((H, W), bool)
        if old is not None and os.path.isfile(os.path.join(out_dir, old["image"])):
            oim = np.array(Image.open(os.path.join(out_dir, old["image"])).convert("RGBA"))
            old_mask[old["y"]:old["y"] + oim.shape[0], old["x"]:old["x"] + oim.shape[1]] = oim[..., 3] > 12
        fresh = int((m & ~old_mask).sum())
        if not force and old_mask.any() and fresh < 0.35 * int(m.sum()):
            print(f"  肩甲切分 {side}: 大臂已独立(新增仅 {fresh}px), 跳过")
            continue

        # 手臂图: 上衣层的像素打底, 老部件(handwear 切出来的袖子)补边
        union = m | old_mask
        acan = np.zeros((H, W, 4), np.uint8)
        acan[m] = tcan[m]
        if old is not None:
            oim = np.array(Image.open(os.path.join(out_dir, old["image"])).convert("RGBA"))
            sub = np.zeros((H, W), bool)
            sub[old["y"]:old["y"] + oim.shape[0], old["x"]:old["x"] + oim.shape[1]] = True
            fill = union & ~m & sub
            if fill.any():
                tile = acan[old["y"]:old["y"] + oim.shape[0], old["x"]:old["x"] + oim.shape[1]]
                src_t = oim.copy()
                src_t[..., 3] = np.where(fill[old["y"]:old["y"] + oim.shape[0],
                                               old["x"]:old["x"] + oim.shape[1]],
                                         src_t[..., 3], 0)
                keep = tile[..., 3] > 12
                tile[~keep] = src_t[~keep]
                acan[old["y"]:old["y"] + oim.shape[0], old["x"]:old["x"] + oim.shape[1]] = tile

        bb = tight_bbox(acan)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        name = f"arm_upper_{side}"
        fname = name + ".png"
        Image.fromarray(acan[y0:y1, x0:x1]).save(os.path.join(out_dir, fname))
        if old is not None:
            if old["image"] != fname and os.path.isfile(os.path.join(out_dir, old["image"])):
                try:
                    os.remove(os.path.join(out_dir, old["image"]))
                except OSError:
                    pass
            old.update({"image": fname, "x": x0, "y": y0,
                        "width": x1 - x0, "height": y1 - y0,
                        "grid": grid_for(x1 - x0, y1 - y0), "cls": "arm_upper"})
            arm_part = old
        else:
            arm_part = {"name": name, "image": fname, "x": x0, "y": y0,
                        "width": x1 - x0, "height": y1 - y0,
                        "grid": grid_for(x1 - x0, y1 - y0), "cls": "arm_upper"}
            parts.append(arm_part)
        # 肩甲压在胸甲上(画师是这么画的): z 必须排到 topwear 前面, 否则胸口会盖住肩甲
        arm_part["_z_arm"] = 22
        made += 1
        # 只挖上衣层里真有的像素(handwear 补进来的边可能原本就透明, 挖它会在
        # 轮廓外糊出一圈), 并且放过烟雾像素
        take |= (acan[..., 3] > 12) & tmask & ~smoke
        print(f"  肩甲切分 {side}: 上衣层切出 肩甲+大臂 {int(m.sum())}px "
              f"-> {fname} {x1 - x0}x{y1 - y0}")

    if not made:
        return parts

    # 把手臂那块的像素从上衣层【挖走】, 再把洞按周围颜色铺平。
    # 只复制不挖的话, 上衣里那张一模一样的肩甲留在原地, 肩甲就被钉死了 ——
    # 表现是"哪边都没拆出来", 左边尤其明显(左肩甲大、几乎和上衣完全重合)。
    body = _shade_hole(tcan, take)
    bb = tight_bbox(body)
    if bb is not None:
        x0, y0, x1, y1 = bb
        Image.fromarray(body[y0:y1, x0:x1]).save(os.path.join(out_dir, tw["image"]))
        tw.update({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
                   "grid": grid_for(x1 - x0, y1 - y0)})
    print(f"  肩甲切分: {made} 条手臂已从上衣层分出, 上衣层已挖洞压暗 "
          f"{int((take & (tcan[..., 3] > 12)).sum())}px")
    return parts


def demote_hair_over_face(parts, out_dir, src_rgba, W, H, thr=0.55):
    """头发把整张脸糊住时, 用原图把"脸那块"补回来, 并压到五官层下面。

    模型偶尔会把"前发/头饰"整块涂成实心头发: 脸的位置糊一块不透明灰(实测
    rgb 90,90,90 / alpha 253), 整颗头就被盖死 —— 拼回去是"没有脸"。
    处理:
      1. 头发 mask 盖住 head mask 的 55% 以上 -> 判定它不是刘海, 是整颗头;
      2. 把它在 head 附近(=脸)的像素**换成原图的像素**, 灰糊就没了;
      3. z 排到 head(30) 之后、五官(34) 之前(31) —— 脸用原图, 眼鼻嘴还在最上面
         (眨眼/闭眼照旧能播)。
    """
    hd = next((p for p in parts if p["name"] == "head"), None)
    if hd is None:
        return parts

    def mask_of(p):
        fp = os.path.join(out_dir, p["image"])
        if not os.path.isfile(fp):
            return None
        im = np.array(Image.open(fp).convert("RGBA"))
        m = np.zeros((H, W), bool)
        m[p["y"]:p["y"] + im.shape[0], p["x"]:p["x"] + im.shape[1]] = im[..., 3] > 12
        return m

    hm = mask_of(hd)
    if hm is None or not hm.any():
        return parts
    if src_rgba is None:
        return parts
    # 脸的范围 = head 轮廓外扩 6px(够盖住模型糊的那圈灰, 又不会吃到两侧的头发)
    face = cv2.dilate(hm.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    fg = src_rgba[..., 3] > 12
    # "脸心"= head 往里收 4px: 只看脸中间那块, 避开轮廓和头发交界
    core = cv2.erode(hm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if not core.any():
        core = hm
    src_col = src_rgba[..., :3][core & fg].reshape(-1, 3).astype(np.float32)
    if len(src_col) < 20:
        return parts
    src_col = src_col.mean(0)

    def face_err(p):
        """这个部件在"脸心"上离原图有多远(越小越像原图)。"""
        fp = os.path.join(out_dir, p["image"])
        im = np.array(Image.open(fp).convert("RGBA"))
        m = np.zeros((H, W), bool)
        m[p["y"]:p["y"] + im.shape[0], p["x"]:p["x"] + im.shape[1]] = im[..., 3] > 200
        sel = core & m
        if int(sel.sum()) < 20:
            return None
        px = np.zeros((H, W, 3), np.float32)
        px[p["y"]:p["y"] + im.shape[0], p["x"]:p["x"] + im.shape[1]] = im[..., :3]
        return float(np.linalg.norm(px[sel].mean(0) - src_col))

    head_err = face_err(hd)
    if head_err is None:
        return parts
    # 合成到"整颗头"的画布上: head 原有像素 + 各头发层(脸那块换成原图)
    hc = np.array(Image.open(os.path.join(out_dir, hd["image"])).convert("RGBA"))
    canvas = np.zeros((H, W, 4), np.uint8)
    canvas[hd["y"]:hd["y"] + hc.shape[0], hd["x"]:hd["x"] + hc.shape[1]] = hc
    merged, drop = [], []
    for p in parts:
        nm = str(p["name"]).lower()
        if not any(k in nm for k in ("hair", "headwear", "headscarf")):
            continue
        if p is hd:
            continue
        m = mask_of(p)
        if m is None:
            continue
        cover = float((m & hm).sum()) / float(hm.sum())
        err = face_err(p)
        # 盖住 head 大半、而且"脸心"的颜色离原图比 head 差很多 -> 它在脸上画错了东西
        bad = err is not None and err > head_err + 35
        if not (cover > thr and bad):
            continue
        im = np.array(Image.open(os.path.join(out_dir, p["image"])).convert("RGBA"))
        tile = im.copy()
        sub = face[p["y"]:p["y"] + im.shape[0], p["x"]:p["x"] + im.shape[1]]
        want = sub & fg[p["y"]:p["y"] + im.shape[0], p["x"]:p["x"] + im.shape[1]]
        tile[want] = src_rgba[p["y"]:p["y"] + im.shape[0],
                              p["x"]:p["x"] + im.shape[1]][want]
        region = canvas[p["y"]:p["y"] + im.shape[0], p["x"]:p["x"] + im.shape[1]]
        add = (tile[..., 3] > 12) & (region[..., 3] <= 12)
        region[add] = tile[add]
        canvas[p["y"]:p["y"] + im.shape[0], p["x"]:p["x"] + im.shape[1]] = region
        drop.append(p)
        merged.append((p["name"], cover))

    if not merged:
        return parts
    # 整颗头直接按【原图】重铺一遍: 模型给的 head 本身也不干净(华雄的额头是一块
    # 灰), 脸+头发既然是同一条刚性骨, 合起来照原图取像素最还原, 也不会出现
    # "原图的脸 vs 模型的脸"两块拼缝。
    total = canvas[..., 3] > 12
    canvas[total] = src_rgba[total]
    bb = tight_bbox(canvas)
    if bb is None:
        return parts
    x0, y0, x1, y1 = bb
    Image.fromarray(canvas[y0:y1, x0:x1]).save(os.path.join(out_dir, hd["image"]))
    hd.update({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
               "grid": grid_for(x1 - x0, y1 - y0)})
    # 头发层从部件表里去掉 —— 它已经被并进 head。
    # 关键: 不能留着单独一块"头发"却把原图的脸烘进去, 那块会挂在发丝飘动链上,
    # 一脸皮跟着头发甩。并进 head 就 100% 跟 b_head 刚性动。
    for p in drop:
        parts.remove(p)
    # 脸已经按原图铺好了(眼睛鼻子嘴都在上面), 模型再给一份"五官"就是重复:
    # 华雄的 eyewhite 是一块深色, 压在原图的琥珀色眼睛上, 眼睛直接看不见。
    # 规则: head bbox 里的 鼻/嘴/眉 丢掉(它们纯属重复); 眼白/睫毛留着 ——
    # 生成器要用它们的位置去做"闭眼补丁", 但 z 排在 head 下面, 不参与显示。
    hbb = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
    for p in list(parts):
        nm = str(p["name"]).lower()
        cx = p["x"] + p["width"] / 2.0
        cy = p["y"] + p["height"] / 2.0
        inside = (hbb["x0"] - 2 <= cx <= hbb["x1"] + 2
                  and hbb["y0"] - 2 <= cy <= hbb["y1"] + 2)
        if inside and any(k in nm for k in ("nose", "mouth", "eyebrow")):
            parts.remove(p)
            print(f"  脸已按原图重铺: 去掉重复的 {p['name']}")
    # 26.2 这种小数没用, 用 36.2: 压住眼白(34)/睫毛(36), 又让闭眼补丁(36.5)盖得住
    hd["_z_arm"] = 36.2
    for nm, c in merged:
        print(f"  头发盖脸: {nm} 覆盖 head {c * 100:.0f}% -> 脸用原图补回, 并入 head(刚性跟头)")
    return parts


def _cut_blades(tcan, outside, out_dir, W, H, cx0, min_area=400):
    """把"髋线以下、双手之外"的那几坨拆成独立道具(刀/剑), 让它们能跟手。

    双手持刃的角色, 模型把刀也算进了 topwear(objects 层是空的)。刀是静态像素,
    手一抬就在刀柄上滑。这里按连通域把刀切出来, 左右各挂一只, 后面 `_attach_props`
    会按外框重叠把它绑到对应的手/小臂上。
    """
    m8 = outside.astype(np.uint8)
    if not m8.any():
        return [], None
    # 先膨胀再连通: 刀身和它内侧那条描边常常是**两坨**(中间隔着 1~2px 的空隙),
    # 不合并的话会拆出一堆 20px 宽的碎片, 跟着手走的时候互相错开。
    d8 = cv2.dilate(m8, np.ones((7, 7), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(d8, 8)
    out, mask = [], np.zeros((H, W), bool)
    used = set()
    for i in range(1, n):
        m = (lab == i) & outside              # 只认原本的前景像素
        area = int(m.sum())
        if area < min_area:
            continue                          # 零碎边角(披风毛边)不当刀
        ys, xs = np.where(m)
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
        if w < 6 or h < 6:
            continue
        side = "L" if cent[i][0] < cx0 else "R"
        name = f"objects_{side}"
        k = 2
        while name in used:                   # 同一侧两把刀
            name = f"objects_{side}{k}"
            k += 1
        used.add(name)
        sub = np.zeros((h, w, 4), np.uint8)
        src = tcan[y:y + h, x:x + w]
        mm = m[y:y + h, x:x + w]
        sub[mm] = src[mm]
        fname = name + ".png"
        Image.fromarray(sub).save(os.path.join(out_dir, fname))
        out.append({"name": name, "image": fname, "x": x, "y": y,
                    "width": w, "height": h, "grid": grid_for(w, h),
                    "cls": "prop", "is_prop": True, "_z_arm": 23,
                    # 先按左右手挂上; _attach_props 之后会按外框重叠再校正一次
                    "follow_part": f"arm_hand_{side}"})
        mask |= m
    if out:
        print(f"  刀剑(从上衣层): 拆出 {len(out)} 件道具 "
              f"({', '.join(p['name'] for p in out)}), 交给手部跟随")
    return out, (mask if out else None)


def split_legs_from_body(parts, out_dir, W, H, knee_ratio=0.45, overlap=4):
    """模型把"长袍下摆 + 腿甲"整块算进 topwear、legwear 是空的时候, 从衣服里切腿。

    华雄/董卓这类穿长袍的重甲: 模型只把靴子分进 footwear, 膝盖以上全留在 topwear。
    没有 leg_L/R 就没有髋/膝骨, 行走、受击、死亡的腿动作全都做不出来 —— 表现就是
    "部件不够, 没法生成动作"。

    切法: 髋线 = 脚底往上 0.47 个身高(成年人比例); 线以下全部算下肢, 按连通域分
    左右, 再按膝高切成上/下两段。切下来的像素从上衣层挖走并压暗(同 `_shade_hole`),
    腿排到衣服前面 —— 留着原像素的话腿就被上衣盖住, 走起来一动不动。
    """
    if any(str(p.get("name", "")).startswith("leg_") for p in parts):
        return parts                          # 已经有腿了, 不动
    tw = next((p for p in parts if p["name"] == "topwear"), None)
    if tw is None:
        return parts
    feet = [p for p in parts if p.get("cls") == "foot"]
    head = next((p for p in parts if p["name"] in ("head", "face")), None)
    # 脚不一定是独立部件: 魔身/全身甲这类, 模型把靴子也塞进 topwear, footwear 整层是空的。
    # 以前"没有 foot 就 return"直接把这条腿的活干掉了 —— 界面只会报"缺腿, 不能生成动作"。
    # 没有脚就用整体轮廓的底边兜底(脚底 ≈ 披风/裙摆的最低点)。
    if feet:
        bot_y = float(max(p["y"] + p["height"] for p in feet))
    else:
        bot_y = float(max(p["y"] + p["height"] for p in parts if p["name"] != "topwear"))
        bot_y = max(bot_y, float(tw["y"] + tw["height"]))
    if head is not None:
        top_y = float(head["y"])
    else:                                     # 连头都没有(只剩躯干+手臂): 用最高的部件
        top_y = float(min(p["y"] for p in parts))
    if bot_y - top_y < 40:
        return parts
    hip = bot_y - 0.47 * (bot_y - top_y)

    fp = os.path.join(out_dir, tw["image"])
    img = np.array(Image.open(fp).convert("RGBA"))
    tcan = np.zeros((H, W, 4), np.uint8)
    tcan[tw["y"]:tw["y"] + img.shape[0], tw["x"]:tw["x"] + img.shape[1]] = img
    lower = tcan[..., 3] > 12
    lower[:int(round(hip)), :] = False

    # 髋线以下往往还挂着**刀剑**: 双手持刃的角色, 模型把刀也塞进 topwear(objects 是空的),
    # 刀从手上一直斜着垂到脚边 —— 不挡住的话会被当成"腿"切下来(魔身实测切出两片刀刃)。
    # 判据: 腿一定在两只手之间。取左右手内缘当水平闸门, 外面的一律不算下肢。
    hands = [p for p in parts if p.get("cls") == "arm_hand"]
    blade_parts, blade_mask = [], None
    if len(hands) >= 2:
        cx0 = float(tw["x"]) + float(tw["width"]) / 2.0
        lh = [p for p in hands if p["x"] + p["width"] / 2.0 < cx0]
        rh = [p for p in hands if p["x"] + p["width"] / 2.0 >= cx0]
        if lh and rh:
            gate_l = min(p["x"] + p["width"] for p in lh)
            gate_r = max(p["x"] for p in rh)
            if gate_r - gate_l > 0.10 * W:            # 闸门太窄就别用, 免得把腿夹没
                band = np.zeros((H, W), bool)
                band[:, int(gate_l):int(gate_r)] = True
                before = int(lower.sum())
                outside = lower & ~band
                lower &= band
                print(f"  腿(从上衣层): 只取双手之间 x={int(gate_l)}..{int(gate_r)} "
                      f"({before} -> {int(lower.sum())}px), 刀剑留在上衣层")
                # 被挡在闸门外的那些 = 手里的刀剑。它们也必须拆出来跟着手走,
                # 否则手随缩短短抬高时, 手就在刀柄上"滑"(实测魔身的两把大剑)。
                blade_parts, blade_mask = _cut_blades(
                    tcan, outside, out_dir, W, H, cx0)
    if int(lower.sum()) < 800:                # 上衣本来就没到下半身, 不切
        return parts

    buf = np.zeros((H, W, 4), np.uint8)
    buf[lower] = tcan[lower]
    segs = split_legs(buf)
    if not segs:
        # 长袍把两条腿连成一片: 按下半身自己的中线硬切左右
        ys, xs = np.where(lower)
        mid = (int(xs.min()) + int(xs.max())) // 2
        l = buf.copy()
        l[:, mid:] = 0
        r = buf.copy()
        r[:, :mid] = 0
        segs = [{"side": "L", "rgba": l}, {"side": "R", "rgba": r}]
        print(f"  腿(从上衣层): 下摆连成一片, 按中线 x={mid} 硬切左右")

    made = 0
    new_parts = []
    for seg in segs:
        side = seg["side"]
        for sub, tag in split_leg_halves(seg["rgba"], knee_ratio, overlap):
            bb = main_bbox(sub)
            if bb is None:
                continue
            x0, y0, x1, y1 = bb
            crop = sub[y0:y1, x0:x1]
            if int((crop[..., 3] > 12).sum()) < 120:
                continue
            name = f"leg_{side}_{tag}"
            fname = name + ".png"
            Image.fromarray(crop).save(os.path.join(out_dir, fname))
            new_parts.append({"name": name, "image": fname, "x": x0, "y": y0,
                              "width": x1 - x0, "height": y1 - y0,
                              "grid": grid_for(x1 - x0, y1 - y0), "cls": "leg",
                              "bind_bone": (f"leg_{side}_up" if tag == "up"
                                            else f"leg_{side}_fore"),
                              # 腿是从上衣层切出来的, 必须排在衣服前面, 否则被盖住
                              "_z_arm": 22})
            made += 1
    if not made and not blade_parts:
        return parts

    # 挖走上衣层里的下半身(压暗当投影), 否则腿被衣服盖着不动
    sub_take = np.zeros((H, W), bool)
    for p in new_parts:
        m = np.zeros((H, W), bool)
        im = np.array(Image.open(os.path.join(out_dir, p["image"])).convert("RGBA"))
        m[p["y"]:p["y"] + im.shape[0], p["x"]:p["x"] + im.shape[1]] = im[..., 3] > 12
        sub_take |= m
    sub_take &= lower
    parts.extend(new_parts)
    body = _shade_hole(tcan, sub_take) if made else tcan
    if blade_mask is not None:                # 刀剑已经从衣服里拿走了, 别留双份
        body = body.copy()
        body[blade_mask] = 0
    bb = tight_bbox(body)
    if bb is not None:
        bx0, by0, bx1, by1 = bb
        Image.fromarray(body[by0:by1, bx0:bx1]).save(fp)
        tw.update({"x": bx0, "y": by0, "width": bx1 - bx0, "height": by1 - by0,
                   "grid": grid_for(bx1 - bx0, by1 - by0)})
    parts.extend(blade_parts)
    if made:
        print(f"  腿(从上衣层): 髋线 y={int(round(hip))}, 切出 {made} 段 "
              f"{int(sub_take.sum())}px, 上衣层已挖洞压暗")
    return parts


def _drop_foreign_blobs(mask, src_rgba):
    """头饰里剔除"颜色明显不是布"的独立小块。

    实测: 模型的前发 mask 会混进一块眉毛/眼窝碎片(面积 800px, 灰绿色), 它和头巾
    之间隔着描边, 属于独立连通域。按"最大那块的颜色"作为布料参考色, 把颜色差得远
    的小块剔掉。返回 (新 mask, 被剔掉的 mask)。
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 2:
        return mask, np.zeros_like(mask, bool)
    areas = sorted(((int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)),
                   reverse=True)
    big_i = areas[0][1]
    ref = src_rgba[..., :3][lab == big_i].reshape(-1, 3).mean(0)
    removed = np.zeros_like(mask, bool)
    for area, i in areas[1:]:
        if area < 30 or area > 0.5 * areas[0][0]:
            continue
        c = src_rgba[..., :3][lab == i].reshape(-1, 3).mean(0)
        if float(np.linalg.norm(c - ref)) > 40:
            removed |= (lab == i)
    return mask & ~removed, removed


def _fold_headwear(parts, out_dir, src_rgba, W, H):
    """把挂在 headwear 上的残影补丁并回主块, 头巾/头盔必须是一整片。

    模型给的头发 mask 常常盖不全头巾(帽顶容易被漏掉), 残影补偿会补出一块
    headwear_overlay —— 那就又变成两片了。这里按"原图取色 + 掩膜并集"重做一次,
    确保头饰是单块, 没有接缝也没有色差。
    """
    hw = next((p for p in parts if p["name"] == "headwear"), None)
    if hw is None:
        return parts
    extra = [p for p in parts if p["name"].startswith("headwear_overlay")]
    if not extra:
        return parts

    mask = np.zeros((H, W), bool)
    for p in [hw] + extra:
        im = np.array(Image.open(os.path.join(out_dir, p["image"])).convert("RGBA"))
        x0, y0 = int(p["x"]), int(p["y"])
        h, w = im.shape[0], im.shape[1]
        x1, y1 = min(W, x0 + w), min(H, y0 + h)
        if x1 <= x0 or y1 <= y0:
            continue
        mask[y0:y1, x0:x1] |= (im[:y1 - y0, :x1 - x0, 3] > 12)

    mask, _junk = _drop_foreign_blobs(mask, src_rgba)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return parts
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    crop = src_rgba[by0:by1, bx0:bx1].copy()
    crop[..., 3] = np.where(mask[by0:by1, bx0:bx1], crop[..., 3], 0)
    Image.fromarray(crop).save(os.path.join(out_dir, hw["image"]))
    hw.update({"x": bx0, "y": by0, "width": bx1 - bx0, "height": by1 - by0,
               "grid": grid_for(bx1 - bx0, by1 - by0)})
    for p in extra:
        try:
            os.remove(os.path.join(out_dir, p["image"]))
        except OSError:
            pass
    print(f"头饰合并: {len(extra)} 块补丁并进 headwear, 现在是一整块 "
          f"{bx1 - bx0}×{by1 - by0}")
    return [p for p in parts if p not in extra]


HEAD_WEAR_MODES = ("hair", "scarf", "helmet")     # 有头发 / 头巾(可飘) / 头盔(刚性)


def set_head_wear(coords, mode):
    """头部装饰模式 -> 写进 coords 的 cls。

    hair   : 有头发, 按名字分类(会飘)
    scarf  : 头巾, cls=cloth —— 包在头上的那圈钉死, 垂下来的部分会飘
    helmet : 头盔, cls=headgear —— 整块刚性跟随头部, 一点都不飘

    用 cls0 记住被覆盖前的 cls, 切回 hair 时能还原。
    """
    if mode not in HEAD_WEAR_MODES:
        mode = "hair"
    for p in coords.get("parts", []):
        if not any(k in p.get("name", "").lower() for k in _HAIR_NAME_KW):
            continue
        if mode != "hair":
            if "cls0" not in p:
                p["cls0"] = p.get("cls")
            p["cls"] = "cloth" if mode == "scarf" else "headgear"
        elif "cls0" in p:
            c0 = p.pop("cls0")
            if c0:
                p["cls"] = c0
            else:
                p.pop("cls", None)
    coords["head_wear"] = mode
    coords["hair"] = (mode == "hair")          # 兼容旧字段
    return coords


# 旧名字保留, 其它脚本若还调它也能用
def set_hair_mode(coords, no_hair):
    return set_head_wear(coords, "helmet" if no_hair else "hair")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"H:\codemake\see-through-repo\see-through-main\workspace\layerdiff_output\nv_f0",
                    help="see-through 输出层目录")
    ap.add_argument("--out", default=r"H:\codemake\video2spine\st_parts",
                    help="输出 parts 目录")
    ap.add_argument("--orig-image", default="",
                    help="原始输入图(用于推导画布尺寸与 pad; 推荐)")
    ap.add_argument("--canvas-w", type=int, default=0, help="原图宽(无 orig-image 时)")
    ap.add_argument("--canvas-h", type=int, default=0, help="原图高")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--no-arm-split", action="store_true",
                    help="禁用 handwear 手臂精细拆分(回退整层导出)")
    ap.add_argument("--white-min", type=int, default=300,
                    help="白底洞的最小面积(px): 小于它的近白块当高光保留, 不抠")
    ap.add_argument("--carve-head", action="store_true",
                    help="头还埋在 topwear 里时, 从身体层把头挖出来(默认关: 这步会把"
                         "两肩和胸甲一起挖进 head, 见 _carve_head_feet_from_body 注释)")
    ap.add_argument("--no-leg-from-body", action="store_true",
                    help="不从上衣层切腿(长袍角色模型把裙子+腿甲算进上衣, 不切就没有腿骨)")
    ap.add_argument("--no-arm-from-body", action="store_true",
                    help="不从上衣层切肩甲+大臂(重甲角色的大护肩会留在躯干上动不了)")
    ap.add_argument("--arm-from-body", action="store_true",
                    help="强制从上衣层切肩甲+大臂(默认只在检测到大臂缺失时才切)")
    ap.add_argument("--leg-halves", action="store_true",
                    help="把每条腿在膝高处切成上/下两片(刚性蒙皮也能屈膝; 交界留几像素重叠)")
    ap.add_argument("--knee-ratio", type=float, default=0.45,
                    help="膝盖在腿高上的位置(0=髋 1=脚), 需与生成器的腿骨比例一致")
    ap.add_argument("--leg-overlap", type=int, default=4,
                    help="腿切片在膝处的重叠像素(遮掩张开的细缝)")
    ap.add_argument("--no-leg-split", action="store_true",
                    help="禁用 legwear 左右腿拆分(回退整层, 腿就不能迈步)")
    ap.add_argument("--no-closed-eyes", action="store_true",
                    help="不生成闭眼补丁(死亡时就闭不上眼)")
    ap.add_argument("--no-fill-missing", action="store_true",
                    help="不做残影补偿(默认会把模型漏分配的部位按原图捡回来)")
    ap.add_argument("--head-wear", default="hair", choices=HEAD_WEAR_MODES,
                    help="头部装饰: hair=有头发(会飘) / scarf=头巾(顶上钉死, 下摆飘) / "
                         "helmet=头盔(整块刚性)")
    ap.add_argument("--no-hair", action="store_true",
                    help="(旧参数) 等同 --head-wear helmet")
    args = ap.parse_args()

    # 推导原图尺寸 (W,H) 与 center_square_pad 参数
    if args.orig_image and os.path.isfile(args.orig_image):
        W, H = Image.open(args.orig_image).size
    elif args.canvas_w and args.canvas_h:
        W, H = args.canvas_w, args.canvas_h
    else:
        # 给了 --orig-image 却读不到时必须叫出来: 静默退回 1280x720 会让整张图
        # 错位(部件坐标、切分区全按错画布算), 拆出来的东西面目全非却没人报错。
        if args.orig_image:
            print(f"警告: 原图读不到 -> {args.orig_image}")
        W, H = 1280, 720          # 兼容旧视频帧默认
    S = max(W, H)
    pad_x = (S - W) // 2
    pad_y = (S - H) // 2
    print(f"原图 {W}x{H} -> square {S} pad=({pad_x},{pad_y})")

    os.makedirs(args.out, exist_ok=True)
    parts = []
    covered = np.zeros((H, W), bool)      # 已被某个部件占用的像素(用于残影补偿)

    def emit_layer(disp_name, frame_layer, cls=None):
        """裁 bbox + 落盘 + 记录 part。frame_layer: (H,W,4) 原图空间。"""
        bbox = main_bbox(frame_layer)
        if bbox is None:
            print(f"  跳过(空): {disp_name}")
            return
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        crop = frame_layer[y0:y1, x0:x1]
        cover = int((crop[..., 3] > 12).sum())
        minpx = FEATURE_MIN.get(disp_name, MIN_PIXELS)   # 小特征别被当噪声扔掉
        if cover < minpx or w < 4 or h < 3:
            print(f"  跳过(太小 cover={cover} {w}x{h}): {disp_name}")
            return
        safe = disp_name.replace(" ", "_")
        # 同名部件(比如 legwear 和 footwear 都出 leg_L_up)不能写同一个文件,
        # 否则后写的把前面的覆盖掉, 两个 part 指向同一张图 -> 脚上贴图错乱
        _fname, _k = safe, 2
        while any(q["image"] == _fname + ".png" for q in parts):
            _fname = f"{safe}_{_k}"
            _k += 1
        out_png = os.path.join(args.out, _fname + ".png")
        Image.fromarray(crop).save(out_png)
        part = {"name": safe, "image": _fname + ".png",
                "x": x0, "y": y0, "width": w, "height": h,
                "grid": grid_for(w, h)}
        if cls:
            part["cls"] = cls
        parts.append(part)
        covered[y0:y1, x0:x1] |= (crop[..., 3] > 12)
        print(f"  OK {disp_name:18s} pos=({x0},{y0}) size={w}x{h} "
              f"cls={cls or '-'} cover={cover}")

    def emit_leg_group(frame_layer, label="腿部"):
        """把"两条腿(含鞋)在一层"的图拆成左右 + 上/下两段, 各自绑髋/膝骨。

        图层循环里的 legwear 和"模型漏掉、从原图捡回来的下半身"都走这里。
        返回 False 表示分不出左右(腿并在一起), 调用方自己决定怎么兜底。
        """
        segs = split_legs(frame_layer)
        if not segs:
            # 两腿被腰甲/裙甲连成一个连通域(重甲角色很常见): 按外框中线硬切左右。
            _bb = main_bbox(frame_layer)
            if _bb is None:
                return False
            _x0, _y0, _x1, _y1 = _bb
            _mid = (_x0 + _x1) // 2
            _l = frame_layer.copy()
            _l[:, _mid:] = 0
            _r = frame_layer.copy()
            _r[:, :_mid] = 0
            segs = [{"side": "L", "rgba": _l}, {"side": "R", "rgba": _r}]
            print(f"  {label}: 两腿连成一片, 按中线硬切左右")
        print(f"  {label}拆分: -> {len(segs)} 条")
        for seg in segs:
            side = seg["side"]
            if args.leg_halves:
                for sub, tag in split_leg_halves(
                        seg["rgba"], args.knee_ratio, args.leg_overlap):
                    n_before = len(parts)
                    emit_layer(f"leg_{side}_{tag}", sub, cls="leg")
                    if len(parts) > n_before:
                        # 上片跟髋骨、下片跟膝骨: 两片以膝为铰链, 刚性也能屈膝
                        parts[-1]["bind_bone"] = (
                            f"leg_{side}_up" if tag == "up"
                            else f"leg_{side}_fore")
                        parts[-1]["_src"] = label      # legwear / 腿甲 / 补腿
            else:
                emit_layer(f"leg_{side}", seg["rgba"], cls="leg")
        return True

    # 头巾/头盔: 模型会把一只头巾拆成"前布+后飘带+补丁", 这里并成一整块
    wear = args.head_wear
    if args.no_hair and wear == "hair":
        wear = "helmet"
    merge_head = (wear != "hair")
    # 原图(残影补偿 / 面部叠放自检都要用): 优先 --orig-image, 否则用模型留的 src_img.png
    src_rgba = None
    srcp = ""
    if args.orig_image and os.path.isfile(args.orig_image):
        srcp = args.orig_image
    elif os.path.isfile(os.path.join(args.src, "src_img.png")):
        srcp = os.path.join(args.src, "src_img.png")
    fg_mask = None
    if srcp:
        src_rgba = np.array(Image.open(srcp).convert("RGBA")
                            .resize((W, H), Image.LANCZOS))
        # 原图本来就带透明通道(抠好的立绘) -> 里面的纯白是画出来的(白刀面/高光),
        # 一个像素都不能动。只有"白底截图"才需要抠白洞。
        _cutout = float((src_rgba[..., 3] < 12).mean()) > 0.02
        src_rgba, fg_mask = strip_bg(src_rgba)   # 带背景的截图先抠掉背景
        # 白底截图剩下的"白洞"(烟雾/发丝之间那块白)会被涂成不透明, 部件图上就是
        # 一块块白方块。大面积纯白一定是底, 抠掉。
        _wh = (white_holes(src_rgba, max(40, args.white_min)) if not _cutout
               else np.zeros((H, W), bool))
        if _wh.any():
            src_rgba = src_rgba.copy()
            src_rgba[_wh, 3] = 0
            fg_mask = fg_mask & ~_wh
            print(f"  白底洞: 抠掉 {int(_wh.sum())}px (烟雾/发丝之间的背景白)")

    head_mask = np.zeros((H, W), bool)
    feat_mask = np.zeros((H, W), bool)     # 五官/耳朵, 不能算进头饰
    face_mask = np.zeros((H, W), bool)     # 脸皮底板(头巾常把它整块吞进去)
    cloth_mask = np.zeros((H, W), bool)    # 衣服(头巾下摆常混进肩膀那块)

    for name in Z_ORDER:
        src_png = os.path.join(args.src, name + ".png")
        if not os.path.exists(src_png):
            print(f"  跳过(无文件): {name}")
            continue
        frame_layer = map_to_frame(src_png, W, H, S, pad_x, pad_y)
        # 带了不透明背景的截图(见 strip_bg): 模型有时会把背景也划进某一层,
        # 到这里统一按"前景 mask"削掉 —— 否则部件会拖着一块底色。
        if fg_mask is not None and not fg_mask.all():
            frame_layer = frame_layer.copy()
            frame_layer[..., 3] = np.where(fg_mask, frame_layer[..., 3], 0)

        # 无头发模式: 前发/后发/头饰其实是同一只头巾(或头盔)的碎片, 合成一整块,
        # 否则会拆成"前布+后飘带+补丁"三片, 拼回去有接缝、颜色也对不上。
        if merge_head and name in _FACE_FEATURE_SLOTS:
            feat_mask |= (frame_layer[..., 3] > 12)
        if merge_head and name == "face":
            face_mask |= (frame_layer[..., 3] > 12)
        if merge_head and name in ("topwear", "bottomwear", "legwear"):
            cloth_mask |= (frame_layer[..., 3] > 12)
        if merge_head and name in _HEAD_WEAR_SLOTS:
            m = frame_layer[..., 3] > 12
            if m.any():
                # "后发"不一定真是头发。判据要稳, 否则会把头巾(fa 的整块黄布)也判成翅膀:
                #   ① 这一层面积巨大(>15% 画布);
                #   ② 同时又有一块**独立且很小**的 headwear 层(真正的头盔)。
                # 两条都满足 -> 这块"后发"是翅膀/披风, 并进头饰会被当成头盔排到最上层
                # (z=95), 把胸口手臂全罩住 —— 魔身实测就是这么变成一团黑噪点的。
                if name == "back hair" and int(m.sum()) > 0.15 * H * W:
                    _hw_area = 0
                    _hw_p = os.path.join(args.src, "headwear.png")
                    if os.path.isfile(_hw_p):
                        _hw = map_to_frame(_hw_p, W, H, S, pad_x, pad_y)
                        _hw_area = int((_hw[..., 3] > 12).sum())
                    if _hw_area and _hw_area < 0.25 * int(m.sum()):
                        emit_layer("wings", frame_layer, cls="body")
                        print(f"  后发其实是翅膀/披风({int(m.sum())}px, 头盔层只有 "
                              f"{_hw_area}px), 单独成层排在最后面")
                        continue
                head_mask |= m
                print(f"  头饰整体: 收进 {name} ({int(m.sum())}px)")
            continue

        # legwear = 两条腿(含鞋)在一层 -> 拆成左右两条, 好给腿挂骨迈步
        if name == "legwear" and not args.no_leg_split:
            if emit_leg_group(frame_layer, "腿"):
                continue
            print("  腿拆分: 分不出左右(腿并在一起?), 保持整层")

        # footwear = 两只鞋在同一层 -> 拆成左右, 各自绑到脚骨。
        # 不拆的话整双鞋会落在 root 上: 腿怎么迈, 鞋都钉在画面原地 ——
        # 看起来就是"最下面一截没连在大腿上"。
        if name == "footwear" and not args.no_leg_split:
            _fb = main_bbox(frame_layer)
            # 模型常把"整条腿甲+靴子"都塞进 footwear (这张图就是): 高度接近半身时
            # 按腿处理(拆左右+上下两段, 绑髋/膝), 否则才当纯鞋子绑脚骨。
            if (_fb is not None and (_fb[3] - _fb[1]) > 0.18 * H
                    and emit_leg_group(frame_layer, "腿甲")):
                print("  腿甲: footwear 其实是整条腿(含靴), 已按上下两段拆开")
                continue
            segs = split_legs(frame_layer)
            if len(segs) >= 2:
                print(f"  鞋拆分: footwear -> {len(segs)} 只")
                for seg in segs:
                    side = seg["side"]
                    n_before = len(parts)
                    emit_layer(f"footwear_{side}", seg["rgba"], cls="foot")
                    if len(parts) > n_before:
                        parts[-1]["bind_bone"] = f"leg_{side}_foot"
                continue
            print("  鞋拆分: 只找到一只/粘连, 保持整层(会钉在 root 上)")

        # handwear = 整条手臂 -> 精细拆分大臂/小臂/手 (左右各一)
        if name == "handwear" and not args.no_arm_split:
            depth_png = os.path.join(args.src, name + "_depth.png")
            depth = None
            if os.path.exists(depth_png):
                dd = map_to_frame(depth_png, W, H, S, pad_x, pad_y)
                depth = dd[..., 0].astype(np.float32)
            segs = split_handwear(frame_layer, depth)
            print(f"  手臂拆分: handwear -> {len(segs)} 片")
            for seg in segs:
                side = seg["side"]
                tag = f"{seg['cls']}_{side}" if side else seg["cls"]
                emit_layer(tag, seg["rgba"], cls=seg["cls"])
            continue

        n0 = len(parts)
        emit_layer(name, frame_layer)
        if name == "objects":          # 手持道具, 稍后绑到手上
            for p in parts[n0:]:
                p["is_prop"] = True

    # 大臂的 z 序看几何: 手臂长在躯干【外侧】(护臂包着胳膊、压着腰甲)时应该排在
    # 衣服前面; 贴着躯干的(被护肩/衣袖盖住)才放后面。pi 那种贴身布置和 suo 这种
    # 外挂护臂, 靠同一条规则区分不了, 必须按实际位置判断。
    _tw = next((p for p in parts if p["name"] == "topwear"), None)
    if _tw is not None and _tw["width"] > 0:
        _tcx = _tw["x"] + _tw["width"] / 2.0
        for _p in parts:
            if _p["name"].startswith("arm_upper"):
                _dd = abs(_p["x"] + _p["width"] / 2.0 - _tcx)
                if _dd > _tw["width"] * 0.32:
                    _p["_z_arm"] = 22          # 压到 topwear(21) 之后
                    print(f"  z: {_p['name']} 在躯干外侧(d={_dd:.0f}) -> 排到衣服前面")

    # 按 Z_RANK 重排 z 序(大臂到躯干之下、小臂/手到最后)
    parts.sort(key=lambda p: p.get("_z_arm", _z_rank(p["name"])))


    # 无头发模式: 把收齐的头巾/头盔碎片落成一块 headwear(颜色直接取原图, 无接缝)
    if merge_head and head_mask.any() and src_rgba is not None:
        cut = int((head_mask & feat_mask).sum())
        head_mask &= ~feat_mask
        # 模型给的"前发/头饰"mask 会把整张脸皮、肩膀那块衣服一起圈进来。
        # face / topwear 两个图层已经各自分得很干净, 直接减掉它们最准;
        # 只减交叠部分, 头巾本体(与它们不重叠的地方)一点不动。
        # 只减【颜色也接近脸/衣服】的像素: 直接减 face 层会把头盔的帽檐一起削掉
        # (帽檐正好压在脸的 bbox 里), 结果帽檐没了、额头露出来, 看着就是"盔被盖住"。
        for _m, _label in ((face_mask, "脸"), (cloth_mask, "衣服")):
            _sel = head_mask & _m
            if int(_sel.sum()) < 40:
                continue
            _col = src_rgba[..., :3][_m].reshape(-1, 3).astype(np.float32).mean(0)
            _d = np.linalg.norm(src_rgba[..., :3].astype(np.float32) - _col, axis=2)
            _drop = _sel & (_d < 38)
            if int(_drop.sum()) > 0:
                head_mask &= ~_drop
                print(f"  头饰剔除: 混进来的{_label} {int(_drop.sum())}px 已摘掉")
        head_mask, junk = _drop_foreign_blobs(head_mask, src_rgba)
        if junk.any():
            covered |= junk          # 这些像素交给下面的脸层, 别让残影补偿捡回来
            print(f"  头饰剔除: 不是布料颜色的碎块 {int(junk.sum())}px 已摘掉")
        # 布缝里夹着的衣服/皮肤小块(通常是被头巾挡住的 topwear, 图层里没有所以减不到):
        # 不删, 直接补成旁边的布色 —— 头巾是连续的一整块布, 留个棕色洞更难看,
        # 删掉又会露出后面的衣服、和上面的脸层对不上。只补小片, 大片(真的是布面
        # 暗部或第二层布)原样保留。
        _wear_rgb = src_rgba[..., :3].astype(np.float32).copy()
        _ref_px = src_rgba[..., :3][head_mask].reshape(-1, 3).astype(np.float32)
        if len(_ref_px) >= 200:
            _q = (_ref_px // 32).astype(np.int32)
            _key = _q[:, 0] * 4096 + _q[:, 1] * 64 + _q[:, 2]
            _vals, _counts = np.unique(_key, return_counts=True)
            _ref = _ref_px[_key == _vals[np.argmax(_counts)]].mean(0)
            _d = np.linalg.norm(_wear_rgb - _ref, axis=2)
            _ok = _d < 60                      # 像布
            _cloth_col = _wear_rgb[head_mask & _ok].reshape(-1, 3).mean(0)
            # 明显不是布、又不是黑描边的像素(max>100 排除描边)统一补成布色。
            # 布面的暗褶子到主色的距离在 90 上下, 不会被误补。
            _lum = _wear_rgb.max(axis=2)
            _bad = head_mask & (_d > 90) & (_lum > 100)
            _fixed = int(_bad.sum())
            if _fixed:
                _wear_rgb[_bad] = _cloth_col
                print(f"  头饰修补: 布面里的衣服/皮肤 {_fixed}px 补成布色")
        buf = np.zeros((H, W, 4), np.uint8)
        buf[head_mask] = np.dstack([_wear_rgb.astype(np.uint8),
                                    src_rgba[..., 3]])[head_mask]
        emit_layer("headwear", buf,
                   cls=("cloth" if wear == "scarf" else "headgear"))
        parts.sort(key=lambda p: p.get("_z_arm", _z_rank(p["name"])))
        print(f"头饰合并: 前发/后发/头饰 -> 一整块 headwear"
              f"({'头巾, 下摆可飘' if wear == 'scarf' else '头盔, 整块刚性'})")

    # 兜底: 模型输出里有、但 Z_ORDER 没列的图层 -> 报警, 否则会静默丢部位(如腿)
    known = {n + ".png" for n in Z_ORDER} | {"reconstruction.png"}
    leftovers = []
    for f in sorted(os.listdir(args.src)):
        if (not f.endswith(".png") or f.endswith("_depth.png")
                or f in known or f.startswith("src_")):
            continue
        a = np.array(Image.open(os.path.join(args.src, f)).convert("RGBA"))[..., 3]
        if int((a > 12).sum()) > MIN_PIXELS:
            leftovers.append(f)
    if leftovers:
        print("警告: 这些图层有内容但没被读取(会被丢掉): " + "、".join(leftovers))
        print("      把它们加进 Z_ORDER 才能参与拆图。")

    # ---- 面部叠放自检: face 是皮肤底板时不能压在 head 上 ----
    if src_rgba is not None:
        parts = _check_face_order(parts, args.out, src_rgba, W, H)

    # ---- 残影补偿 ----
    # 模型偶尔会把某块内容漏分配给任何图层(实测: 盾牌、头巾)。这些像素不属于任何
    # 部件, 合成出来就"凭空消失"。这里把原图中没被任何部件覆盖的连通块捡回来,
    # 按"与哪个部件外框重叠最多"命名成 <部件>_overlay, 直接复用该部件的骨骼。
    if not args.no_fill_missing and src_rgba is not None:
        miss = ((src_rgba[..., 3] > 12) & ~covered).astype(np.uint8)
        if int(miss.sum()) > MIN_PIXELS:
            n, lab, stats, _ = cv2.connectedComponentsWithStats(miss, 8)
            extras = []
            lower = np.zeros((H, W), bool)      # 下半身漏块(腿/鞋)
            for i in range(1, n):
                x, y, w, h, area = (int(v) for v in stats[i])
                if area < MIN_PIXELS or w < 6 or h < 6:
                    continue
                mask = (lab[y:y + h, x:x + w] == i)
                # 模型经常整个漏掉裤子/鞋(画风相近, 归不到 bottomwear/legwear),
                # 残影会把它们捡成 "xxx_overlay" 然后挂到手臂当道具 —— 腿就永远
                # 不会迈步。落在下半身的大块直接归给腿, 走正常的两段式绑定。
                if (y + h / 2.0) > H * 0.55 and h >= 24 and w >= 12:
                    lower[y:y + h, x:x + w] |= mask
                    continue
                # 纯白块不是"模型漏画的部位", 是原图里没抠干净的白底(实测华雄
                # 刀和裙摆之间那块三角白, 100% 近白)。捡回来会变成一块白贴图跟着
                # 手飞 —— 直接放过, 让它保持透明。
                _px = src_rgba[y:y + h, x:x + w][..., :3][mask]
                if _px.size and float((_px.min(axis=1) > 226).mean()) > 0.85:
                    continue
                crop = src_rgba[y:y + h, x:x + w].copy()
                crop[..., 3] = np.where(mask, crop[..., 3], 0)
                best, best_a = None, 0
                for p in parts:
                    ox = max(0, min(x + w, p["x"] + p["width"]) - max(x, p["x"]))
                    oy = max(0, min(y + h, p["y"] + p["height"]) - max(y, p["y"]))
                    if ox * oy > best_a:
                        best, best_a = p["name"], ox * oy
                # 残影落在头部上方 -> 那是头饰(头盔/帽檐), 不是道具。
                # 不拦的话会被 _attach_props 挂到手臂上, 头一动盔就跟着手飞。
                _hb = next((q for q in parts if q["name"] == "head"), None)
                _hw_flag = bool(_hb is not None and w >= 40
                                and (y + h / 2.0) < _hb["y"] + _hb["height"] * 0.75
                                and not any(q["name"] == "headwear" for q in parts))
                if _hw_flag:
                    # 帽檐被"头/脸"那层啃掉了(残影只捡回帽顶和描边): 在头盔所在
                    # 区域, 把原图里"不是肤色"的前景像素补回来, 帽檐才是实心的。
                    _fg = fg_mask if fg_mask is not None else (src_rgba[..., 3] > 12)
                    _add = _fg.copy()
                    if face_mask.any():
                        _skin = src_rgba[..., :3][face_mask].reshape(-1, 3).astype(np.float32).mean(0)
                        _dsk = np.linalg.norm(src_rgba[..., :3].astype(np.float32) - _skin, axis=2)
                        _add &= (_dsk > 30) & ~face_mask
                    _add = _add & ~covered
                    _add[:max(0, y - 8), :] = False
                    _add[min(H, y + h + 8):, :] = False
                    _add[:, :max(0, x - 4)] = False
                    _add[:, min(W, x + w + 4):] = False
                    _sub = _add[y:y + h, x:x + w]
                    if int(_sub.sum()) > 0:
                        mask |= _sub
                        print(f"  头盔补帽檐: +{int(_sub.sum())}px")
                    mask = fill_small_holes(mask)
                    crop = src_rgba[y:y + h, x:x + w].copy()   # 补洞后重做一次
                    crop[..., 3] = np.where(mask, crop[..., 3], 0)
                    _hm, _sm = split_wear_parts(crop)
                    if False and int(_hm.sum()) > 150 and int(_sm.sum()) > 150:
                        # 头盔和头巾分两片: 头巾在下(压脸)、头盔在上(压头巾)
                        for _nm, _mm, _cl in (("headwear", _hm, "headgear"),
                                              ("headscarf", _sm, "cloth")):
                            _c2 = crop.copy()
                            _c2[..., 3] = np.where(_mm, _c2[..., 3], 0)
                            Image.fromarray(_c2).save(os.path.join(args.out, _nm + ".png"))
                            extras.append((_nm, {"name": _nm, "image": _nm + ".png",
                                                 "x": x, "y": y, "width": w, "height": h,
                                                 "grid": grid_for(w, h), "cls": _cl}))
                        print(f"  头饰拆分: 头盔 {int(_hm.sum())}px + 头巾 {int(_sm.sum())}px")
                        continue
                    name = safe = "headwear"
                else:
                    name = f"{best}_overlay" if best else f"residual_{i}"
                    safe = name.replace(" ", "_")
                k = 2
                while any(e[0] == safe for e in extras) or \
                        any(p["name"] == safe for p in parts):
                    safe = f"{name.replace(' ', '_')}_{k}"
                    k += 1
                # 挂哪个骨: 名字(决定 z 序)可能挂在头发上(如头巾压在头发下),
                # 但骨头要挂到最近的"非头发"部件, 否则整片布会跟着头发飘、
                # 和底下的头皮错位。cls 写进 coords, 生成器优先用它。
                cls = None
                if _classify is not None:
                    if _classify(best or "", False) in ("hair", "ribbon"):
                        cands = [p for p in parts
                                 if _classify(p["name"], False)
                                 not in ("hair", "ribbon")]
                        b2, a2 = None, 0
                        for p in cands:
                            ox = max(0, min(x + w, p["x"] + p["width"]) - max(x, p["x"]))
                            oy = max(0, min(y + h, p["y"] + p["height"]) - max(y, p["y"]))
                            if ox * oy > a2:
                                b2, a2 = p["name"], ox * oy
                        cls = _classify(b2, False) if b2 else "body"
                    else:
                        cls = _classify(best or "", False)
                Image.fromarray(crop).save(os.path.join(args.out, safe + ".png"))
                extras.append((safe, {"name": safe, "image": safe + ".png",
                                      "x": x, "y": y, "width": w, "height": h,
                                      "grid": grid_for(w, h),
                                      "is_prop": not _hw_flag,
                                      **({"cls": cls} if cls else {})}))
                print(f"  残影 {safe:26s} pos=({x},{y}) size={w}x{h} "
                      f"cls={cls or '-'} cover={area}")
            # 腿已经从 legwear 拆出来过就别再补一遍 —— 否则 leg_L_up/low 会出现两份,
            # 骨骼/插槽同名, 脚上会叠一块位置不对的贴图(实测 quan 的"鞋贴图错")。
            _has_legs = any(q["name"].startswith("leg_") for q in parts)
            if lower.any() and not args.no_leg_split and not _has_legs:
                lbuf = np.zeros((H, W, 4), np.uint8)
                lbuf[lower] = src_rgba[lower]
                if emit_leg_group(lbuf, "下半身补腿"):
                    print(f"  下半身补腿: 模型漏掉的腿/鞋 {int(lower.sum())}px 已按原图捡回")
                else:
                    emit_layer("legwear", lbuf, cls="leg")
                    print("  下半身补腿: 分不出左右, 整块作为 legwear 保留")

            if extras:
                names = {p["name"] for p in parts}
                merged = []
                for p in parts:
                    merged.append(p)
                    merged.extend(e for nm, e in extras
                                  if nm.startswith(p["name"] + "_overlay"))
                merged.extend(e for nm, e in extras
                              if not any(nm.startswith(pn + "_overlay")
                                         for pn in names))
                parts = merged
                print(f"残影补偿: {len(extras)} 片 —— 模型漏掉的部位已按原图找回")

    parts = drop_blank_face(parts, args.out, src_rgba)
    parts = widen_headwear(parts, args.out, src_rgba, W, H, fg_mask)
    parts = demote_hair_over_face(parts, args.out, src_rgba, W, H)
    parts = trim_arm_body(parts, args.out, W, H)
    # 长袍角色: 模型把裙摆+腿甲算进上衣层、legwear 是空的 -> 没有腿就做不出行走。
    if not args.no_leg_split and not args.no_leg_from_body:
        parts = split_legs_from_body(parts, args.out, W, H,
                                     args.knee_ratio, args.leg_overlap)
    # 肩甲+大臂从上衣层切出来(重甲角色模型只给肘以下, 肩甲永远留在躯干上)。
    # 必须排在 trim_arm_body 之后: 那一步按"和上衣层重叠且同色"剔身体, 会把手
    # 刚切出来的肩甲当成"上衣的边"再削掉一半。
    if not args.no_arm_from_body:
        parts = split_arm_from_body(parts, args.out, W, H, force=args.arm_from_body)
    parts = _attach_props(parts)
    if merge_head and src_rgba is not None:
        parts = _fold_headwear(parts, args.out, src_rgba, W, H)
    if not args.no_closed_eyes:
        parts = _make_closed_eyes(parts, args.out)

    # 一条腿的碎片来自多个图层时(legwear 出大腿/小腿, footwear 出小腿+鞋):
    # 先把同侧的几块叠成"整条腿", 再按膝高重新切成上/下两段 —— 直接按名字二选一
    # 会丢脚或者缺大腿, 叠完不重切则上下两段互相重叠、屈膝时会错位。
    for _side in ("L", "R"):
        _pieces = [q for q in parts if str(q["name"]).startswith(f"leg_{_side}_")]
        if len(_pieces) < 3:
            continue
        _x0 = min(q["x"] for q in _pieces)
        _y0 = min(q["y"] for q in _pieces)
        _x1 = max(q["x"] + q["width"] for q in _pieces)
        _y1 = max(q["y"] + q["height"] for q in _pieces)
        _cv = np.zeros((int(_y1 - _y0), int(_x1 - _x0), 4), np.uint8)
        for q in _pieces:
            _im = np.array(Image.open(os.path.join(args.out, q["image"])).convert("RGBA"))
            _ox, _oy = int(q["x"] - _x0), int(q["y"] - _y0)
            _h, _w = _im.shape[:2]
            _m = _im[..., 3] > 12
            _tile = _cv[_oy:_oy + _h, _ox:_ox + _w]
            _tile[_m] = _im[_m]
            _cv[_oy:_oy + _h, _ox:_ox + _w] = _tile
        for q in _pieces:
            parts.remove(q)
        # 腿要是"从上衣层切出来的", 新片段也得排在衣服前面, 否则重切后被衣服盖住
        _legz = max((q.get("_z_arm", 0) for q in _pieces), default=0) or None
        for _sub, _tag in split_leg_halves(_cv, args.knee_ratio, args.leg_overlap):
            _bb = main_bbox(_sub)
            if _bb is None:
                continue
            _sx0, _sy0, _sx1, _sy1 = _bb
            _crop = _sub[_sy0:_sy1, _sx0:_sx1]
            _fname = f"leg_{_side}_{_tag}"
            Image.fromarray(_crop).save(os.path.join(args.out, _fname + ".png"))
            _np = {"name": _fname, "image": _fname + ".png",
                   "x": int(_x0) + _sx0, "y": int(_y0) + _sy0,
                   "width": _sx1 - _sx0, "height": _sy1 - _sy0,
                   "grid": grid_for(_sx1 - _sx0, _sy1 - _sy0),
                   "cls": "leg",
                   "bind_bone": (f"leg_{_side}_up" if _tag == "up"
                                 else f"leg_{_side}_fore")}
            if _legz:
                _np["_z_arm"] = _legz
            parts.append(_np)
        print(f"  腿合并重切: leg_{_side} 由 {len(_pieces)} 块并成整条腿, "
              f"再按膝高切成上/下两段")

    # 同名部件去重(留面积大的): 同名会导致同名骨/同名 slot, 贴图会叠错
    _seen = {}
    _uniq = []
    for _p in sorted(parts, key=lambda q: -(q["width"] * q["height"])):
        if _p["name"] in _seen:
            print(f"  部件去重: {_p['name']} 重复({int(_p['width'])}x{int(_p['height'])}) 已丢弃")
            continue
        _seen[_p["name"]] = True
        _uniq.append(_p)
    if len(_uniq) != len(parts):
        parts = _uniq

    # 腿片段是"合并后重新 append"的, 会跑到列表最后 -> 画在最上面盖住衣服。
    # 这里按 z 序再排一次(leg=10 < topwear=21, 腿在衣服后面)。
    parts.sort(key=lambda q: q.get("_z_arm", _z_rank(q["name"])))

    # 腿**不切脚**。切出来的脚片只有靴尖(28x16), 靴筒还留在小腿里 —— 靴子被劈成
    # 两截, 用户看到的就是"腿的下半部分分解不正确"。gen_idle_bones 里 cls=="leg"
    # 本来就有"膝->踝"的权重渐变, 一条整腿照样能让脚掌跟踝骨动, 不需要切。
    # 头和脚被包在身体层里(头被头盔包、脚被胫甲包)时, 从 topwear 里挖出来
    parts = _carve_head_feet_from_body(parts, args.out, W, H,
                                        carve_head=getattr(args, 'carve_head', False))
    parts = _strip_cloth_from_props(parts, args.out)
    parts = _back_layer_to_bottom(parts)
    parts = _rescue_crushed_headwear(parts, args.out)
    parts = _face_over_headwear(parts)
    # 五官错位/头被撑成"头+肩甲+胸甲"的兜底: 判据见函数注释, 放在所有合并之后
    parts = _repair_face_split(parts, args.out, args.src, src_rgba, W, H,
                               frame=(S, pad_x, pad_y))
    parts = _clean_prop_junk(parts, args.out, src_rgba, W, H)
    parts = _repaint_from_source(parts, args.out, src_rgba, W, H)
    parts = _prune_headwear_junk(parts, args.out, W, H)

    for _p in parts:                     # 临时标记不要写进 coords
        _p.pop("_z_arm", None)
        _p.pop("_src", None)

    coords = {
        # 拆图结果版本号: 界面/生成器据此判断"这是不是旧版结果"(旧版没有腿、
        # 头巾没合并、没有道具绑定), 避免拿旧数据生成出一堆缺件的动画。
        "schema": PARTS_SCHEMA,
        "fps": args.fps,
        "canvas": {"w": W, "h": H},
        "transform": {"s": 1.0, "tx": 0.0, "ty": 0.0},
        "parts": parts,
    }
    set_head_wear(coords, wear)
    if merge_head:
        n = sum(1 for p in parts if p.get("cls") in ("cloth", "headgear"))
        print(f"头饰模式 {wear}: {n} 块" +
              ("头巾(顶部钉死/下摆飘)" if wear == "scarf" else "头盔(刚性跟随头部)"))
    coords_path = os.path.join(args.out, "coords.json")
    with open(coords_path, "w", encoding="utf-8") as f:
        json.dump(coords, f, ensure_ascii=False, indent=2)

    # 原图带底色时, 最后再过一遍: 把部件里残留的底色像素抠掉
    if src_rgba is not None:
        _raw = np.array(Image.open(srcp).convert("RGBA")) if srcp else None
        _bg = None
        if _raw is not None and (_raw[..., 3] < 12).mean() < 0.02:
            _pts = np.concatenate([_raw[:6, :6, :3].reshape(-1, 3),
                                   _raw[:6, -6:, :3].reshape(-1, 3),
                                   _raw[-6:, :6, :3].reshape(-1, 3),
                                   _raw[-6:, -6:, :3].reshape(-1, 3)])
            _bg = np.median(_pts, axis=0)
            print(f"  原图底色 {_bg.astype(int).tolist()}")
        if _bg is not None:
            strip_part_bg(parts, args.out, _bg)

    # 白底残留在"手臂和身体之间/腿之间/烟雾围出来的洞"里 —— 这些白被角色包在中间,
    # 不碰部件边界, 上面那套"贴边才删"的规则管不到, 导出后就是一块块死白 + 一圈白边。
    # 按画布统一算一次 mask 再回切到各部件(白常被劈在好几个部件里, 逐部件判断会漏边)。
    try:
        import strip_white as _SW
        _n = _SW.clean_dir(args.out, src_img=srcp if 'srcp' in dir() else None)
        if _n:
            print(f"  去纯白残留: {_n}px")
            coords["parts"] = json.load(open(os.path.join(args.out, "coords.json"),
                                             encoding="utf-8"))["parts"]
            parts = coords["parts"]
    except Exception as _ex:                      # noqa: BLE001
        print(f"  去纯白残留跳过: {_ex}")

    # 清掉上一轮遗留、这张表里已经不用的部件图(比如被判成"白板脸"的 face.png)。
    # 不清的话: 预览/别的工具按目录扫图, 会把废弃的白脸/旧手臂当成部件列出来。
    _used = {p["image"] for p in parts} | {"coords.json", "headwear_full.png"}
    for _f in sorted(os.listdir(args.out)):
        if _f.endswith(".png") and _f not in _used:
            try:
                os.remove(os.path.join(args.out, _f))
                print(f"  清理废弃部件图: {_f}")
            except OSError:
                pass

    print(f"\n部件 {len(parts)} 个 -> {args.out}")
    print(f"coords -> {coords_path}")

    # 最后一步: 头饰(头盔/头巾)从原图补齐边, 并摘掉混进来的五官/额头皮肤。
    # 必须放在所有合并、补丁之后 —— 放中间会被后面的步骤覆盖回去(踩过这个坑)。
    if srcp and wear in ("scarf", "helmet"):
        try:
            import importlib
            import fix_headwear
            importlib.reload(fix_headwear)
            root = os.path.dirname(os.path.abspath(args.out))
            fix_headwear.main(root, srcp)
        except Exception as ex:                            # noqa: BLE001
            print(f"  头饰修整跳过: {ex}")


if __name__ == "__main__":
    main()
