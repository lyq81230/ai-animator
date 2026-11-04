# -*- coding: utf-8 -*-
"""极简 DragonBones 5.5 播放器(绑定姿态), 用来核对导出的 _ske.json 是否自洽。

用法: python db_player.py <cocos目录> [--anim walk] [--time 0] [--out x.png]
输出: 贴图 + 骨骼(圆圈+三角) 叠在一起的图, 方便一眼看出"骨骼和贴图对不对得上"。
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from spine_render import sample_tri


def load(d):
    ske = [f for f in os.listdir(d) if f.endswith("_ske.json")][0]
    tex = [f for f in os.listdir(d) if f.endswith("_tex.json")][0]
    png = [f for f in os.listdir(d) if f.endswith("_tex.png")][0]
    S = json.load(open(os.path.join(d, ske), encoding="utf-8"))
    T = json.load(open(os.path.join(d, tex), encoding="utf-8"))
    atlas = np.array(Image.open(os.path.join(d, png)).convert("RGBA"))
    return S, T, atlas


def rot_of(tr):
    if "rotate" in tr or "skew" in tr:
        return float(tr.get("rotate", 0.0))
    if "skX" in tr or "skY" in tr:
        return float(tr.get("skY", 0.0))
    return 0.0


def bone_worlds(bones, anim_data=None, time=0.0, fps=24):
    """返回 {name: (x, y, rot, sx, sy)} 世界变换(含累积缩放)。

    缩放必须算: 生成器用"骨缩到 0.001"来隐藏闭眼补丁, 不算缩放的话
    预览图上闭眼贴图会盖在脸上, 看着像角色一直闭着眼。
    """
    by = {b["name"]: b for b in bones}
    anim = {}
    if anim_data:
        for e in anim_data:
            anim[e["name"]] = e

    def sample(frames, key, default=0.0):
        if not frames:
            return default
        # 简化: 取第一帧(帧 0)
        f = frames[0]
        return float(f.get(key, default))

    out = {}

    def rec(name):
        if name in out:
            return out[name]
        b = by[name]
        tr = b.get("transform") or {}
        x, y = float(tr.get("x", 0.0)), float(tr.get("y", 0.0))
        rot = rot_of(tr)
        sx = float(tr.get("scX", 1.0))
        sy = float(tr.get("scY", 1.0))
        a = anim.get(name) or {}
        if a.get("translateFrame"):
            f = a["translateFrame"][0]
            x += float(f.get("x", 0.0))
            y += float(f.get("y", 0.0))
        if a.get("rotateFrame"):
            f = a["rotateFrame"][0]
            rot += float(f.get("rotate", 0.0))
        if a.get("scaleFrame"):
            f = a["scaleFrame"][0]
            sx *= float(f.get("x", 1.0))
            sy *= float(f.get("y", 1.0))
        par = b.get("parent")
        if par:
            px, py, pr, psx, psy = rec(par)
            r = math.radians(pr)
            lx, ly = x * psx, y * psy
            wx = px + (lx * math.cos(r) - ly * math.sin(r))
            wy = py + (lx * math.sin(r) + ly * math.cos(r))
            out[name] = (wx, wy, pr + rot, psx * sx, psy * sy)
        else:
            out[name] = (x, y, rot, sx, sy)
        return out[name]

    for b in bones:
        rec(b["name"])
    return out


def draw(d, out_p, use_anim=None, draw_bones=True):
    S, T, atlas = load(d)
    REG = {st["name"]: (st["x"], st["y"], st["width"], st["height"])
           for st in (T.get("SubTexture") or [])}
    arm = S["armature"][0]
    W = int(S.get("armature", [{}])[0].get("aabb", {}).get("width") or 0)
    if not W:
        W = T.get("width") or 336
    Ht = int(arm.get("aabb", {}).get("height") or T.get("height") or 336)
    fps = S.get("frameRate") or 24

    ad = None
    if use_anim:
        a = arm["animation"].get(use_anim)
        if a:
            ad = a.get("bone") or []

    bw = bone_worlds(arm["bone"], ad)

    slot_parent = {s2["name"]: s2.get("parent") for s2 in arm["slot"]}
    canvas = np.zeros((Ht, W, 4), np.float32)
    PA = np.array([atlas.shape[1], atlas.shape[0]])

    for s in arm["skin"][0]["slot"]:
        for disp in s["display"]:
            verts = disp.get("vertices")
            uvs = disp.get("uvs")
            tris = disp.get("triangles")
            if not verts or not uvs or not tris:
                # 图片显示(type 0): 没有网格, 直接用区域矩形 + transform + pivot
                key = disp.get("path") or disp.get("name")
                if key in REG and not verts:
                    rx, ry, rw, rh = REG[key]
                    tr = disp.get("transform") or {}
                    pv = tr.get("pivot") or {"x": 0.0, "y": 0.0}
                    par = slot_parent.get(s["name"])
                    if par and par in bw:
                        bx, by, br, bsx, bsy = bw[par]
                    else:
                        bx, by, br, bsx, bsy = 0.0, 0.0, 0.0, 1.0, 1.0
                    # 图片跟着父骨走: 位移按父骨缩放, 尺寸也乘父骨缩放,
                    # 这样"骨缩到 0.001"的闭眼补丁会缩成一个点(看不见), 和编辑器一致。
                    lx = float(tr.get("x") or 0.0)
                    ly = float(tr.get("y") or 0.0)
                    rad = math.radians(br)
                    cx = bx + (lx * bsx * math.cos(rad) - ly * bsy * math.sin(rad))
                    cy = by + (lx * bsx * math.sin(rad) + ly * bsy * math.cos(rad))
                    dw, dh = rw * bsx, rh * bsy
                    x0 = cx - float(pv.get("x") or 0.0) * dw
                    y0 = cy - float(pv.get("y") or 0.0) * dh
                    P = [(x0, y0), (x0 + dw, y0), (x0 + dw, y0 + dh), (x0, y0 + dh)]
                    Q = [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]
                    for t in ((0, 1, 2), (0, 2, 3)):
                        sample_tri(canvas, atlas, [P[i] for i in t], [Q[i] for i in t])
                continue
            n = len(verts) // 2
            sp = disp.get("slotPose") or [1, 0, 0, 1, 0, 0]
            P = [(verts[2 * i] + sp[4], verts[2 * i + 1] + sp[5]) for i in range(n)]
            if not disp.get("weights"):
                # 刚性网格: 顶点是 "部件中心(相对父骨) ± 50" 的 100 单位归一化空间,
                # 真实像素尺寸由 width/height 提供 (像素 = 顶点 * 尺寸 / 100)。
                par = slot_parent.get(s["name"])
                st1 = REG.get(disp.get("path") or disp.get("name")) or (0, 0, 1, 1)
                w1, h1 = float(st1[2]) or 1.0, float(st1[3]) or 1.0
                xs = verts[0::2]; ys = verts[1::2]
                cx = (min(xs) + max(xs)) / 2.0
                cy = (min(ys) + max(ys)) / 2.0
                if par and par in bw:
                    bx, by = bw[par][0], bw[par][1]
                else:
                    bx, by = 0.0, 0.0
                # 顶点 = (图片内像素 - 尺寸/2) * 100/尺寸 + 中心(相对父骨)
                # 反解: 相对父骨像素 = (顶点 - 中心) * 尺寸/100 + 中心
                P = [(bx + (verts[2 * i] - cx) * w1 / 100.0 + cx,
                      by + (verts[2 * i + 1] - cy) * h1 / 100.0 + cy)
                     for i in range(n)]
            # uv 两种写法都要支持:
            #   整张图集归一化(标准 DragonBones) / 本格 0~1(DragonBones Pro 编辑器)
            st0 = REG.get(disp.get("path") or disp.get("name"))
            su = max(uvs[0::2]) - min(uvs[0::2])
            sv = max(uvs[1::2]) - min(uvs[1::2])
            local = (st0 is not None and abs(su - 1) < 1e-3 and abs(sv - 1) < 1e-3)
            if local:
                rx, ry, rw, rh = st0
                Q = [(rx + float(uvs[2 * i]) * rw, ry + float(uvs[2 * i + 1]) * rh)
                     for i in range(n)]
            else:
                Q = [(float(uvs[2 * i]) * PA[0], float(uvs[2 * i + 1]) * PA[1])
                     for i in range(n)]
            for t in range(0, len(tris), 3):
                a, b, c = tris[t], tris[t + 1], tris[t + 2]
                if P[a] is None or P[b] is None or P[c] is None:
                    continue
                sample_tri(canvas, atlas, [P[a], P[b], P[c]], [Q[a], Q[b], Q[c]])

    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    if draw_bones:
        dr = ImageDraw.Draw(img)
        for b in arm["bone"]:
            x, y, r = bw[b["name"]][:3]
            ln = float(b.get("length") or 0.0)
            rad = math.radians(r)
            ex, ey = x + ln * math.cos(rad), y + ln * math.sin(rad)
            dr.line([(x, y), (ex, ey)], fill=(255, 60, 60, 255), width=2)
            dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 220, 0, 255), width=2)
            dr.polygon([(ex, ey),
                        (x + (ln * 0.8) * math.cos(rad - 0.25), y + (ln * 0.8) * math.sin(rad - 0.25)),
                        (x + (ln * 0.8) * math.cos(rad + 0.25), y + (ln * 0.8) * math.sin(rad + 0.25))],
                       fill=(120, 200, 255, 255))
        for b in arm["bone"]:
            x, y, r = bw[b["name"]][:3]
            dr.text((x + 3, y - 10), b["name"], fill=(255, 255, 255, 255))
    if out_p:
        img.save(out_p)
    return img


if __name__ == "__main__":
    d = sys.argv[1]
    out = None
    anim = None
    for i, a in enumerate(sys.argv):
        if a == "--out":
            out = sys.argv[i + 1]
        if a == "--anim":
            anim = sys.argv[i + 1]
    draw(d, out or os.path.join(d, "_db_preview.png"), anim)
    print("ok")
