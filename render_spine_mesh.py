# -*- coding: utf-8 -*-
"""把 Spine 包按"绑定姿态"渲染成一张图，用来肉眼检查网格/uv/权重是否真的对。

做法: 逐三角形把图集上的三角形仿射映射到骨架空间 —— 和引擎里 mesh 的绘制方式一致。
用法: python render_spine_mesh.py output/fa/spine [--out xx.png]
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


def _sample(track, t, default, keys):
    """在 spine 的关键帧数组里线性取值(时间单位秒)。"""
    if not track:
        return list(default)
    pts = sorted(track, key=lambda k: k.get("time", 0.0))

    def gv(kf, key, dflt):
        # 4.0 把 rotate 的 "angle" 改名成 "value"; 两版都认, 否则读 4.0 包时
        # 所有旋转都取默认值 0 —— 渲染出来是一具"不会转"的假动画。
        if key in kf:
            return kf[key]
        if key == "angle" and "value" in kf:
            return kf["value"]
        if key == "value" and "angle" in kf:
            return kf["angle"]
        return dflt

    if t <= pts[0].get("time", 0.0):
        return [gv(pts[0], k, default[i]) for i, k in enumerate(keys)]
    for a, b in zip(pts, pts[1:]):
        t0, t1 = a.get("time", 0.0), b.get("time", 0.0)
        if t0 <= t <= t1:
            f = (t - t0) / max(1e-6, t1 - t0)
            return [gv(a, k, default[i]) * (1 - f) + gv(b, k, default[i]) * f
                    for i, k in enumerate(keys)]
    return [gv(pts[-1], k, default[i]) for i, k in enumerate(keys)]


def bone_matrices(bones, anim=None, t=None):
    idx = {b["name"]: i for i, b in enumerate(bones)}
    cache = {}
    tracks = (anim or {}).get("bones") or {}

    def world(i):
        if i in cache:
            return cache[i]
        b = bones[i]
        x, y = b.get("x", 0.0), b.get("y", 0.0)
        r = math.radians(b.get("rotation", 0.0))
        sx, sy = b.get("scaleX", 1.0), b.get("scaleY", 1.0)
        tr = tracks.get(b["name"]) or {}
        if t is not None and tr:
            dx, dy = _sample(tr.get("translate"), t, (0.0, 0.0), ("x", "y"))
            x, y = x + dx, y + dy
            if tr.get("rotate"):
                r += math.radians(_sample(tr.get("rotate"), t, (0.0,), ("angle",))[0])
            if tr.get("scale"):
                sc = _sample(tr.get("scale"), t, (1.0, 1.0), ("x", "y"))
                sx, sy = sx * sc[0], sy * sc[1]
        c, s = math.cos(r), math.sin(r)
        M = np.array([[c * sx, -s * sy, x], [s * sx, c * sy, y], [0, 0, 1.0]])
        p = b.get("parent")
        if p and p in idx:
            M = world(idx[p]) @ M
        cache[i] = M
        return M

    return idx, world


def mesh_points(att, world):
    """加权顶点 -> 骨架空间坐标列表。"""
    v = att["vertices"]
    i = 0
    pts = []
    while i < len(v):
        c = int(v[i])
        acc = np.zeros(2)
        for k in range(c):
            b = i + 1 + 4 * k
            bi = int(v[b])
            xa, ya, w = float(v[b + 1]), float(v[b + 2]), float(v[b + 3])
            p = world(bi) @ np.array([xa, ya, 1.0])
            acc += np.array([p[0], p[1]]) * w
        pts.append(acc)
        i += 1 + 4 * c
    return pts


def parse_atlas(p):
    W = H = 0
    regions, cur = {}, None
    for line in open(p, encoding="utf-8"):
        s = line.rstrip("\n")
        if s.startswith("size:"):
            W, H = [int(v) for v in s.split(":")[1].strip().split(",")]
            cur = None
        elif s.startswith("  ") and cur:
            k = s.strip()
            if k.startswith("xy:"):
                x, y = [int(v) for v in k.split(":")[1].strip().split(",")]
                regions[cur] = [x, y, 0, 0]
            elif k.startswith("size:"):
                w, h = [int(v) for v in k.split(":")[1].strip().split(",")]
                regions[cur][2:] = [w, h]
        elif ":" not in s and not s.lower().endswith(".png"):
            cur = s.strip()
        else:
            cur = None
    return regions, W, H


def render(d, out=None, ss=2, anim=None, t=None):
    js = [f for f in os.listdir(d) if f.endswith(".json")][0]
    at = [f for f in os.listdir(d) if f.endswith(".atlas")][0]
    png = [f for f in os.listdir(d) if f.endswith(".png")][0]
    J = json.load(open(os.path.join(d, js), encoding="utf-8"))
    regions, TW, TH = parse_atlas(os.path.join(d, at))
    atlas = Image.open(os.path.join(d, png)).convert("RGBA")

    CW = int(J["skeleton"].get("width") or 0) or 512
    CH = int(J["skeleton"].get("height") or 0) or 512
    canvas = Image.new("RGBA", (CW * ss, CH * ss), (0, 0, 0, 0))

    ANIM = (J.get("animations") or {}).get(anim) if anim else None
    _idx, world = bone_matrices(J["bones"], ANIM, t)
    _bidx = {}
    for _i, _b in enumerate(J["bones"]):
        _bidx[_b["name"]] = _i
    name2att = {}
    for skin in J["skins"]:
        for sname, att in (skin.get("attachments") or {}).items():
            name2att[sname] = att

    for sl in J["slots"]:
        sname = sl["name"]
        aname = sl.get("attachment")
        if not aname:
            continue
        att = (name2att.get(sname) or {}).get(aname)
        if not att:
            continue
        if att.get("type") == "mesh":
            pts = mesh_points(att, world)
            uv = att["uvs"]
            tri = att["triangles"]
        elif att.get("type") in (None, "region"):
            # 普通图片附件: 四角 = 中心 ± 半宽/半高, 挂在槽位骨骼上
            bi = _bidx.get(sl.get("bone"))
            if bi is None:
                continue
            M = world(bi)
            rw = att.get("width") or (regions.get(aname) or [0, 0, 0, 0])[2]
            rh = att.get("height") or (regions.get(aname) or [0, 0, 0, 0])[3]
            cx, cy = float(att.get("x", 0)), float(att.get("y", 0))
            # Spine 的 y 向上, 而贴图 v=0 在顶部: uv(0,0) 对应本地【上】边
            corners = [(cx - rw / 2, cy + rh / 2), (cx + rw / 2, cy + rh / 2),
                       (cx + rw / 2, cy - rh / 2), (cx - rw / 2, cy - rh / 2)]
            pts = [tuple((M @ np.array([x, y, 1.0]))[:2]) for x, y in corners]
            uv = [0, 0, 1, 0, 1, 1, 0, 1]
            tri = [0, 1, 2, 0, 2, 3]
        else:
            continue
        # spine y 向上 -> 图片 y 向下
        P = [(p[0] * ss, (CH - p[1]) * ss) for p in pts]
        # 网格 uv 是【相对自己那块区域】的 0~1(Spine 3.8/4.0 都如此), 采样时要加区域偏移
        _r = regions.get(aname)
        if _r is None:
            continue
        _rx, _ry, _rw, _rh = _r
        U = [(_rx + uv[2 * i] * _rw, _ry + uv[2 * i + 1] * _rh)
             for i in range(len(pts))]
        for t in range(0, len(tri), 3):
            i, j, k = tri[t], tri[t + 1], tri[t + 2]
            src = np.array([U[i], U[j], U[k]], float)
            dst = np.array([P[i], P[j], P[k]], float)
            # 仿射: src -> dst
            A = np.vstack([src.T, np.ones(3)])
            B = np.vstack([dst.T, np.ones(3)])
            # 退化三角形(三点共线 / 面积≈0)没法求仿射, 直接跳过。
            # 大幅度动作时被两个骨拉散的三角形会塌成一条线, 以前这里会抛
            # LinAlgError: Singular matrix 把整次渲染打断。
            _e1 = dst[1] - dst[0]
            _e2 = dst[2] - dst[0]
            if abs(_e1[0] * _e2[1] - _e1[1] * _e2[0]) < 0.35:
                continue
            try:
                M = B @ np.linalg.inv(A)
                inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                continue
            x0 = max(0, int(math.floor(min(dst[:, 0]))))
            y0 = max(0, int(math.floor(min(dst[:, 1]))))
            x1 = min(canvas.size[0], int(math.ceil(max(dst[:, 0]))) + 1)
            y1 = min(canvas.size[1], int(math.ceil(max(dst[:, 1]))) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            # PIL 的 transform 是"目标 -> 源", 所以取逆。注意传进去的是**相对裁剪框**
            # 的局部坐标, 所以平移项要加上裁剪原点: t = inv[2] + inv[0]*x0 + inv[1]*y0
            coef = (inv[0, 0], inv[0, 1], inv[0, 2] + inv[0, 0] * x0 + inv[0, 1] * y0,
                    inv[1, 0], inv[1, 1], inv[1, 2] + inv[1, 0] * x0 + inv[1, 1] * y0)
            piece = atlas.transform((x1 - x0, y1 - y0), Image.AFFINE, coef,
                                    resample=Image.BILINEAR)
            # 必须按三角形裁剪: transform 返回的是整个包围盒, 盒子四角的像素属于
            # 图集里别的部件 —— 不遮掉就会"漏"出一块碎图(看起来像部件飞出去了)。
            m = Image.new("L", piece.size, 0)
            ImageDraw.Draw(m).polygon(
                [(float(dst[j, 0]) - x0, float(dst[j, 1]) - y0) for j in range(3)],
                fill=255)
            a = piece.getchannel("A").point(lambda v: v)
            piece.putalpha(Image.composite(a, Image.new("L", a.size, 0), m))
            canvas.alpha_composite(piece, (x0, y0))

    canvas = canvas.resize((CW, CH), Image.LANCZOS)
    out = out or os.path.join(d, "_mesh_render.png")
    canvas.save(out)
    print("渲染 ->", out, canvas.size)
    return out


if __name__ == "__main__":
    o = None
    if "--out" in sys.argv:
        o = sys.argv[sys.argv.index("--out") + 1]
    a = sys.argv[sys.argv.index("--anim") + 1] if "--anim" in sys.argv else None
    tt = float(sys.argv[sys.argv.index("--time") + 1]) if "--time" in sys.argv else None
    render(sys.argv[1], o, anim=a, t=tt)
