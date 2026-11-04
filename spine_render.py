"""从 Spine json + atlas 渲染一帧, 用来确认打出来的包(骨架/图集/uv)是否对得上。

和 render_db_bind.py 是同一套思路, 但按 Spine 3.8 的规矩:
    - y 轴向上, rotation 逆时针; 屏幕坐标要翻转 y
    - slot 的绘制顺序 = slots 数组顺序
    - 网格顶点是骨局部坐标, 直接乘骨世界矩阵

用法:
    python spine_render.py <name>.json [out.png] [--anim walk] [--time 0.25]
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image


def trs(x, y, rot=0.0, sx=1.0, sy=1.0):
    r = math.radians(rot)
    a, b = math.cos(r), math.sin(r)
    return np.array([[a * sx, -b * sy, x],
                     [b * sx, a * sy, y],
                     [0, 0, 1]], dtype=np.float64)


def bind_worlds(bones):
    by = {b["name"]: b for b in bones}
    world = {}

    def w(name):
        if name in world:
            return world[name]
        b = by[name]
        t = trs(b.get("x", 0.0), b.get("y", 0.0), b.get("rotation", 0.0),
                b.get("scaleX", 1.0), b.get("scaleY", 1.0))
        p = b.get("parent")
        world[name] = (w(p) @ t) if p else t
        return world[name]

    for b in bones:
        w(b["name"])
    return world


def _kf_val(fr, key):
    if key == "rotate":
        return float(fr.get("angle", fr.get("value", 0.0)))
    if key == "translate":
        return (float(fr.get("x", 0.0)), float(fr.get("y", 0.0)))
    return (float(fr.get("x", 1.0)), float(fr.get("y", 1.0)))


def _kf_at(tl, t, key):
    """时间线线性采样。spine 的 curve 只影响缓动形状, 验收帧不追求这个。"""
    if not tl:
        return None
    if t <= tl[0]["time"]:
        return _kf_val(tl[0], key)
    if t >= tl[-1]["time"]:
        return _kf_val(tl[-1], key)
    for i in range(len(tl) - 1):
        a, b = tl[i], tl[i + 1]
        if a["time"] <= t <= b["time"]:
            u = (t - a["time"]) / max(1e-9, b["time"] - a["time"])
            va, vb = _kf_val(a, key), _kf_val(b, key)
            if key == "rotate":
                return va + (vb - va) * u
            return (va[0] + (vb[0] - va[0]) * u, va[1] + (vb[1] - va[1]) * u)
    return None


def anim_worlds(bones, anim, t):
    """带动画采样的世界矩阵(和 measure_motion.world_at 同一套算法)。"""
    by = {b["name"]: b for b in bones}
    tracks = (anim or {}).get("bones", {})
    world = {}

    def w(name):
        if name in world:
            return world[name]
        b = by[name]
        x, y = b.get("x", 0.0), b.get("y", 0.0)
        rot = b.get("rotation", 0.0)
        sx, sy = b.get("scaleX", 1.0), b.get("scaleY", 1.0)
        tl = tracks.get(name) or {}
        v = _kf_at(tl.get("translate"), t, "translate")
        if v:
            x += v[0]
            y += v[1]
        v = _kf_at(tl.get("rotate"), t, "rotate")
        if v is not None:
            rot += v
        v = _kf_at(tl.get("scale"), t, "scale")
        if v:
            sx *= v[0]
            sy *= v[1]
        mat = trs(x, y, rot, sx, sy)
        p = b.get("parent")
        world[name] = (w(p) @ mat) if p else mat
        return world[name]

    for b in bones:
        w(b["name"])
    return world


def parse_atlas(p):
    lines = [l.rstrip("\n") for l in open(p, encoding="utf-8")]
    i = 0
    png = lines[0].strip()
    regions = {}
    i = 1
    while i < len(lines) and ":" in lines[i]:
        i += 1
    while i < len(lines):
        name = lines[i].strip()
        if not name:
            i += 1
            continue
        i += 1
        r = {"name": name, "rotate": False, "xy": (0, 0), "size": (0, 0)}
        while i < len(lines) and lines[i].startswith(" "):
            k, _, v = lines[i].strip().partition(":")
            k, v = k.strip(), v.strip()
            if k == "xy":
                r["xy"] = tuple(int(t) for t in v.split(","))
            elif k == "size":
                r["size"] = tuple(int(t) for t in v.split(","))
            elif k == "rotate":
                r["rotate"] = (v == "true")
            i += 1
        regions[name] = r
    return png, regions


def sample_tri(dst, atlas, P, Q):
    x0 = max(0, int(np.floor(min(p[0] for p in P))))
    x1 = min(dst.shape[1] - 1, int(np.ceil(max(p[0] for p in P))))
    y0 = max(0, int(np.floor(min(p[1] for p in P))))
    y1 = min(dst.shape[0] - 1, int(np.ceil(max(p[1] for p in P))))
    if x1 < x0 or y1 < y0:
        return
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    (ax, ay), (bx, by), (cx, cy) = P
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(den) < 1e-9:
        return
    w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
    w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -0.002) & (w1 >= -0.002) & (w2 >= -0.002)
    if not inside.any():
        return
    (qa, qb, qc) = Q
    u = (w0 * qa[0] + w1 * qb[0] + w2 * qc[0]).astype(np.int64)
    v = (w0 * qa[1] + w1 * qb[1] + w2 * qc[1]).astype(np.int64)
    u = np.clip(u, 0, atlas.shape[1] - 1)
    v = np.clip(v, 0, atlas.shape[0] - 1)
    src = atlas[v, u]
    sub = dst[y0:y1 + 1, x0:x1 + 1]
    a = src[..., 3:4].astype(np.float32) / 255.0
    sub[..., :3] = src[..., :3] * a + sub[..., :3] * (1 - a)
    sub[..., 3:4] = np.maximum(sub[..., 3:4], src[..., 3:4])
    dst[y0:y1 + 1, x0:x1 + 1] = sub


def render(json_p, out_p=None, anim=None, t=0.0):
    doc = json.load(open(json_p, encoding="utf-8"))
    d = os.path.dirname(os.path.abspath(json_p))
    name = os.path.splitext(os.path.basename(json_p))[0]
    png_rel, regions = parse_atlas(os.path.join(d, name + ".atlas"))
    atlas = np.array(Image.open(os.path.join(d, png_rel)).convert("RGBA"), dtype=np.float32)

    sk = doc["skeleton"]
    cw, ch = int(sk.get("width") or 336), int(sk.get("height") or 336)
    pad = 8
    canvas = np.zeros((ch + pad * 2, cw + pad * 2, 4), dtype=np.float32)

    world = bind_worlds(doc["bones"])
    if anim:
        anims = doc.get("animations") or {}
        if anim in anims:
            world = anim_worlds(doc["bones"], anims[anim], t)
        else:
            print(f"  警告: 没有动画 {anim}（有: {', '.join(anims) or '无'}）")
    att = {}
    for skin in doc.get("skins", []):
        for sname, a in (skin.get("attachments") or {}).items():
            att.setdefault(sname, a)

    drawn = 0
    for slot in doc["slots"]:
        sname = slot["name"]
        a = att.get(sname)
        if not a:
            continue
        aname = slot.get("attachment") or next(iter(a))
        ad = a.get(aname)
        if not ad:
            continue
        ad = dict(ad)
        ad["name"] = aname
        rname = ad.get("path") or aname
        reg = regions.get(rname)
        if not reg:
            print(f"  atlas 里没有 region: {rname}")
            continue
        M = world.get(slot["bone"], np.eye(3))
        vs = ad.get("vertices") or []
        uvs = ad.get("uvs") or []
        tris = ad.get("triangles") or []

        if not vs and not uvs:
            # 普通图片附件: 用 width/height 与 (x,y) 画一个矩形
            w0, h0 = ad.get("width", reg["size"][0]), ad.get("height", reg["size"][1])
            ox_, oy_ = ad.get("x", 0.0), ad.get("y", 0.0)
            sx = ad.get("scaleX", 1.0)
            sy = ad.get("scaleY", 1.0)
            loc = [(ox_ - w0 / 2 * sx, oy_ + h0 / 2 * sy), (ox_ + w0 / 2 * sx, oy_ + h0 / 2 * sy),
                   (ox_ + w0 / 2 * sx, oy_ - h0 / 2 * sy), (ox_ - w0 / 2 * sx, oy_ - h0 / 2 * sy)]
            pts = []
            for x, y in loc:
                p = M @ np.array([x, y, 1.0])
                pts.append((p[0] + pad, ch - p[1] + pad))
            rx, ry = reg["xy"]
            rw, rh = reg["size"]
            quv = [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]
            for idx in ([0, 1, 2], [0, 2, 3]):
                sample_tri(canvas, atlas, [pts[i] for i in idx], [quv[i] for i in idx])
                drawn += 1
            continue

        pts = []
        if len(vs) == len(uvs):
            # 无权重网格: 直接是骨局部坐标
            for i in range(0, len(vs), 2):
                p = M @ np.array([vs[i], vs[i + 1], 1.0])
                pts.append((p[0] + pad, ch - p[1] + pad))
        else:
            # 加权网格: [boneCount, boneIdx, x, y, weight, ...] 逐顶点打包
            bone_names = [b["name"] for b in doc["bones"]]
            i = 0
            while i < len(vs):
                bc = int(vs[i]); i += 1
                if bc == 0:
                    p = M @ np.array([vs[i], vs[i + 1], 1.0])
                    pts.append((p[0] + pad, ch - p[1] + pad))
                    i += 2
                    continue
                acc = np.zeros(3)
                for _ in range(bc):
                    bi = int(vs[i])
                    bx, by, wgt = vs[i + 1], vs[i + 2], vs[i + 3]
                    i += 4
                    W = world[bone_names[bi]] if bi < len(bone_names) else np.eye(3)
                    acc = acc + wgt * (W @ np.array([bx, by, 1.0]))
                pts.append((acc[0] + pad, ch - acc[1] + pad))
        rx, ry = reg["xy"]
        rw, rh = reg["size"]
        quv = []
        for i in range(0, len(uvs), 2):
            quv.append((rx + uvs[i] * rw, ry + uvs[i + 1] * rh))
        for k in range(0, len(tris), 3):
            idx = tris[k:k + 3]
            if max(idx) >= len(pts) or max(idx) >= len(quv):
                continue
            sample_tri(canvas, atlas, [pts[i] for i in idx], [quv[i] for i in idx])
            drawn += 1

    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGBA")
    out_p = out_p or os.path.join(d, name + "_preview.png")
    img.save(out_p)
    alpha = np.array(img)[..., 3]
    print(f"  渲染 {out_p} {img.size} 三角形 {drawn} 覆盖率 {(alpha > 8).mean()*100:.1f}%")
    return out_p


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {}
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a.startswith("--"):
            k = a[2:]
            v = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else ""
            opt[k] = v
    render(pos[0], pos[1] if len(pos) > 1 else None,
           anim=opt.get("anim") or None, t=float(opt.get("time") or 0.0))
