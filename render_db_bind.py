"""把导出的龙骨数据按"绑定姿态"离线渲染成 PNG, 用来确认贴图/图集 uv 是否正确。

不依赖龙骨编辑器: 自己算骨骼世界矩阵 -> 网格顶点 -> 从 _tex.png 采样。
渲染出来的图应该和 split_preview 里的角色一致; 如果整片空白/糊成一块,
说明 uv 或图集对不上(编辑器里就会显示 Missing)。

用法: python render_db_bind.py <ske.json> [out.png] [--scale 1.0]
"""
import json
import os
import sys

import numpy as np
from PIL import Image


def trs(x, y, rot=0.0, sx=1.0, sy=1.0, skx=0.0, sky=0.0):
    r = np.radians(rot)
    a, b = np.cos(r), np.sin(r)
    # DragonBones: x 轴向右, y 轴向下, 正角为顺时针
    m = np.array([[a * sx, -b * sy, x],
                  [b * sx, a * sy, y],
                  [0, 0, 1]], dtype=np.float64)
    return m


def bind_worlds(bones):
    by_name = {b["name"]: b for b in bones}
    world = {}
    order = []

    def walk(b):
        t = b.get("transform") or {}
        local = trs(t.get("x", 0.0), t.get("y", 0.0), t.get("rotation", 0.0),
                    t.get("scX", 1.0), t.get("scY", 1.0))
        p = b.get("parent")
        world[b["name"]] = (world[p] @ local) if p in world else local
        order.append(b["name"])
        for c in bones:
            if c.get("parent") == b["name"]:
                walk(c)

    for b in bones:
        if not b.get("parent"):
            walk(b)
    for b in by_name.values():            # 兜底: 孤立骨
        if b["name"] not in world:
            world[b["name"]] = trs(*[0, 0])
    return world


def sample_tri(dst, atlas, P, Q):
    """把源三角形 Q 的像素画到目标三角形 P (重心坐标, 最近邻采样)。"""
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
    u = w0 * qa[0] + w1 * qb[0] + w2 * qc[0]
    v = w0 * qa[1] + w1 * qb[1] + w2 * qc[1]
    u = np.clip(u.astype(np.int64), 0, atlas.shape[1] - 1)
    v = np.clip(v.astype(np.int64), 0, atlas.shape[0] - 1)
    src = atlas[v, u]
    sub = dst[y0:y1 + 1, x0:x1 + 1]
    a = src[..., 3:4].astype(np.float32) / 255.0
    sub[..., :3] = src[..., :3] * a + sub[..., :3] * (1 - a)
    sub[..., 3:4] = np.maximum(sub[..., 3:4], src[..., 3:4])
    dst[y0:y1 + 1, x0:x1 + 1] = sub


def render(ske_p, out_p=None, scale=1.0):
    ske = json.load(open(ske_p, encoding="utf-8"))
    d = os.path.dirname(ske_p)
    name = os.path.basename(ske_p)[:-len("_ske.json")]
    tex = json.load(open(os.path.join(d, name + "_tex.json"), encoding="utf-8"))
    atlas = np.array(Image.open(os.path.join(d, tex.get("imagePath") or name + "_tex.png"))
                     .convert("RGBA"), dtype=np.float32)
    st = {s["name"]: s for s in tex["SubTexture"]}

    arm = ske["armature"][0]
    world = bind_worlds(arm["bone"])
    aabb = arm.get("aabb") or {}
    pad = 8
    W = int(math_ceil(aabb.get("width", 336))) + pad * 2
    H = int(math_ceil(aabb.get("height", 336))) + pad * 2
    ox, oy = pad - aabb.get("x", 0), pad - aabb.get("y", 0)
    canvas = np.zeros((H, W, 4), dtype=np.float32)

    skin = arm["skin"][0]
    draws = []
    for sl in skin["slot"]:
        dp = sl["display"][0]
        key = dp.get("path") or dp["name"]
        if key not in st:
            continue
        bone = None
        for s2 in arm["slot"]:
            if s2["name"] == sl["name"]:
                bone = s2.get("parent") or "root"
        M = world.get(bone, np.eye(3))
        if sl["display"][0].get("weights"):
            # 加权网格的顶点已经是"绑定姿态世界坐标"(slotPose=单位阵), 直接用
            M = np.eye(3)
        vs = dp.get("vertices") or []
        uvs = dp.get("uvs") or []
        tris = dp.get("triangles") or []
        r = st[key]
        if not vs and dp.get("type") in (0, "image"):
            # 纯图片显示: 以 SubTexture 尺寸 + pivot 定位, 按两个三角形画矩形
            tf = dp.get("transform") or {}
            pv = tf.get("pivot") or {"x": 0.5, "y": 0.5}
            tx, ty = tf.get("x", 0.0), tf.get("y", 0.0)
            w0, h0 = r["width"], r["height"]
            loc = [(tx - pv.get("x", 0.5) * w0, ty - pv.get("y", 0.5) * h0),
                   (tx + (1 - pv.get("x", 0.5)) * w0, ty - pv.get("y", 0.5) * h0),
                   (tx + (1 - pv.get("x", 0.5)) * w0, ty + (1 - pv.get("y", 0.5)) * h0),
                   (tx - pv.get("x", 0.5) * w0, ty + (1 - pv.get("y", 0.5)) * h0)]
            pts = [tuple((M @ np.array([x, y, 1.0]))[:2] + np.array([ox, oy])) for x, y in loc]
            ux0, uy0 = r["x"], r["y"]
            quv = [(ux0, uy0), (ux0 + w0, uy0), (ux0 + w0, uy0 + h0), (ux0, uy0 + h0)]
            for idx in ([0, 1, 2], [0, 2, 3]):
                draws.append(([pts[i] for i in idx], [quv[i] for i in idx], sl["name"]))
            continue
        pts = []
        for i in range(0, len(vs), 2):
            p = M @ np.array([vs[i], vs[i + 1], 1.0])
            pts.append((p[0] + ox, p[1] + oy))
        quv = []
        for i in range(0, len(uvs), 2):
            quv.append((uvs[i] * tex["width"], uvs[i + 1] * tex["height"]))
        for t in range(0, len(tris), 3):
            idx = tris[t:t + 3]
            if max(idx) >= len(pts) or max(idx) >= len(quv):
                continue
            draws.append(([pts[i] for i in idx], [quv[i] for i in idx], sl["name"]))

    for P, Q, nm in draws:
        if max(q[0] for q in Q) > tex["width"] or max(q[1] for q in Q) > tex["height"]:
            print(f"  uv 越界: {nm}")
        sample_tri(canvas, atlas, P, Q)

    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGBA")
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    out_p = out_p or os.path.join(d, name + "_bind_render.png")
    img.save(out_p)

    # 覆盖率: 判断有没有真的画出东西
    alpha = np.array(img)[..., 3]
    cov = float((alpha > 8).mean())
    print(f"  渲染 {out_p} {img.size} 覆盖率 {cov*100:.1f}%")
    return out_p, cov


def math_ceil(v):
    import math
    return math.ceil(float(v))


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return
    ske_p = a[0]
    out_p = a[1] if len(a) > 1 and not a[1].startswith("--") else None
    render(ske_p, out_p)


if __name__ == "__main__":
    main()
