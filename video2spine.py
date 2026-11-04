#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video2spine.py

PSD 部件 PNG + 固定坐标 + 动画视频  ->  Spine 3.8 json (mesh + deform 动画)

全自动:
  1. 为每个部件按 grid 自动生成网格 mesh (hull-first 顶点排序)
  2. 组装 Spine 3.8 骨架 / slots / skins json
  3. 用 LK 光流追踪每个顶点在视频里的逐帧位置
  4. 把逐帧位移烘焙成 deform timeline
  5. 关键帧抽稀(基于线性插值误差)
  6. 写出 result.json, 并把部件 PNG 拷到 images 目录

用法:
  python video2spine.py --parts ./psd_parts --coords coords.json \
      --video video.mp4 --out ./out/result.json --grid 8x8 --tol 1.0

坐标约定:
  coords.json 里每个部件的 x,y 是该部件左上角在【视频像素】中的位置 (y 向下)。
  width/height 不填则用 PNG 原始尺寸。
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("缺少 opencv-python: pip install opencv-python")

try:
    from PIL import Image
except ImportError:
    sys.exit("缺少 Pillow: pip install Pillow")


# --------------------------------------------------------------------------
# 网格生成
# --------------------------------------------------------------------------
def build_grid_mesh(px, py, w, h, gx, gy):
    """
    生成规则网格 mesh。返回:
      verts_video : [(vx, vy), ...]  顶点在视频像素坐标 (y 向下)
      uvs         : [u, v, ...]      flat, 与 verts 对应 (top-left 原点)
      triangles   : [i0,i1,i2, ...]
      hull        : int  外壳顶点数 (前 hull 个顶点为外壳, 顺时针)
      edges       : [a,b, ...]       外壳边 (顶点索引 *2)

    顶点排序: 先外壳(顺时针一圈)再内部, 满足 Spine 对 hull 的要求。
    """
    # 先按 (i,j) 收集所有点的视频坐标 + uv
    def point(i, j):
        vx = px + (i / gx) * w
        vy = py + (j / gy) * h
        u = i / gx
        v = j / gy
        return (vx, vy), (u, v)

    order = []          # [(i,j), ...] 最终顶点顺序
    seen = set()

    def add(i, j):
        if (i, j) not in seen:
            seen.add((i, j))
            order.append((i, j))

    # --- 外壳: 顺时针 ---
    for i in range(0, gx + 1):          # 上边 j=0
        add(i, 0)
    for j in range(1, gy + 1):          # 右边 i=gx
        add(gx, j)
    for i in range(gx - 1, -1, -1):     # 下边 j=gy
        add(i, gy)
    for j in range(gy - 1, 0, -1):      # 左边 i=0
        add(0, j)
    hull = len(order)

    # --- 内部 ---
    for j in range(1, gy):
        for i in range(1, gx):
            add(i, j)

    index_of = {ij: k for k, ij in enumerate(order)}

    verts_video = []
    uvs = []
    for (i, j) in order:
        (vx, vy), (u, v) = point(i, j)
        verts_video.append((vx, vy))
        uvs.extend([u, v])

    # --- 三角形: 每个格子两片 ---
    triangles = []
    for j in range(gy):
        for i in range(gx):
            a = index_of[(i, j)]
            b = index_of[(i + 1, j)]
            c = index_of[(i + 1, j + 1)]
            d = index_of[(i, j + 1)]
            triangles += [a, c, b, a, d, c]

    # --- 外壳边 ---
    edges = []
    for k in range(hull):
        edges += [k * 2, ((k + 1) % hull) * 2]

    return verts_video, uvs, triangles, hull, edges


def _resample_boundary(poly, step):
    """沿闭合多边形按 step 间距重采样, 长边插点 -> 外壳点分布均匀。"""
    out = []
    n = len(poly)
    for k in range(n):
        a = poly[k]
        b = poly[(k + 1) % n]
        out.append(a)
        seg = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        m = int(seg // step)
        for t in range(1, m + 1):
            f = t / (m + 1)
            out.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    return np.array(out, float)


def build_shape_mesh(px, py, w, h, gx, gy, mask):
    """按 alpha 形状自适应布点网格 (替代纯正方形)。
    mask: bbox 区域的 alpha (HxW uint8, y下)。返回与 build_grid_mesh 同四件套:
      verts(世界像素 y下) / uvs(归一 0~1, 与 build_grid_mesh 同, 由上层 uv_box 重映射)
      / triangles / hull / edges。失败返回 None 让上层回退正方形网格。
    原理: 轮廓做外壳(均匀重采样) + 内部规则网格点(仅保留落 mask 内者) ->
    Delaunay 三角化 -> 剔质心在透明区的三角形。透明角不布点/不被拉伸。"""
    try:
        import cv2 as _cv
        from scipy.spatial import Delaunay
    except Exception:                                  # noqa: BLE001
        return None
    H, W = mask.shape[:2]
    if W < 4 or H < 4:
        return None
    m = (mask > 24).astype(np.uint8)
    if int(m.sum()) < 16:
        return None
    cnts, _ = _cv.findContours(m, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    areas = [(float(_cv.contourArea(c)), c) for c in cnts]
    amax = max(a for a, _ in areas)
    if amax < 0.02 * W * H:
        return None
    # 保留所有岛(头发细缕/辫尖/飘带末端等与主体分离的块)。
    # 阈值必须很小: 头巾垂布这类"主体旁边挂一条"的岛, 面积往往只有主体的 4% 左右,
    # 用 6% 会把整条垂布丢掉 —— 部件图里有、动画里没有(用户: "缺图片的左下部分")。
    # 所以按面积从大到小取, 再用顶点预算 600 兜底(超了就不加最小的, 而不是整体回退)。
    total = float(m.sum())
    thr = max(0.010 * amax, 0.0008 * total, 12.0)
    keep = [(a, c) for a, c in areas if a >= thr]
    keep.sort(key=lambda t: -t[0])
    if not keep:
        return None
    step = max(4.0, max(W / max(1, gx), H / max(1, gy)))
    boundary_list = []                                 # 每岛重采样边界
    poly_list = []                                     # 每岛轮廓(内部点/三角剔除)
    nb = 0
    for _area, c in keep:                              # 大岛优先, 预算满了跳过小岛
        peri = _cv.arcLength(c, True)
        # eps 系数偏小: 细长发束/辫尖的轮廓折点不被简化掉 (0.012 会丢 ~11% 像素)
        approx = _cv.approxPolyDP(c, max(1.0, 0.004 * peri), True)
        approx = approx.reshape(-1, 2).astype(float)
        if len(approx) < 3:
            continue
        b = _resample_boundary(approx, step)
        if len(b) < 3:
            continue
        if boundary_list and nb + len(b) > 600:        # 顶点预算: 只跳小岛, 不回退
            continue
        boundary_list.append(b)
        poly_list.append(c)
        nb += len(b)
    if not boundary_list:
        return None
    if nb < 3:
        return None
    boundary = np.vstack(boundary_list)
    interior = []
    margin = step * 0.4
    for j in range(1, gy):
        for i in range(1, gx):
            x = i / gx * W
            y = j / gy * H
            for c in poly_list:
                if _cv.pointPolygonTest(c, (float(x), float(y)), True) > margin:
                    interior.append((x, y))
                    break
    pts = np.vstack([boundary, np.array(interior, float)]) if interior \
        else boundary
    try:
        d = Delaunay(pts)
    except Exception:                                  # noqa: BLE001
        return None
    tris = []
    for tri in d.simplices:
        cc = pts[tri].mean(axis=0)
        for c in poly_list:
            if _cv.pointPolygonTest(c, (float(cc[0]), float(cc[1])), False) >= 0:
                tris += [int(tri[0]), int(tri[1]), int(tri[2])]
                break
    if len(tris) < 3:
        return None
    verts = [(px + float(x), py + float(y)) for (x, y) in pts]
    uvs = []
    for (x, y) in pts:
        uvs.extend([float(x) / W, float(y) / H])
    # edges: 各岛边界各自闭环(边界点已按岛序排在 pts 最前, 偏移连续)
    edges = []
    off = 0
    for b in boundary_list:
        nbi = len(b)
        for k in range(nbi):
            edges += [(off + k) * 2, (off + (k + 1) % nbi) * 2]
        off += nbi
    return verts, uvs, tris, nb, edges


# --------------------------------------------------------------------------
# Spine 坐标转换
# --------------------------------------------------------------------------
def video_to_spine(vx, vy, canvas_h):
    """视频像素(y向下) -> Spine 世界(y向上)。root 在原点, 顶点即世界坐标。"""
    return vx, canvas_h - vy


# --------------------------------------------------------------------------
# 光流追踪
# --------------------------------------------------------------------------
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def track_points(video_path, init_pts):
    """
    init_pts: (N,2) float32  视频像素坐标
    返回 tracks: (F, N, 2)   每帧每点像素坐标。丢失的点沿用上一帧。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"打不开视频: {video_path}")

    ok, frame = cap.read()
    if not ok:
        sys.exit("视频空")

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts = init_pts.reshape(-1, 1, 2).astype(np.float32)

    tracks = [init_pts.copy()]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nxt, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, pts, None, **LK_PARAMS
        )
        st = st.reshape(-1)
        cur = pts.reshape(-1, 2).copy()
        good = nxt.reshape(-1, 2)
        mask = st == 1
        cur[mask] = good[mask]          # 成功的更新, 失败的沿用
        tracks.append(cur.copy())
        pts = cur.reshape(-1, 1, 2).astype(np.float32)
        prev_gray = gray

    cap.release()
    return np.stack(tracks, axis=0)     # (F, N, 2)


# --------------------------------------------------------------------------
# 关键帧抽稀 (线性插值误差驱动)
# --------------------------------------------------------------------------
def decimate(values, tol):
    """
    values: (F, D)  每帧一个向量
    返回保留的帧索引列表。线性插值与原值误差 <= tol。
    """
    F = values.shape[0]
    if F <= 2:
        return list(range(F))
    kept = [0]
    anchor = 0
    i = 1
    while i < F:
        a = values[anchor]
        b = values[i]
        span = i - anchor
        fits = True
        for j in range(anchor + 1, i):
            t = (j - anchor) / span
            interp = a + (b - a) * t
            if np.max(np.abs(values[j] - interp)) > tol:
                fits = False
                break
        if not fits:
            kept.append(i - 1)
            anchor = i - 1
            # i 不变, 重新从新 anchor 探测
        else:
            i += 1
    if kept[-1] != F - 1:
        kept.append(F - 1)
    return kept


# --------------------------------------------------------------------------
# deform vertices 修剪 (去首尾全零)
# --------------------------------------------------------------------------
def trim_deform(flat, eps=1e-4):
    """flat: list[float] 偏移。返回 (offset, trimmed_list) 或 (None, None) 全零。"""
    arr = np.array(flat, dtype=float)
    nz = np.where(np.abs(arr) > eps)[0]
    if len(nz) == 0:
        return None, None
    start = int(nz[0])
    end = int(nz[-1]) + 1
    return start, arr[start:end].tolist()


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="部件 PNG 目录")
    ap.add_argument("--coords", required=True, help="coords.json 路径")
    ap.add_argument("--video", required=True, help="动画视频")
    ap.add_argument("--out", required=True, help="输出 json 路径")
    ap.add_argument("--grid", default="8x8", help="默认网格密度 NxM")
    ap.add_argument("--tol", type=float, default=1.0, help="抽稀容差(像素)")
    ap.add_argument("--fps", type=float, default=0, help="覆盖 coords 里的 fps")
    args = ap.parse_args()

    g_def = args.grid.lower().split("x")
    GX, GY = int(g_def[0]), int(g_def[1])

    with open(args.coords, "r", encoding="utf-8") as f:
        coords = json.load(f)

    fps = args.fps or coords.get("fps", 30)

    # 画布高度: 用视频高度
    cap = cv2.VideoCapture(args.video)
    canvas_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    canvas_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # ---- 为每个部件建 mesh, 收集全局追踪点 ----
    parts = []          # 每个: dict(name, slot, mesh信息, 顶点全局区间)
    all_init = []       # 全局顶点初始视频坐标
    for p in coords["parts"]:
        name = p["name"]
        png = p.get("image", name + ".png")
        png_path = os.path.join(args.parts, png)
        if not os.path.exists(png_path):
            sys.exit(f"找不到部件图: {png_path}")
        im = Image.open(png_path)
        iw, ih = im.size
        x = float(p["x"])
        y = float(p["y"])
        w = float(p.get("width", iw))
        h = float(p.get("height", ih))
        gx, gy = p.get("grid", [GX, GY])
        is_bird = bool(p.get("bird", False))

        verts_video, uvs, triangles, hull, edges = build_grid_mesh(
            x, y, w, h, gx, gy
        )

        # 鸟层 v1: 不追踪(不进全局点集), setup 顶点直接用静态位置
        if is_bird:
            span = None
            static_verts = verts_video
        else:
            start = len(all_init)
            all_init.extend(verts_video)
            end = len(all_init)
            span = (start, end)
            static_verts = None

        # 拷贝图到 images
        shutil.copy(png_path, os.path.join(img_dir, png))

        parts.append(dict(
            name=name,
            attachment=os.path.splitext(png)[0],
            uvs=uvs,
            triangles=triangles,
            hull=hull,
            edges=edges,
            nverts=len(verts_video),
            span=span,
            static_verts=static_verts,
            bird=is_bird,
            iw=iw, ih=ih,
        ))

    init_pts = np.array(all_init, dtype=np.float32)

    # ---- 追踪 ----
    print(f"追踪 {len(init_pts)} 个顶点 ...")
    tracks = track_points(args.video, init_pts)   # (F, N, 2)
    F = tracks.shape[0]
    print(f"视频 {F} 帧, fps={fps}")

    # ---- 组装 Spine json 骨架 ----
    spine = {
        "skeleton": {
            "spine": "3.8.86",
            "x": 0, "y": 0,
            "width": canvas_w, "height": canvas_h,
            "images": "./images/", "audio": "",
        },
        "bones": [{"name": "root"}],
        "slots": [],
        "skins": [{"name": "default", "attachments": {}}],
        "animations": {},
    }

    skin_att = spine["skins"][0]["attachments"]

    for p in parts:
        slot = {"name": p["name"], "bone": "root", "attachment": p["attachment"]}
        if p["bird"]:
            slot["color"] = "ffffff00"   # v1: 鸟初始隐藏
        spine["slots"].append(slot)
        # setup pose 顶点 (spine 世界坐标)
        if p["bird"]:
            src = p["static_verts"]
        else:
            s, e = p["span"]
            src = all_init[s:e]
        verts_spine = []
        for (vx, vy) in src:
            sx, sy = video_to_spine(vx, vy, canvas_h)
            verts_spine.extend([sx, sy])

        skin_att[p["name"]] = {
            p["attachment"]: {
                "type": "mesh",
                "uvs": p["uvs"],
                "triangles": p["triangles"],
                "vertices": verts_spine,
                "hull": p["hull"],
                "edges": p["edges"],
                "width": p["iw"],
                "height": p["ih"],
            }
        }

    # ---- 烘焙 deform ----
    deform_skin = {}
    for p in parts:
        if p["bird"]:
            continue
        s, e = p["span"]
        nv = p["nverts"]
        # 每帧偏移向量 (相对首帧, spine 坐标): dx, -dy
        # tracks[:, s:e, :] -> (F, nv, 2)
        seg = tracks[:, s:e, :]                  # 视频像素
        base = seg[0]                            # (nv, 2)
        offs = seg - base[None, :, :]            # (F, nv, 2) 视频位移
        # 转 spine: dx 不变, dy 取反
        offs[:, :, 1] *= -1.0
        flat = offs.reshape(F, nv * 2)           # (F, 2nv)

        keep = decimate(flat, args.tol)

        keyframes = []
        for fi in keep:
            t = fi / fps
            offset, trimmed = trim_deform(flat[fi].tolist())
            kf = {"time": round(t, 5)}
            if trimmed is not None:
                kf["offset"] = offset
                kf["vertices"] = [round(v, 4) for v in trimmed]
            keyframes.append(kf)

        deform_skin.setdefault(p["name"], {})[p["attachment"]] = keyframes

    spine["animations"]["anim"] = {"deform": {"default": deform_skin}}

    # ---- 写出 ----
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(spine, f, ensure_ascii=False, separators=(",", ":"))

    total_kf = sum(len(v[p2]) for v in deform_skin.values() for p2 in v)
    print(f"完成 -> {args.out}")
    print(f"部件 {len(parts)} 个, deform 关键帧合计 {total_kf}")
    print(f"图片在 {img_dir}")


if __name__ == "__main__":
    main()
