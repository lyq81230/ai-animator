# -*- coding: utf-8 -*-
"""体检 Spine 包里的网格数据：uv 是否落在图集格内、权重/三角/索引是否合法。

用法: python check_spine_mesh.py output/fa/spine
"""
import json
import os
import sys


def parse_atlas(p):
    """返回 ({region: (x,y,w,h)}, W, H)。只认 rotate:false 的常规格式。"""
    W = H = 0
    regions = {}
    cur = None
    for line in open(p, encoding="utf-8"):
        s = line.rstrip("\n")
        if not s.strip():
            continue
        if s.startswith("size:"):
            W, H = [int(v) for v in s.split(":")[1].strip().split(",")]
            cur = None
            continue
        if s.startswith("  "):
            if cur and s.strip().startswith("xy:"):
                x, y = [int(v) for v in s.split(":")[1].strip().split(",")]
                regions[cur] = [x, y, 0, 0]
            elif cur and s.strip().startswith("size:"):
                w, h = [int(v) for v in s.split(":")[1].strip().split(",")]
                regions[cur][2:] = [w, h]
            continue
        if s.endswith(".png") or ":" in s:
            cur = None
            continue
        cur = s.strip()
    return {k: tuple(v) for k, v in regions.items()}, W, H


def verts_of(att):
    """解析 Spine 3.8 加权顶点 -> [(bone_indices, weights, x, y)]。"""
    v = att["vertices"]
    i = 0
    out = []
    while i < len(v):
        c = int(v[i])
        idx, wts, x, y = [], [], 0.0, 0.0
        for k in range(c):
            b = i + 1 + 4 * k
            idx.append(int(v[b]))
            x, y = float(v[b + 1]), float(v[b + 2])
            wts.append(float(v[b + 3]))
        out.append((idx, wts, x, y))
        i += 1 + 4 * c
    return out, i


def problems(d):
    """返回 (问题附件数, 附件总数)。给批量脚本做质量闸门用。"""
    js = [f for f in os.listdir(d) if f.endswith(".json")][0]
    at = [f for f in os.listdir(d) if f.endswith(".atlas")][0]
    J = json.load(open(os.path.join(d, js), encoding="utf-8"))
    regions, W, H = parse_atlas(os.path.join(d, at))
    bones = [b["name"] for b in J["bones"]]
    bad = total = 0
    for _grp, items in J["skins"][0]["attachments"].items():
        for name, a in items.items():
            total += 1
            # 武器已改成普通图片附件(region): 没有 uvs 是正常的, 不算问题
            if a.get("type") in (None, "region") and not a.get("uvs"):
                total -= 1
                continue
            if a.get("type") != "mesh" or not a.get("uvs"):
                bad += 1
                continue
            uv = a["uvs"]
            u0, u1 = min(uv[0::2]), max(uv[0::2])
            v0, v1 = min(uv[1::2]), max(uv[1::2])
            # 网格 uv 是"区域局部坐标"(0~1), Spine 3.8/4.0 都一样 —— 运行时自己会
            # 加上 region.u 再映射。以前这里按"整页坐标必须落在自己格子里"校验,
            # 那正是我们写错的写法, 反过来把正确数据判成坏。
            if name not in regions:
                bad += 1
                continue
            if u0 < -0.01 or u1 > 1.01 or v0 < -0.01 or v1 > 1.01:
                bad += 1
                continue
            vs, used = verts_of(a)
            if used != len(a["vertices"]):
                bad += 1
                continue
            okw = True
            for idx, wts, _x, _y in vs:
                if any(b < 0 or b >= len(bones) for b in idx) \
                        or abs(sum(wts) - 1.0) > 0.02:
                    okw = False
                    break
            tri = a["triangles"]
            if not okw or (tri and max(tri) >= len(vs)):
                bad += 1
                continue
            # edges: 官方导出器写的是【顶点序号 * 2】(hull=4 -> [0,2,2,4,4,6,0,6],
            # 已用官方 spineboy 3.8/4.0 原件核对)。所以合法范围是 0 .. 2*(顶点数-1),
            # 并且必须是偶数、成对出现。之前按"顶点索引"校验过一次, 那是错的。
            ed = a.get("edges") or []
            if ed:
                if len(ed) % 2 or max(ed) >= 2 * len(vs) or min(ed) < 0:
                    bad += 1
                elif any(v % 2 for v in ed):
                    bad += 1
    return bad, total


def main(d):
    js = [f for f in os.listdir(d) if f.endswith(".json")][0]
    at = [f for f in os.listdir(d) if f.endswith(".atlas")][0]
    J = json.load(open(os.path.join(d, js), encoding="utf-8"))
    regions, W, H = parse_atlas(os.path.join(d, at))
    bones = [b["name"] for b in J["bones"]]
    print(f"{js}  图集 {W}x{H}  {len(regions)} 个 region  骨骼 {len(bones)}")

    bad = 0
    print("%-13s %5s %5s %-22s %-22s %s"
          % ("slot", "顶点", "三角", "uv 范围", "图集格范围", "结论"))
    for grp, items in J["skins"][0]["attachments"].items():
        for name, a in items.items():
            if a.get("type") != "mesh":
                print("  %-11s 不是 mesh: %s" % (name, a.get("type")))
                bad += 1
                continue
            uvs = a["uvs"]
            u0, u1 = min(uvs[0::2]), max(uvs[0::2])
            v0, v1 = min(uvs[1::2]), max(uvs[1::2])
            r = regions.get(name)
            notes = []
            if r is None:
                notes.append("图集里没有该 region")
            else:
                x, y, w, h = r
                e = (x / W, (x + w) / W, y / H, (y + h) / H)
                # 判据: uv 必须**落在**本格内(采样才不会串到邻居)。
                # 网格可以只覆盖格子的一部分(部件图里带透明边时很正常), 那不算错。
                tol = 0.004
                if (u0 < e[0] - tol or u1 > e[1] + tol
                        or v0 < e[2] - tol or v1 > e[3] + tol):
                    notes.append("uv 超出图集格")
                elif (abs(u0 - e[0]) > tol or abs(u1 - e[1]) > tol
                        or abs(v0 - e[2]) > tol or abs(v1 - e[3]) > tol):
                    notes.append("子区域(留白)")
            vs, used_len = verts_of(a)
            if used_len != len(a["vertices"]):
                notes.append("顶点解析长度不符")
            for idx, wts, _x, _y in vs:
                if any(b < 0 or b >= len(bones) for b in idx):
                    notes.append("骨骼索引越界")
                    break
                if abs(sum(wts) - 1.0) > 0.02:
                    notes.append("权重和不等于 1")
                    break
            tri = a["triangles"]
            if tri and max(tri) >= len(vs):
                notes.append("三角索引越界")
            if isinstance(a.get("hull"), int):
                # Spine 3.8: hull 是"凸包顶点个数"(标量), 不是索引表
                if a["hull"] > len(vs):
                    notes.append("hull 个数超过顶点数")
            elif a.get("hull") and max(a["hull"]) >= len(vs):
                notes.append("hull 索引越界")
            hard = [x for x in notes if x not in ("子区域(留白)",)]
            ok = not hard
            bad += 0 if ok else 1
            print("%-13s %5d %5d %-22s %-22s %s"
                  % (name, len(vs), len(tri) // 3,
                     "%.3f-%.3f / %.3f-%.3f" % (u0, u1, v0, v1),
                     ("%.3f-%.3f / %.3f-%.3f" % r_norm(regions.get(name), W, H))
                     if regions.get(name) else "-",
                     ("OK" if not notes else
                      ("OK " + "/".join(sorted(set(notes))) if ok
                       else "!! " + ", ".join(sorted(set(notes)))))))
    print("\n有问题: %d 个附件" % bad)
    return bad


def r_norm(r, W, H):
    x, y, w, h = r
    return (x / W, (x + w) / W, y / H, (y + h) / H)


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1]) else 0)
