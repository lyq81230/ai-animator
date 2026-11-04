# -*- coding: utf-8 -*-
"""把网格版导成"三种坐标写法各一份"的诊断包, 一次导入就能看出哪种对。

背景: 官方样例(shizuku.json)里网格 display 是
    {"type":"mesh", "vertices":[...], "uvs":[...], "triangles":[...]}
没有 transform、没有 width/height、没有 weights。我们照抄了, 但 DragonBones Pro
里部件位置还是散, 说明差的是"顶点写在哪个坐标系"。三种可能:

    A 骨局部   顶点 = 内容框 - 父骨世界位置          (当前导出)
    B 绝对     顶点 = 内容框(画布坐标, y 向下)
    C 居中     顶点 = ±宽高/2, 位置交给 transform(x,y,pivot 0.5)
                —— 和能正常显示的图片版完全同一套定位方式

产出: <角色目录>/_mesh_test/ 里一份 ske, 里面放 3 个 armature
      (名字后缀 _A / _B / _C), 共用同一张图集。导入一次就能并排看三个。

用法: python make_mesh_variants.py output/sprite_pi_white_336
"""
import json
import math
import os
import shutil
import sys

import numpy as np
from PIL import Image


def bone_worlds(bones):
    by = {b["name"]: b for b in bones}
    out = {}

    def rec(n):
        if n in out:
            return out[n]
        b = by[n]
        tr = b.get("transform") or {}
        x, y = float(tr.get("x", 0.0)), float(tr.get("y", 0.0))
        r = float(tr.get("skY", tr.get("rotate", 0.0)))
        sx = float(tr.get("scX", 1.0))
        sy = float(tr.get("scY", 1.0))
        p = b.get("parent")
        if p:
            px, py, pr, psx, psy = rec(p)
            rad = math.radians(pr)
            lx, ly = x * psx, y * psy
            out[n] = (px + lx * math.cos(rad) - ly * math.sin(rad),
                      py + lx * math.sin(rad) + ly * math.cos(rad),
                      pr + r, psx * sx, psy * sy)
        else:
            out[n] = (x, y, r, sx, sy)
        return out[n]

    for b in bones:
        rec(b["name"])
    return out


def main():
    D = sys.argv[1]
    src_dir = os.path.join(D, "cocos_mesh")
    out_dir = os.path.join(D, "_mesh_test")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    ske_f = [f for f in os.listdir(src_dir) if f.endswith("_ske.json")][0]
    tex_f = [f for f in os.listdir(src_dir) if f.endswith("_tex.json")][0]
    png_f = [f for f in os.listdir(src_dir) if f.endswith("_tex.png")][0]
    S = json.load(open(os.path.join(src_dir, ske_f), encoding="utf-8"))
    T = json.load(open(os.path.join(src_dir, tex_f), encoding="utf-8"))

    name = os.path.basename(os.path.abspath(D))
    base = S["armature"][0]
    bw = bone_worlds(base["bone"])
    slots = {s["name"]: s for s in base["slot"]}

    variants = []
    for tag in ("A", "B", "C"):
        arm = json.loads(json.dumps(base))          # 深拷贝
        arm["name"] = f"{name}_mesh_{tag}"
        for sk in arm["skin"]:
            for sl in sk["slot"]:
                dp = sl["display"][0]
                vs = dp.get("vertices")
                if not vs:
                    continue
                par = slots[sl["name"]].get("parent")
                bx, by = (bw[par][0], bw[par][1]) if par in bw else (0.0, 0.0)
                xs = vs[0::2]
                ys = vs[1::2]
                cx = (min(xs) + max(xs)) / 2.0
                cy = (min(ys) + max(ys)) / 2.0
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                if tag == "A":
                    pass                            # 已经是骨局部
                elif tag == "B":
                    new = []
                    for i in range(0, len(vs), 2):
                        new += [round(vs[i] + bx, 3), round(vs[i + 1] + by, 3)]
                    dp["vertices"] = new
                else:                               # C: 居中 + transform
                    dp["vertices"] = [round(-w / 2, 3), round(-h / 2, 3),
                                      round(w / 2, 3), round(-h / 2, 3),
                                      round(w / 2, 3), round(h / 2, 3),
                                      round(-w / 2, 3), round(h / 2, 3)]
                    dp["transform"] = {"x": round(cx, 3), "y": round(cy, 3),
                                       "pivot": {"x": 0.5, "y": 0.5}}
        variants.append(arm)

    out = {"name": f"{name}_mesh_test", "version": S.get("version", "5.5"),
           "frameRate": S.get("frameRate", 24), "isGlobal": S.get("isGlobal", False),
           "armature": variants}
    json.dump(out, open(os.path.join(out_dir, f"{name}_mesh_test_ske.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    T2 = dict(T)
    T2["name"] = f"{name}_mesh_test"
    T2["imagePath"] = f"{name}_mesh_test_tex.png"
    json.dump(T2, open(os.path.join(out_dir, f"{name}_mesh_test_tex.json"), "w",
                       encoding="utf-8"), ensure_ascii=False)
    shutil.copy2(os.path.join(src_dir, png_f),
                 os.path.join(out_dir, f"{name}_mesh_test_tex.png"))
    print(f"诊断包 -> {out_dir}")
    print("  3 个 armature: " + ", ".join(a["name"] for a in variants))


if __name__ == "__main__":
    main()
