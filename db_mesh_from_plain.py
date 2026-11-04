# -*- coding: utf-8 -*-
"""把网格版改成"和图片版完全同一套定位"：四边形铺满整张贴图格，位置抄图片版的 transform。

依据: 用户实测 C 方案（顶点居中 + transform 定位）基本正确，但头巾这类部件还有偏差 ——
因为 C 的四边形是按"网格内容框"算的，比贴图格小 1~2px，贴图被塞进小框里就轻微压扁。
这里直接:
    1. 四边形 = 整张贴图格（宽高取 SubTexture 的 width/height，uv 取整格）
    2. transform = 图片版同一个槽的 transform（x/y/pivot 0.5）
这样网格和图片两种显示的定位在数学上完全等价，只是换了绘制方式。

用法:
    python db_mesh_from_plain.py output/sprite_pi_white_336           # 改 cocos_mesh/
    python db_mesh_from_plain.py output/sprite_pi_white_336 --test     # 另存 _mesh_fix/
"""
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image


def load(dirp):
    ske = [f for f in os.listdir(dirp) if f.endswith("_ske.json")][0]
    tex = [f for f in os.listdir(dirp) if f.endswith("_tex.json")][0]
    return (json.load(open(os.path.join(dirp, ske), encoding="utf-8")),
            json.load(open(os.path.join(dirp, tex), encoding="utf-8")), ske, tex)


def align(D, verbose=True):
    """就地改写 D/cocos_mesh，让它和 D/cocos 的定位完全等价。"""
    plain_dir = os.path.join(D, "cocos")
    mesh_dir = os.path.join(D, "cocos_mesh")
    SP, TP, _, _ = load(plain_dir)
    SM, TM, ske_f, tex_f = load(mesh_dir)

    sub = {s["name"]: s for s in TM["SubTexture"]}
    TW, TH = TM["width"], TM["height"]
    pslot = {s["name"]: s for s in SP["armature"][0]["skin"][0]["slot"]}
    mslot = {s["name"]: s for s in SM["armature"][0]["skin"][0]["slot"]}

    fixed = miss = 0
    for name, sl in mslot.items():
        dp = sl["display"][0]
        if "vertices" not in dp:
            continue
        st = sub.get(dp.get("path") or dp.get("name"))
        if st is None:
            continue
        w, h = float(st["width"]), float(st["height"])
        pd = pslot.get(name, {}).get("display", [{}])[0]
        tr = pd.get("transform") or {}
        tx = float(tr.get("x") or 0.0)
        ty = float(tr.get("y") or 0.0)

        # ---- DragonBones Pro 自己的写法(从它导出的文件反推, 21/21 全吻合) ----
        # 顶点 = "部件中心(相对父骨) ± 50", 即 100 单位归一化空间;
        # 真实像素尺寸由 width/height 提供(像素 = 顶点 * width / 100)。
        # uv  = 整张图集的归一化坐标。
        # 顶点顺序 0=左上 1=右上 2=左下 3=右下, 三角形 [1,0,3, 0,2,3]。
        dp["width"] = w
        dp["height"] = h
        dp["vertices"] = [round(tx - 50.0, 3), round(ty - 50.0, 3),
                          round(tx + 50.0, 3), round(ty - 50.0, 3),
                          round(tx - 50.0, 3), round(ty + 50.0, 3),
                          round(tx + 50.0, 3), round(ty + 50.0, 3)]
        u0, v0 = st["x"] / TW, st["y"] / TH
        u1, v1 = (st["x"] + w) / TW, (st["y"] + h) / TH
        dp["uvs"] = [u0, v0, u1, v0, u0, v1, u1, v1]
        dp["triangles"] = [1, 0, 3, 0, 2, 3]
        dp["edges"] = [0, 1, 1, 3, 3, 2, 2, 0]
        dp["userEdges"] = []
        dp.pop("transform", None)      # 网格没有 transform, 位置已经烘进顶点
        dp.pop("path", None)           # Pro 导出的网格只有 name, 没有 path
        dp["type"] = "mesh"
        fixed += 1

    if False:
        out = os.path.join(D, "_mesh_fix")
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        base = os.path.basename(os.path.abspath(D))
        SM["armature"][0]["name"] = base + "_mesh_fix"
        TM["name"] = base + "_mesh_fix"
        TM["imagePath"] = base + "_mesh_fix_tex.png"
        json.dump(SM, open(os.path.join(out, base + "_mesh_fix_ske.json"), "w",
                           encoding="utf-8"), ensure_ascii=False,
                  separators=(",", ":"))
        json.dump(TM, open(os.path.join(out, base + "_mesh_fix_tex.json"), "w",
                           encoding="utf-8"), ensure_ascii=False)
        png = [f for f in os.listdir(mesh_dir) if f.endswith("_tex.png")][0]
        shutil.copy2(os.path.join(mesh_dir, png),
                     os.path.join(out, base + "_mesh_fix_tex.png"))
        print(f"测试包 -> {out}")
    else:
        json.dump(SM, open(os.path.join(mesh_dir, ske_f), "w", encoding="utf-8"),
                  ensure_ascii=False, separators=(",", ":"))
    if verbose:
        print(f"  网格对齐图片版: 四边形铺满贴图格 {fixed} 个槽"
              f"{', 无定位可抄 %d 个' % miss if miss else ''}")
    return fixed


def main():
    D = sys.argv[1]
    test = "--test" in sys.argv
    if test:
        plain_dir = os.path.join(D, "cocos")
        mesh_dir = os.path.join(D, "cocos_mesh")
        SP, TP, _, _ = load(plain_dir)
        SM, TM, ske_f, tex_f = load(mesh_dir)
        align(D, verbose=True)
        out = os.path.join(D, "_mesh_fix")
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        base = os.path.basename(os.path.abspath(D))
        SM = json.load(open(os.path.join(mesh_dir, ske_f), encoding="utf-8"))
        TM = json.load(open(os.path.join(mesh_dir, tex_f), encoding="utf-8"))
        SM["armature"][0]["name"] = base + "_mesh_fix"
        TM["name"] = base + "_mesh_fix"
        TM["imagePath"] = base + "_mesh_fix_tex.png"
        json.dump(SM, open(os.path.join(out, base + "_mesh_fix_ske.json"), "w",
                           encoding="utf-8"), ensure_ascii=False,
                  separators=(",", ":"))
        json.dump(TM, open(os.path.join(out, base + "_mesh_fix_tex.json"), "w",
                           encoding="utf-8"), ensure_ascii=False)
        png = [f for f in os.listdir(mesh_dir) if f.endswith("_tex.png")][0]
        shutil.copy2(os.path.join(mesh_dir, png),
                     os.path.join(out, base + "_mesh_fix_tex.png"))
        print(f"测试包 -> {out}")
    else:
        align(D)


if __name__ == "__main__":
    main()
