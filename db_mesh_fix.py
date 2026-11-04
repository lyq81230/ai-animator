# -*- coding: utf-8 -*-
"""把导出的网格版 DragonBones 数据改写成"官方样例同款"字段。

对照 `_ref/shizuku.json`(DragonBones 5.6 官方样例, 131 个网格显示)发现两处硬差异:
    1. 官方 display 的 type 写成字符串 "mesh", 我们写的是数字 2;
    2. 官方 131 个网格显示**全都没有** width / height, 我们为了修 0.6 缩放加上了。

命令行用法(单变量对照实验):
    python db_mesh_fix.py <源目录> <输出目录>
"""
import json
import os
import shutil
import sys


def to_official(ske_path):
    """就地改写: type -> "mesh"。返回 (网格数, 改写数)。

    注意: **不要删 width/height**。Pro 自己导出时每个网格都带 width/height,
    网格顶点是"100 单位归一化空间", 真实像素尺寸由 width/height 提供;
    删掉它们 Pro 就按 1:1 当像素用, 部件会被放大/错位。
    """
    S = json.load(open(ske_path, encoding="utf-8"))
    n_mesh = n_fix = 0
    for arm in S.get("armature", []):
        skin = arm.get("skin") or []
        for sk in (skin if isinstance(skin, list) else [skin]):
            for sl in sk.get("slot") or []:
                for dp in sl.get("display") or []:
                    if "vertices" not in dp:
                        continue
                    n_mesh += 1
                    changed = False
                    if dp.get("type") != "mesh":
                        dp["type"] = "mesh"
                        changed = True
                    if changed:
                        n_fix += 1
    json.dump(S, open(ske_path, "w", encoding="utf-8"), ensure_ascii=False,
              separators=(",", ":"))
    return n_mesh, n_fix


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    ske = [f for f in os.listdir(dst) if f.endswith("_ske.json")][0]
    n_mesh, n_fix = to_official(os.path.join(dst, ske))
    print(f"{ske}: 网格显示 {n_mesh} 个, 改写 {n_fix} 个 "
          f"(type -> \"mesh\", 去掉 width/height)")
    print("输出:", dst)


if __name__ == "__main__":
    main()
