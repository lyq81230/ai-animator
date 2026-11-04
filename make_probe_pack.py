# -*- coding: utf-8 -*-
"""生成"极简探针包"：只留 2 个槽(headwear / topwear)，供在 DragonBones Pro 里
导入 -> 立刻导出数据，再拿 Pro 自己的 ske.json 来对比格式差异。

为什么要这么做：我们的图集、坐标、名字、uv 都已经和图片版逐字节一致，
剩下只有 mesh 的写法差异，而 Pro 的实际期望无法从文档推断 —— 让 Pro 自己写一遍
最快，它写出来的 vertices/transform 就是标准答案。

用法:
    python make_probe_pack.py output/sprite_pi_white_336
"""
import json
import os
import shutil
import sys

KEEP = ("headwear", "topwear")


def build(D):
    D = os.path.abspath(D)
    src = os.path.join(D, "_verify")
    ske_f = [f for f in os.listdir(src) if f.endswith("_v3_ske.json")][0]
    tex_f = [f for f in os.listdir(src) if f.endswith("_v3_tex.json")][0]
    S = json.load(open(os.path.join(src, ske_f), encoding="utf-8"))
    T = json.load(open(os.path.join(src, tex_f), encoding="utf-8"))

    base = os.path.basename(D)
    tag = base + "_probe"

    out = os.path.join(D, "_probe")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    arm = S["armature"][0]          # 只用 v3A
    arm["name"] = tag
    arm["slot"] = [s for s in arm["slot"] if s["name"] in KEEP]
    arm["skin"] = [{"name": arm["skin"][0].get("name", "default"),
                    "slot": [s for s in arm["skin"][0]["slot"]
                             if s["name"] in KEEP]}]
    midx = {s["name"]: i for i, s in enumerate(arm["slot"])}
    if "animation" in arm:
        for an in arm["animation"].values():
            an["slot"] = [s for s in (an.get("slot") or [])
                          if s.get("name") in KEEP]
    S["armature"] = [arm]
    S["name"] = tag

    T["name"] = tag
    T["imagePath"] = tag + "_tex.png"
    T["SubTexture"] = [s for s in T["SubTexture"] if s["name"] in KEEP]

    json.dump(S, open(os.path.join(out, tag + "_ske.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    json.dump(T, open(os.path.join(out, tag + "_tex.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 图集原样拷贝: 一改尺寸就多了一个变量, 探针要和 v3A 完全等价(只少几个图标)
    png = [f for f in os.listdir(src) if f.endswith("_v3_tex.png")][0]
    shutil.copy2(os.path.join(src, png), os.path.join(out, tag + "_tex.png"))
    x1, y1 = T["width"], T["height"]
    print(f"探针包 -> {out}")
    print(f"  骨架 {tag}，槽: {[s['name'] for s in arm['slot']]}")
    print(f"  图集 {x1}x{y1}，格子 {[(s['name'], s['x'], s['y'], s['width'], s['height']) for s in T['SubTexture']]}")
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "output/sprite_pi_white_336")
