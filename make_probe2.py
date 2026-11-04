# -*- coding: utf-8 -*-
"""探针包 B：把部件拆成"一个部件一张贴图"，彻底排除图集采样这一层。

现在的症状(头巾缺左半、且换成局部/全局 uv 都一样)说明问题不在 uv 数值上,
而是 Pro 对"图集里的子区域"和网格 uv 的组合理解与我们不同。

如果每个部件都是自己的一张 png、自己的 tex.json(SubTexture 就是整张图),
那 uv 无论按整张图还是按本格解释, 结果都相同 —— 采样这一层就无懈可击。
此时若还错, 就只剩顶点几何这一层了。

用法:
    python make_probe2.py output/sprite_pi_white_336
"""
import json
import os
import shutil
import sys

from PIL import Image

KEEP = ("topwear", "headwear")


def build(D):
    D = os.path.abspath(D)
    src = os.path.join(D, "_final2")
    ske_f = [f for f in os.listdir(src) if f.endswith("_ske.json")][0]
    tex_f = [f for f in os.listdir(src) if f.endswith("_tex.json")][0]
    S = json.load(open(os.path.join(src, ske_f), encoding="utf-8"))
    T = json.load(open(os.path.join(src, tex_f), encoding="utf-8"))
    atlas = Image.open(os.path.join(src, [f for f in os.listdir(src)
                                         if f.endswith("_tex.png")][0])).convert("RGBA")
    sub = {s["name"]: s for s in T["SubTexture"]}

    head = os.path.basename(D) + "_p2"
    out = os.path.join(D, "_probe2")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    arm = S["armature"][0]
    arm["name"] = head
    arm["slot"] = [s for s in arm["slot"] if s["name"] in KEEP]
    arm["skin"] = [{"name": "default",
                    "slot": [s for s in arm["skin"][0]["slot"] if s["name"] in KEEP]}]

    # 每个部件: 独立 png + 独立 tex.json, SubTexture 就是整张图
    for name in KEEP:
        st = sub[name]
        w, h = st["width"], st["height"]
        crop = atlas.crop((st["x"], st["y"], st["x"] + w, st["y"] + h))
        png = f"{head}_{name}_tex.png"
        crop.save(os.path.join(out, png))
        tex = {"name": f"{head}_{name}", "imagePath": png,
               "width": w, "height": h,
               "SubTexture": [{"name": name, "x": 0, "y": 0,
                               "width": w, "height": h,
                               "frameX": 0, "frameY": 0,
                               "frameWidth": w, "frameHeight": h}]}
        json.dump(tex, open(os.path.join(out, f"{head}_{name}_tex.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
        # display 里的 uv 保持 0~1 / 顶点保持 0..w
        for s in arm["skin"][0]["slot"]:
            if s["name"] == name:
                s["display"][0]["path"] = name
                s["display"][0]["name"] = name

    S2 = dict(S)
    S2["name"] = head
    S2["armature"] = [arm]
    json.dump(S2, open(os.path.join(out, head + "_ske.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"探针B -> {out}")
    for s in arm["skin"][0]["slot"]:
        d = s["display"][0]
        print(f"  {s['name']}: 顶点{d['vertices']} uv{d['uvs']} transform={d.get('transform')}")
    print("  文件:", sorted(os.listdir(out)))
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "output/sprite_pi_white_336")
