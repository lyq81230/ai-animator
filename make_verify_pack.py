# -*- coding: utf-8 -*-
"""生成"绝对不会和项目里旧资源撞名"的网格版验证包。

已知事实(来自 DragonBones 官方 ObjectDataParser.ts):
  * vertices 原样存储, 是 slot(父骨)局部坐标, 不做 pivot 偏移;
  * 网格**不使用 pivot**(只有 Image 显示才解析 pivot);
  * uvs 原样存储成 0~1, 渲染时乘纹理宽高 —— 规范是"整张图集"归一化,
    但 DragonBones **Pro 编辑器内部把每个 SubTexture 存成独立贴图**,
    所以它按"本格 0~1"采样(这是编辑器的实现细节)。

因此给 Pro 的包要: uv 用**本格局部 0~1**, 顶点用"居中 + display.transform"。

用法:
    python make_verify_pack.py output/sprite_pi_white_336
产出 <角色>/_verify/<name>_v3_*, 里面两个骨架:
    v3A = 局部 uv + 顶点居中 + transform
    v3B = 局部 uv + 平移烘进顶点 + 无 transform
"""
import json
import os
import shutil
import sys

import db_mesh_from_plain as M


def local_uv(SM, TM):
    """把 display 的 uv 从"图集全局"换成"本格局部 0~1"。"""
    sub = {s["name"]: s for s in TM["SubTexture"]}
    TW, TH = TM["width"], TM["height"]
    for a in SM["armature"]:
        for s in a["skin"][0]["slot"]:
            for dp in s["display"]:
                if "uvs" not in dp:
                    continue
                st = sub.get(dp.get("path") or dp.get("name"))
                if not st:
                    continue
                uv = list(dp["uvs"])
                val = []
                for i, c in enumerate(uv):
                    if i % 2 == 0:                      # u
                        val.append(round((c * TW - st["x"]) / st["width"], 6))
                    else:                               # v
                        val.append(round((c * TH - st["y"]) / st["height"], 6))
                dp["uvs"] = val


def build(D, tag_suffix="v3"):
    D = os.path.abspath(D)
    mesh_dir = os.path.join(D, "cocos_mesh")
    SM, TM, ske_f, tex_f = M.load(mesh_dir)
    base = os.path.basename(D)
    tag = base + "_" + tag_suffix

    out = os.path.join(D, "_verify")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    TM["name"] = tag
    TM["imagePath"] = tag + "_tex.png"
    json.dump(TM, open(os.path.join(out, tag + "_tex.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    png = [f for f in os.listdir(mesh_dir) if f.endswith("_tex.png")][0]
    shutil.copy2(os.path.join(mesh_dir, png), os.path.join(out, tag + "_tex.png"))

    arms = []

    # A: 顶点居中 + display.transform 定位
    A = json.loads(json.dumps(SM["armature"][0]))
    A["name"] = tag + "A"
    arms.append(A)

    # B: 把 transform 的平移烘进顶点, 去掉 transform
    B = json.loads(json.dumps(SM["armature"][0]))
    B["name"] = tag + "B"
    for s in B["skin"][0]["slot"]:
        for dp in s["display"]:
            tr = dp.get("transform") or {}
            tx, ty = float(tr.get("x") or 0.0), float(tr.get("y") or 0.0)
            v = dp["vertices"]
            dp["vertices"] = [v[i] + (tx if i % 2 == 0 else ty)
                              for i in range(len(v))]
            dp.pop("transform", None)
    arms.append(B)

    # 两个骨架都改成局部 uv
    tmp = {"armature": arms}
    local_uv(tmp, TM)

    SM2 = dict(SM)                    # 保留 isGlobal 等顶层字段, 只换名字
    SM2["name"] = tag
    SM2["armature"] = arms
    json.dump(SM2, open(os.path.join(out, tag + "_ske.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"验证包 -> {out}  (armature: {tag}A / {tag}B, uv=本格局部)")

    # 自检
    d = next(s["display"][0] for s in arms[0]["skin"][0]["slot"]
             if s["name"] == "headwear")
    print("  headwear uv:", [round(x, 4) for x in d["uvs"]])
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "output/sprite_pi_white_336")
