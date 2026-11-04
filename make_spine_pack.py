"""把多个动作的 Spine json + 贴图目录 打成一个可以直接拖进 LoongBones 的包。

为什么要这个: LoongBones 的导入只支持三种 —— 它自己的 json / dbproj、Spine 3.8.75
(需要 png + atlas + json 三件一起拖)、PSD。**它不能导入 DragonBones 的 _ske.json**,
所以之前无论怎么改 _ske 都导不进去/显示错误。我们本来产出的就是 Spine 格式的动作,
直接按 Spine 3.8.75 打包含 atlas 即可。

产出:
    <out>/<name>.json    骨架 + 多条动画(合并)
    <out>/<name>.atlas   图集描述
    <out>/<name>.png     图集

用法:
    python make_spine_pack.py --spine a/result.json:anim --spine b/result.json:anim \
        --out output/xxx/spine --name hero
不写 :anim 就用 json 里自带的动画名。
"""
import argparse
import json
import os
import sys

import spine2dragonbones as S2D

SPINE_VER = "3.8.75"


def load(p):
    return json.load(open(p, encoding="utf-8"))


def merge(items, name):
    docs = []
    for p, img, anim in items:
        d = load(p)
        docs.append({"doc": d, "images": img, "anim": anim, "path": p})
    base = docs[0]["doc"]
    out = {
        "skeleton": dict(base.get("skeleton") or {}),
        "bones": base["bones"],
        "slots": base["slots"],
        "skins": base["skins"],
        "animations": {},
    }
    out["skeleton"]["spine"] = SPINE_VER
    out["skeleton"]["images"] = "./"
    for it in docs:
        for k, v in (it["doc"].get("animations") or {}).items():
            nm = it["anim"] or k
            # 名字冲突时加后缀
            if nm in out["animations"]:
                i = 2
                while f"{nm}{i}" in out["animations"]:
                    i += 1
                nm = f"{nm}{i}"
            out["animations"][nm] = v
    # 顶层 transform 是 Spine 4.x 的字段, 3.8 没有, 去掉更干净
    return out, docs, list(out["animations"])


def build_atlas(rects, W, H, lines, png_name):
    buf = [png_name, f"size: {W},{H}", "format: RGBA8888",
           "filter: Linear,Linear", "repeat: none"]
    for rname, (x, y, w, h) in lines:
        buf += [rname, "  rotate: false", f"  xy: {x}, {y}",
                f"  size: {w}, {h}", f"  orig: {w}, {h}",
                "  offset: 0, 0", "  index: -1"]
    return "\n".join(buf) + "\n"


def build(items, out, name=None):
    """items: [(result.json, images_dir, anim_name_or_None)]; 返回产出信息。"""
    name = name or os.path.splitext(os.path.basename(items[0][0]))[0]
    out_dir = out
    os.makedirs(out_dir, exist_ok=True)

    doc, docs, anims = merge(items, name)

    # 收集要打包的贴图: 按 skin attachment 里写的 path/名字
    used = {}
    for sl in doc["slots"]:
        sname = sl["name"]
        att = None
        for skin in doc["skins"]:
            if sname in (skin.get("attachments") or {}):
                att = skin["attachments"][sname]
                break
        if not att:
            continue
        aname = sl.get("attachment") or next(iter(att))
        ad = att.get(aname)
        if not ad:
            continue
        png = ad.get("path") or (aname + ".png")
        if not png.lower().endswith(".png"):
            png += ".png"
        if png in used:
            continue
        for it in docs:
            fp = os.path.join(it["images"], png)
            if os.path.isfile(fp):
                from PIL import Image
                with Image.open(fp) as im:
                    used[png] = (im.size[0], im.size[1], fp)
                break
        else:
            print(f"  缺贴图: {png}")
    if not used:
        raise SystemExit("没有任何贴图可打包")

    png_name = name + ".png"
    W, H, rects = S2D.pack_atlas(used, os.path.join(out_dir, png_name))
    lines = [(os.path.splitext(n)[0], r) for n, r in rects.items()]
    open(os.path.join(out_dir, name + ".atlas"), "w", encoding="utf-8", newline="\n") \
        .write(build_atlas(rects, W, H, lines, png_name))

    # attachment 的 path 写成图集里的 region 名(不带扩展名), Spine 默认就用附件名
    for skin in doc["skins"]:
        for sname, att in (skin.get("attachments") or {}).items():
            for aname, ad in att.items():
                ad.pop("path", None)

    json.dump(doc, open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"Spine 包 -> {out_dir}")
    print(f"  {name}.json   骨架 + {len(anims)} 条动画 {anims}")
    print(f"  {name}.atlas  图集描述")
    print(f"  {name}.png    图集 {W}x{H}，{len(used)} 张部件")
    print("  导入: 把 png + atlas + json 三个文件一起拖进 LoongBones 的资源库面板")
    return {"dir": out_dir, "json": os.path.join(out_dir, name + ".json"),
            "atlas": os.path.join(out_dir, name + ".atlas"),
            "png": os.path.join(out_dir, png_name), "anims": anims}


def main():
    ap = argparse.ArgumentParser(description="Spine 动作 -> LoongBones 可导入的 Spine 3.8.75 包")
    ap.add_argument("--spine", action="append", required=True,
                    help="result.json[:动画名], 可重复")
    ap.add_argument("--images", action="append", default=[],
                    help="贴图目录, 与 --spine 一一对应(可省, 默认同级 images/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    a = ap.parse_args()

    items = []
    for i, spec in enumerate(a.spine):
        p, _, anim = spec.partition(":")
        img = a.images[i] if i < len(a.images) else os.path.join(
            os.path.dirname(os.path.abspath(p)), "images")
        items.append((p, img, anim or None))
    build(items, a.out, a.name or None)


if __name__ == "__main__":
    main()
