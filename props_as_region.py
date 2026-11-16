"""把武器类(prop)的网格附件改成普通图片附件(region)。

为什么:
    武器是刚性物件, 跟着一根骨走, 本来就不需要网格变形。但它被做成 4 顶点四边形网格,
    于是在导入端要经过"网格"这条路: 顶点/uv/hull/edges 任何一处被工具处理得不对,
    细尖(枪尖/刀尖)就会丢。实测我们的网格渲染与原图只差 1px 级的边缘, 但用户那边
    反复看不到枪尖。
    图片附件(region)没有顶点, 只有 x/y/宽/高, 任何工具都是直接贴图, 不可能少一块。

做法:
    对每个 mesh 附件: 要求 4 顶点、每顶点 1 根骨且权重 1 -> 取四角平均作为中心,
    写成 region 附件 {width, height, x, y, rotation: 0, scaleX: 1, scaleY: 1}。
    其余(多顶点/多骨)保持网格不变。

用法:
    python props_as_region.py <角色目录> [--all]
        默认只处理 coords.json 里 cls == prop 的部件; --all 处理所有 4 顶点网格
"""
import json
import os
import sys


def quad_center(att):
    """4 顶点、各 1 骨、权重 1 的网格 -> (cx, cy) 或 None。"""
    V = att.get("vertices") or []
    i = 0
    pts = []
    while i < len(V):
        c = int(V[i])
        if c != 1:
            return None
        g = i + 1
        if abs(float(V[g + 3]) - 1.0) > 1e-3:
            return None
        pts.append((float(V[g + 1]), float(V[g + 2])))
        i += 5
    if len(pts) != 4:
        return None
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    return cx, cy


def convert(D, only_prop=True, quiet=False):
    D = D.rstrip("/\\")
    name = os.path.basename(D)
    sp = os.path.join(D, "spine")
    jf = os.path.join(sp, name + ".json")
    J = json.load(open(jf, encoding="utf-8"))
    props = set()
    cp = os.path.join(D, "coords.json")
    if only_prop and os.path.isfile(cp):
        co = json.load(open(cp, encoding="utf-8"))
        props = {p["name"] for p in co["parts"] if str(p.get("cls") or "") == "prop"}
    n = 0
    for skin in J["skins"]:
        for slot, atts in (skin.get("attachments") or {}).items():
            for aname, a in list(atts.items()):
                if a.get("type") != "mesh":
                    continue
                if only_prop and aname not in props:
                    continue
                c = quad_center(a)
                if c is None:
                    if not quiet:
                        print(f"  跳过 {aname}: 不是 4 顶点单骨网格")
                    continue
                w = a.get("width")
                h = a.get("height")
                atts[aname] = {
                    "width": w, "height": h,
                    "x": round(c[0], 3), "y": round(c[1], 3),
                    "rotation": 0, "scaleX": 1, "scaleY": 1,
                }
                n += 1
                if not quiet:
                    print(f"  {aname:<12} 网格 -> 图片附件  {w}x{h} @({c[0]:.1f},{c[1]:.1f})")
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, separators=(",", ":"))
    if not quiet:
        print(f"转换 {n} 个武器附件 -> {jf}")
    return n




def main():
    D = sys.argv[1].rstrip("/\\")
    convert(D, only_prop="--all" not in sys.argv)


if __name__ == "__main__":
    main()
