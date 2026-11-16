# -*- coding: utf-8 -*-
"""调整绘制顺序(Spine 的 slot 顺序 = 绘制顺序, 后面的画在上面)。

用法:
    # 让某几个部件画在 topwear 之前(被躯干遮住)
    python set_draw_order.py output/sprite_pi_white_336 --behind arm_hand_R
    # 恢复默认(去掉 z 覆盖)
    python set_draw_order.py output/sprite_pi_white_336 --reset arm_hand_R

只改 coords.json 里的 "z" 字段: 小的先画, 同 z 保持拆图顺序。改完重新导出即可。
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("char")
    ap.add_argument("--behind", nargs="+", default=[],
                    help="这些部件画在 --than 之前(默认 topwear), 即被它遮住")
    ap.add_argument("--than", default="topwear", help="参照部件(默认 topwear)")
    ap.add_argument("--front", nargs="+", default=[],
                    help="这些部件画在最后(最上层)")
    ap.add_argument("--reset", nargs="+", default=[], help="去掉这些部件的 z 覆盖")
    ap.add_argument("--z", nargs="+", default=[],
                    help="直接指定 z, 写法 部件名=数值 (如 footwear_L=0.5)")
    a = ap.parse_args()

    cp = os.path.join(a.char, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    by = {p["name"]: p for p in co["parts"]}

    ref = by.get(a.than)
    ref_z = ref.get("z", 0) if ref else 0

    if a.reset:
        for n in a.reset:
            by[n].pop("z", None) if n in by else None
        print("已清除 z:", ", ".join(a.reset))

    if a.behind:
        for n in a.behind:
            if n not in by:
                print("警告: 没有部件", n)
                continue
            by[n]["z"] = ref_z - 1
        print(f"放到 {a.than} 之下(先画): " + ", ".join(a.behind))

    if a.z:
        for item in a.z:
            n, _, v = item.partition("=")
            n, v = n.strip(), v.strip()
            if n not in by:
                print("警告: 没有部件", n)
                continue
            try:
                by[n]["z"] = float(v)
            except ValueError:
                print("警告: z 值不合法", item)
                continue
        print("直接指定 z: " + ", ".join(a.z))

    if a.front:
        zmax = max((p.get("z", 0) for p in co["parts"]), default=0)
        for n in a.front:
            if n in by:
                by[n]["z"] = zmax + 1
        print("放到最上层: " + ", ".join(a.front))

    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # 拆图目录里也留一份, 重新拆图/生成时以 coords.json 为准
    pc = os.path.join(a.char, "parts", "coords.json")
    if os.path.isfile(pc):
        json.dump(co, open(pc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("已写入", cp)


if __name__ == "__main__":
    main()
