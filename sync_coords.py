# -*- coding: utf-8 -*-
"""把 coords.json 和磁盘上的 PNG 对齐一次。

为什么需要
----------
拆图流水线里好几个后处理脚本(repair_scarf_v2 / merge_fan_smoke / trim_parts ...)
会重裁部件、改 coords; 只要中间有一次没同步, 下一个脚本就会拿着**旧尺寸**去切
**新 PNG**, 直接抛

    ValueError: could not broadcast input array from shape (202,220) into shape (202,217)

整个"拆图"就在那一步中断 —— 界面上看起来就是"app 拆图失败"。这个脚本按 PNG 的
真实尺寸回填 width/height, 并把超出画布的部分用**扩画布**的方式救回来(宁可画布
大几像素, 也不要裁掉角色身体)。

用法:
    python sync_coords.py output/<角色> [--quiet]
"""
import json
import os
import sys

from PIL import Image


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    quiet = "--quiet" in sys.argv
    if not args:
        print("用法: python sync_coords.py output/<角色> [--quiet]")
        return 2
    D = args[0]
    cp = os.path.join(D, "coords.json")
    pc = os.path.join(D, "parts", "coords.json")
    # 谁是"最新"就用谁: 后处理脚本一般只写 parts/coords.json, 而 <work>/coords.json
    # 是界面/其它脚本读的那份。不同步就会拿旧坐标去切新 PNG。
    use = cp
    if os.path.isfile(pc) and (not os.path.isfile(cp)
                               or os.path.getmtime(pc) > os.path.getmtime(cp) + 0.5):
        use = pc
    if not os.path.isfile(use):
        print(f"没有 coords.json: {cp}")
        return 1
    co = json.load(open(use, encoding="utf-8"))
    cw, ch = int(co["canvas"]["w"]), int(co["canvas"]["h"])
    fixed, dropped = [], []
    for p in list(co["parts"]):
        f = os.path.join(D, "parts", p["image"])
        if not os.path.isfile(f):
            dropped.append(p["name"])
            co["parts"].remove(p)
            continue
        w, h = Image.open(f).size
        if (w, h) != (p.get("width"), p.get("height")):
            fixed.append(f"{p['name']} {p.get('width')}x{p.get('height')} -> {w}x{h}")
            p["width"], p["height"] = w, h
        if p["x"] < 0 or p["y"] < 0:
            fixed.append(f"{p['name']} 原点 {p['x']},{p['y']} -> 0,0")
            p["x"], p["y"] = max(0, p["x"]), max(0, p["y"])
        cw = max(cw, p["x"] + p["width"])
        ch = max(ch, p["y"] + p["height"])
    if (cw, ch) != (co["canvas"]["w"], co["canvas"]["h"]):
        fixed.append(f"画布 {co['canvas']['w']}x{co['canvas']['h']} -> {cw}x{ch}")
        co["canvas"]["w"], co["canvas"]["h"] = cw, ch
    for f in (cp, os.path.join(D, "parts", "coords.json")):
        if os.path.isdir(os.path.dirname(f)):
            json.dump(co, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if not quiet:
        for x in fixed:
            print("  同步 " + x)
        for x in dropped:
            print("  删掉缺图的部件 " + x)
        print(f"坐标同步完成: {len(fixed)} 处修正, {len(dropped)} 个部件删除"
              if (fixed or dropped) else "坐标已经一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
