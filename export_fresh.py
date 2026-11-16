# -*- coding: utf-8 -*-
"""把已导出的 spine 包复制成一份"改名"+带时间戳的包, 供编辑器首次干净导入。

为什么需要: 贴图页文件名固定(<角色>.png), 编辑器/引擎常常按文件名缓存上一版纹理。
atlas 说他 512x188, 缓存里还是 512x284 的旧图 —— 每个网格都从错的位置采样,
表现就是"导入后全乱"(颜色还是角色的, 位置全错)。

用法: python export_fresh.py output/sprite_suo_white_336 [后缀]
产出: <角色>_<后缀>.json/.atlas/.png  (atlas 第一行同步改名)
"""
import json
import os
import re
import shutil
import sys
import time

import numpy as np
from PIL import Image


def main():
    d = sys.argv[1].rstrip("/\\")
    name = os.path.basename(d)
    suf = sys.argv[2] if len(sys.argv) > 2 else time.strftime("%m%d_%H%M")
    src = os.path.join(d, "spine")
    j = os.path.join(src, name + ".json")
    a = os.path.join(src, name + ".atlas")
    p = os.path.join(src, name + ".png")
    for f in (j, a, p):
        if not os.path.isfile(f):
            print("缺文件", f)
            return
    new = f"{name}_{suf}"
    out = os.path.join(d, "spine", new)
    os.makedirs(out, exist_ok=True)

    J = json.load(open(j, encoding="utf-8"))
    if "--uv-swap" in sys.argv:
        # LoongBones 读 mesh 的 uvs 时把 u/v 取反了: 标准 Spine 的 uv 进去,
        # 每个部件都从图集错的地方横向采样, 表现是"轮廓对、贴图变成横条色带"。
        # 导出时把 (u,v) 换成 (v,u), 正好抵消它的取反。
        n = 0
        for sk in J.get("skins", []):
            for slot, atts in (sk.get("attachments") or {}).items():
                for an, att in atts.items():
                    if att.get("type") != "mesh":
                        continue
                    uv = att["uvs"]
                    sw = []
                    for i in range(0, len(uv), 2):
                        sw += [uv[i + 1], uv[i]]
                    att["uvs"] = sw
                    n += 1
        print(f"  uv 交换: {n} 个网格")
    json.dump(J, open(os.path.join(out, new + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))

    txt = open(a, encoding="utf-8").read().splitlines()
    txt[0] = new + ".png"
    open(os.path.join(out, new + ".atlas"), "w", encoding="utf-8").write(
        "\n".join(txt) + "\n")
    shutil.copyfile(p, os.path.join(out, new + ".png"))

    W, H = Image.open(os.path.join(out, new + ".png")).size
    m = re.search(r"size:\s*(\d+),(\d+)", "\n".join(txt[:6]))
    print(f"导出 {new}  png {W}x{H}  atlas size {m.group(1)}x{m.group(2)}"
          + ("  ✓ 一致" if (int(m.group(1)), int(m.group(2))) == (W, H)
             else "  !! 尺寸不一致"))
    print("目录", out)


if __name__ == "__main__":
    main()
