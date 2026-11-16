# -*- coding: utf-8 -*-
"""体检: 每个网格的 uv 是否落在它在 atlas 里那一格内, atlas 的 size 是否等于 png 实际尺寸。

这是"导入编辑器后贴图全乱"的第一嫌疑人: uv 跑出格子(采样到别人的图), 或者
atlas 写的尺寸和 png 对不上(整页错位)。
"""
import json
import os
import re
import sys

from PIL import Image


def parse_atlas(path):
    """文件头是 文件名 / size / format / filter / repeat, 之后每个区域: 名字 + 缩进字段。
    页尺寸必须取【文件头里的第一条 size】, 不是第一个区域的 size(踩过)。"""
    lines = open(path, encoding="utf-8").read().splitlines()
    W = H = None
    regions = {}
    cur = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        indented = bool(ln[:1].isspace())
        if not indented and i > 0 and not s.split(":")[0] in (
                "size", "format", "filter", "repeat"):
            cur = s
            regions[cur] = {}
            continue
        if s.startswith("size:"):
            nums = re.findall(r"\d+", s)
            if len(nums) >= 2:
                if cur is None:
                    W, H = int(nums[0]), int(nums[1])   # 页尺寸
                else:
                    regions[cur]["size"] = (int(nums[0]), int(nums[1]))
        elif s.startswith("xy:") and cur:
            nums = re.findall(r"\d+", s)
            if len(nums) >= 2:
                regions[cur]["xy"] = (int(nums[0]), int(nums[1]))
    return W, H, regions


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "output"
    bad_total = 0
    for name in sorted(os.listdir(root)):
        sp = os.path.join(root, name, "spine")
        j = os.path.join(sp, name + ".json")
        a = os.path.join(sp, name + ".atlas")
        p = os.path.join(sp, name + ".png")
        if not all(map(os.path.isfile, (j, a, p))):
            continue
        W, H, reg = parse_atlas(a)
        pw, ph = Image.open(p).size
        J = json.load(open(j, encoding="utf-8"))
        bad = 0
        tot = 0
        for sk in J.get("skins", []):
            for slot, atts in (sk.get("attachments") or {}).items():
                for att_name, att in atts.items():
                    if att.get("type") != "mesh":
                        continue
                    r = reg.get(att_name)
                    if not r or "xy" not in r or "size" not in r:
                        continue
                    x, y = r["xy"]
                    w, h = r["size"]
                    u0, u1 = x / W, (x + w) / W
                    v0, v1 = y / H, (y + h) / H
                    uv = att["uvs"]
                    us, vs = uv[0::2], uv[1::2]
                    tot += 1
                    if (min(us) < u0 - 0.012 or max(us) > u1 + 0.012
                            or min(vs) < v0 - 0.012 or max(vs) > v1 + 0.012):
                        bad += 1
                        print(f"   越界 {name}/{att_name} "
                              f"u[{min(us):.3f},{max(us):.3f}] 应[{u0:.3f},{u1:.3f}] "
                              f"v[{min(vs):.3f},{max(vs):.3f}] 应[{v0:.3f},{v1:.3f}]")
        flag = "OK" if (bad == 0 and (W, H) == (pw, ph)) else "!!"
        print(f"{flag} {name:<30} 网格{tot:3d} 越界{bad}  "
              f"atlas {W}x{H} png {pw}x{ph}")
        bad_total += bad + (0 if (W, H) == (pw, ph) else 1)
    print("总计问题", bad_total)


if __name__ == "__main__":
    main()
