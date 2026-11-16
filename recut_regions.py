# -*- coding: utf-8 -*-
"""按"区域 + 目标部件"重新切图: 用于 AI 分层把部件归错了地方的角色。

用法: python recut_regions.py <角色目录> <规格.json>
规格格式:
{
  "objects_L": {"box": [28,190,108,300], "cls": "prop", "follow": "arm_hand_L"},
  "leg_L_up":  {"box": [106,198,154,258], "cls": "leg"}
}

规则:
  1. 目标部件的内容 = 源图在该矩形内、且不透明(alpha>40)的像素;
  2. 这些像素会从**其它所有部件**里删掉(避免残影);
  3. 按规格顺序处理, 后处理的会从先处理的区域里扣掉(先剑后腿, 免得剑被腿吞);
  4. 颜色一律取源图; 处理完建议跑 fill_holes 收尾。
"""
import json
import os
import sys

import numpy as np
from PIL import Image


def main():
    D = sys.argv[1]
    spec = json.load(open(sys.argv[2], encoding="utf-8"))
    cp = os.path.join(D, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    W, H = co["canvas"]["w"], co["canvas"]["h"]
    char = os.path.basename(os.path.normpath(D))
    src = np.array(Image.open(os.path.join("E:/pic", char + ".png")).convert("RGBA"))
    if src.shape[1] != W or src.shape[0] != H:
        src = np.array(Image.fromarray(src).resize((W, H), Image.LANCZOS))
    by = {p["name"]: p for p in co["parts"]}

    def canvas_of(p):
        a = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))
        m = np.zeros((H, W, 4), np.uint8)
        m[p["y"]:p["y"] + a.shape[0], p["x"]:p["x"] + a.shape[1]] = a
        return m

    canv = {p["name"]: canvas_of(p) for p in co["parts"]}
    claimed = np.zeros((H, W), bool)

    drop = []
    for name, cfg in spec.items():
        if cfg.get("clear"):
            if name in by:
                drop.append(name)
                print(f"  {name:<12} 删除部件(内容归到别处)")
            continue
        # 支持 "boxes": 多个矩形取并集 —— 头这种"上宽下窄"的形状, 单个矩形要么
        # 把胸甲框进来, 要么把下巴切掉; 主体用大框 + 下巴用窄框就正好。
        reg = np.zeros((H, W), bool)
        for bx in (cfg.get("boxes") or [cfg["box"]]):
            x0, y0, x1, y1 = bx
            reg[y0:y1 + 1, x0:x1 + 1] = True
        for ex in cfg.get("exclude", []):
            ex0, ey0, ex1, ey1 = ex
            reg[ey0:ey1 + 1, ex0:ex1 + 1] = False
        take = reg & (src[..., 3] > 40) & ~claimed
        if cfg.get("subtract_others"):
            # 背后画的部件(翅膀/披风)老是把肩膀、手臂也吞进来。只保留"别人没画到"
            # 的那部分像素, 既不影响绑定姿态的画面, 也不会让翅膀骨拖着肩膀走。
            others = np.zeros((H, W), bool)
            for on, oc in canv.items():
                if on != name:
                    others |= oc[..., 3] > 40
            take &= ~others
        claimed |= take
        if name not in by:
            # 规格里出现新部件名: 现场建一个(img 空白, 稍后由 take 填内容)
            tmpl = dict(co["parts"][0])
            tmpl.update({"name": name, "image": name + ".png", "x": 0, "y": 0,
                         "width": 1, "height": 1})
            tmpl.pop("z", None)
            if cfg.get("follow"):
                tmpl["follow_part"] = cfg["follow"]
            tmpl["cls"] = cfg.get("cls") or "prop"
            idx = next((i for i, q in enumerate(co["parts"])
                        if q["name"].startswith("arm_upper_R")), len(co["parts"]))
            co["parts"].insert(idx, tmpl)
            by[name] = tmpl
            canv[name] = np.zeros((H, W, 4), np.uint8)
            print("  新建部件", name)
        # 从其它部件里扣掉(注意: 头这类"压在别人上面"的部件不该扣 —— 头转动时
        # 下面要露出领口/胸甲, 扣掉就是挖一个洞)
        for other, c in ([] if cfg.get("no_steal") else canv.items()):
            if other == name:
                continue
            if c[..., 3][take].any():
                c[..., 3][take] = 0
        # 目标部件先清空: 否则它原来错归的内容会留在包围盒里(实测 leg_L_up
        # 保留了左剑那块的旧像素, 包围盒从 50x62 涨到 75x78)
        canv[name][...] = 0
        canv[name][take] = src[take]          # 目标部件拿到这块像素(颜色取源图)
        if cfg.get("keep_main"):
            # 只留"主体"岛: 矩形切法难免把旁边别人的碎块一起框进来
            # (刀旁边一截手、腿旁边一块甲)。留岛面积 >= keep_main 的。
            from scipy import ndimage as _ndi
            m = canv[name][..., 3] > 40
            lab, k = _ndi.label(m, structure=np.ones((3, 3)))
            if k > 1:
                sizes = _ndi.sum(m, lab, range(1, k + 1))
                floor = float(cfg["keep_main"])
                keep = np.zeros_like(m)
                for i, sz in enumerate(sizes, 1):
                    if sz >= floor:
                        keep |= lab == i
                canv[name][..., 3][m & ~keep] = 0
                print(f"      保留岛 >= {floor:.0f}px: 去掉 {int((m & ~keep).sum())}px")
        if cfg.get("cls"):
            by[name]["cls"] = cfg["cls"]
        if cfg.get("follow"):
            by[name]["follow_part"] = cfg["follow"]
        _b = cfg.get("box") or cfg.get("boxes")
        print(f"  {name:<12} 取 {int(take.sum()):5d}px  box={_b}")

    for name in drop:
        p = by[name]
        f = os.path.join(D, "parts", p["image"])
        if os.path.isfile(f):
            os.remove(f)
        co["parts"] = [q for q in co["parts"] if q["name"] != name]
        canv.pop(name, None)
        by.pop(name, None)

    # 写回
    for name, c in canv.items():
        p = by[name]
        ys, xs = np.nonzero(c[..., 3] > 8)
        if len(ys) == 0:
            print(f"  !! {name} 变成空部件")
            c[..., 3] = 0
            p.update({"x": 0, "y": 0, "width": 1, "height": 1})
            Image.fromarray(np.zeros((1, 1, 4), np.uint8)).save(
                os.path.join(D, "parts", p["image"]))
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        Image.fromarray(c[y0:y1 + 1, x0:x1 + 1]).save(
            os.path.join(D, "parts", p["image"]))
        p.update({"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1})

    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pc = os.path.join(D, "parts", "coords.json")
    if os.path.isfile(pc):
        json.dump(co, open(pc, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("完成")


if __name__ == "__main__":
    main()
