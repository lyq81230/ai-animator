# -*- coding: utf-8 -*-
"""合成"闭眼"贴图：AI 分层给的 eyes_closed 其实是睁眼重绘（眼白+红瞳），
直接当补丁用等于没换图。这里用「头部皮肤 + 压扁的睫毛」现造一张闭眼图。

做法（对每个角色）：
  1. 取 eyes_closed 的包围盒
  2. 底色 = head 在同一区域的像素；透明处用最近的不透明像素补
  3. 眼缝 = eyelash（没有就用 eyewhite）的深色部分，垂直压到约 18%，放在眼睛中线
  4. 输出同尺寸的 eyes_closed.png，位置不变（coords 不用改）

用法:
    python make_closed_eyes.py output/fa            # 覆盖 parts/eyes_closed.png
    python make_closed_eyes.py output/fa --dry      # 只出对比图，不改文件
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


def load(D):
    co = json.load(open(os.path.join(D, "coords.json"), encoding="utf-8"))
    return co, {p["name"]: p for p in co["parts"]}


def canvas_of(D, co, names):
    W, H = co["canvas"]["w"], co["canvas"]["h"]
    parts = {p["name"]: p for p in co["parts"]}
    can = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for n in names:
        p = parts.get(n)
        if not p:
            continue
        can.alpha_composite(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"),
                            (p["x"], p["y"]))
    return np.array(can)


def synth(D, p_eye, p_head, p_lash):
    """保留头部皮肤，抹掉眼白/瞳孔，再画一条闭眼缝。"""
    import math
    from scipy import ndimage as ndi
    co, d = load(D)
    head = canvas_of(D, co, ["head"])
    x0, y0, w, h = p_eye["x"], p_eye["y"], p_eye["width"], p_eye["height"]

    base = head[y0:y0 + h, x0:x0 + w].astype(np.uint8).copy()
    if base.shape[0] != h or base.shape[1] != w:
        pad = np.zeros((h, w, 4), np.uint8)
        pad[:base.shape[0], :base.shape[1]] = base
        base = pad
    rgb = base[..., :3].astype(int)
    alpha = base[..., 3]
    # 整只眼睛的区域 = 眼白/瞳孔/睫毛三层在该框内的不透明像素
    eyes_can = canvas_of(D, co, ["eyewhite", "irides", "eyelash"])
    eye = eyes_can[y0:y0 + h, x0:x0 + w, 3] > 8
    if eye.shape != (h, w):
        pad = np.zeros((h, w), bool)
        pad[:eye.shape[0], :eye.shape[1]] = eye
        eye = pad
    # 睫毛/眼线是深色, 也算眼睛一部分
    mx = rgb.max(2); mn = rgb.min(2)
    dark_line = (mx < 90) & (alpha > 8)
    eye = eye | dark_line
    skin_px = rgb[(~eye) & (alpha > 8)]
    if len(skin_px) < 60:
        skin_px = rgb[alpha > 8] if (alpha > 8).any() else np.array([[150, 150, 140]])
    med = np.median(skin_px, axis=0)
    dark = np.percentile(skin_px, 8, axis=0)

    if eye.any():
        fill = np.zeros_like(base)
        fill[..., :3] = med.astype(np.uint8)
        fill[..., 3] = 255
        m3 = ndi.binary_dilation(eye, np.ones((3, 3)), iterations=1)
        m3 &= (alpha > 0)
        base[m3] = fill[m3]
        rgb = base[..., :3].astype(float)
        # 用周围皮肤轻微平滑, 去掉"补丁感"
        blur = ndi.uniform_filter(rgb, size=(5, 5, 1))
        base[..., :3] = np.where(m3[..., None], blur, rgb).astype(np.uint8)
    base[..., 3] = 255
    img = Image.fromarray(base)

    # 眼缝: 中间略高的弧线(上眼皮闭下)
    dr = ImageDraw.Draw(img)
    cx0, cx1 = int(w * 0.14), int(w * 0.86)
    if cx1 - cx0 < 4:
        cx0, cx1 = 0, w - 1
    cy = h * 0.56
    amp = max(1.0, h * 0.16)
    thick = max(1.6, h * 0.10)
    col = (int(dark[0] * 0.45), int(dark[1] * 0.45), int(dark[2] * 0.45), 255)
    for x in range(cx0, cx1):
        tt = (x - cx0) / max(1, (cx1 - cx0 - 1))
        yy = cy - amp * math.sin(math.pi * tt)
        dr.ellipse([x - thick / 2, yy - thick / 2, x + thick / 2, yy + thick / 2],
                   fill=col)
    return img


def main():
    D = os.path.abspath(sys.argv[1])
    dry = "--dry" in sys.argv
    co, d = load(D)
    p_eye = d.get("eyes_closed")
    if not p_eye:
        print(f"{D}: 没有 eyes_closed 部件，跳过")
        return
    img = synth(D, p_eye, d.get("head"), d.get("eyelash") or d.get("eyewhite"))
    if dry:
        out = os.path.join(D, "_eyes_closed_synth.png")
        img.save(out)
        print("试作 ->", out)
        return
    img.save(os.path.join(D, "parts", p_eye["image"]))
    print(f"已替换 {D}/parts/{p_eye['image']}  ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
