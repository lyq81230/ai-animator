# -*- coding: utf-8 -*-
"""把"不该属于手臂"的盔甲尖角从手臂部件里摘掉(左右对称判据)。

现象(魔神): arm_upper_R 里带着肩甲上那根大尖角 —— 而 arm_upper_L 没有(左边那根
尖角当初被切给了 topwear / wings)。于是走路时右肩的角跟着上臂甩出去、左肩不动,
两边不对称, 看着像"手臂上挂着多余的东西"。

判据: 这个角色是左右镜像画的(魔神镜像常数 339, 逐像素平均色差 20)。所以
    "R 件里有、而镜像的 L 件里没有" 的那几块 = 不是手臂自己的东西。
只摘掉**在别的部件里也画着**的那些块(覆盖率 >= 85%), 所以绑定姿态下一个像素都不丢 ——
摘掉的地方本来就由躯干/披风在画, 只是不再跟着手臂动。

用法:
    python trim_arm_dup.py output/<角色> [--dry] [--min-cov 0.85] [--min-px 20]
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

PAIRS = (("arm_upper_L", "arm_upper_R"),
         ("arm_fore_L", "arm_fore_R"),
         ("arm_hand_L", "arm_hand_R"))


def load_parts(D, co):
    H, W = co["canvas"]["h"], co["canvas"]["w"]
    out = {}
    for p in co["parts"]:
        a = np.array(Image.open(os.path.join(D, "parts", p["image"])).convert("RGBA"))
        m = np.zeros((H, W, 4), np.uint8)
        m[p["y"]:p["y"] + a.shape[0], p["x"]:p["x"] + a.shape[1]] = a
        out[p["name"]] = m
    return out


def mirror(mask, K):
    """x' = K - x"""
    out = np.zeros_like(mask)
    W = mask.shape[1]
    for x in range(W):
        xm = K - x
        if 0 <= xm < W:
            out[:, xm] = mask[:, x]
    return out


def best_axis(a, b, W):
    """找让 a 镜像后与 b 最贴合的常数 K"""
    best = (-1.0, None)
    # 镜像常数的量级就是画布宽(轴在 W/2 附近 -> x' = 2*axis - x ≈ W - x),
    # 实际角色的轴不一定正好在中间, 所以绕着 W 扫一圈。
    for K in range(W - 40, W + 40):
        mi = mirror(a, K)
        inter = int((mi & b).sum())
        union = int((mi | b).sum())
        iou = inter / max(1, union)
        if iou > best[0]:
            best = (iou, K)
    return best[1], best[0]


def main():
    D = sys.argv[1].rstrip("/\\")
    dry = "--dry" in sys.argv
    min_cov = 0.85
    min_px = 20
    for i, a in enumerate(sys.argv):
        if a == "--min-cov":
            min_cov = float(sys.argv[i + 1])
        if a == "--min-px":
            min_px = int(sys.argv[i + 1])

    cp = os.path.join(D, "coords.json")
    co = json.load(open(cp, encoding="utf-8"))
    # 只有 config/<角色>/rig.json 里显式打开才动 —— 这个判据会"摘像素",
    # 别的角色还没逐个肉眼确认过, 不能一把梭。
    rig = {}
    rp = os.path.join(D, "rig.json")
    if os.path.isfile(rp):
        try:
            rig = json.load(open(rp, encoding="utf-8"))
        except Exception:                                  # noqa: BLE001
            rig = {}
    if not rig.get("trim_arm_dup"):
        print("  未启用 trim_arm_dup, 跳过")
        return
    by = {p["name"]: p for p in co["parts"]}
    imgs = load_parts(D, co)
    H, W = co["canvas"]["h"], co["canvas"]["w"]

    changed = 0
    for nl, nr in PAIRS:
        if nl not in by or nr not in by:
            continue
        aL = imgs[nl][..., 3] > 40
        aR = imgs[nr][..., 3] > 40
        K, iou = best_axis(aL, aR, W)
        if K is None or iou < 0.55:
            print("  %s/%s 镜像拟合太差(IoU %.2f), 跳过" % (nl, nr, iou))
            continue
        mirL = mirror(aL, K)
        mirR = mirror(aR, K)
        # "别的部件"= 除这两个以外的所有部件
        other = np.zeros((H, W), bool)
        for nm, m in imgs.items():
            if nm in (nl, nr):
                continue
            other |= m[..., 3] > 40
        for part, extra in ((nr, aR & ~ndi.binary_dilation(mirL, np.ones((3, 3)))),
                            (nl, aL & ~ndi.binary_dilation(mirR, np.ones((3, 3))))):
            if not extra.any():
                continue
            lab, n = ndi.label(extra, np.ones((3, 3)))
            for i in range(1, n + 1):
                blk = lab == i
                size = int(blk.sum())
                if size < min_px:
                    continue
                cov = float((blk & other).sum()) / max(1, size)
                if cov < min_cov:
                    continue
                ys, xs = np.nonzero(blk)
                print("  %-12s 摘掉 %4dpx  x%d-%d y%d-%d  (别处覆盖率 %.0f%%)"
                      % (part, size, xs.min(), xs.max(), ys.min(), ys.max(), 100 * cov))
                imgs[part][blk] = 0
                changed += size

    if not changed:
        print("  无需修改")
        return

    if dry:
        print("  [dry] 共 %d px, 未写入" % changed)
        # 还是把预览图吐出来
        for nm in imgs:
            a = imgs[nm][..., 3]
            if not a.any():
                continue
            ys, xs = np.nonzero(a > 40)
            sub = imgs[nm][ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            if nm in [p for pr in PAIRS for p in pr]:
                Image.fromarray(sub, "RGBA").save("_dry_%s.png" % nm)
        return

    # 写回: 重新裁 bbox 并更新 coords
    for p in co["parts"]:
        nm = p["name"]
        a = imgs[nm][..., 3]
        if not a.any():
            continue
        ys, xs = np.nonzero(a > 40)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub = imgs[nm][y0:y1, x0:x1]
        Image.fromarray(sub, "RGBA").save(os.path.join(D, "parts", p["image"]))
        p.update({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0})
    json.dump(co, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  写回完成, 共摘掉 %d px" % changed)


if __name__ == "__main__":
    main()
