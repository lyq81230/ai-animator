# -*- coding: utf-8 -*-
"""把小到没有意义的部件并回相邻部件。

拆图会留下一些"碎片段": 手臂沿主轴切三段时, 交界处会切出 1x13、1x4 这种
一条线, 或者 8x7 的小疙瘩。它们在骨骼里会变成各自独立的槽位/骨骼 ——
既没法看, 还会干扰"哪块是真正的大臂"的判断(魔神左大臂就是这么被 8x7
碎块顶掉、整块绑到躯干上的)。这里把这类碎片按"离谁最近"并回去,
像素一个不丢。

判据(只对结构性部件生效, 五官/头发一律不碰):
    px == 0                                    -> 空片, 直接并
    px < 120 且 (短边 <= 3 或 px <= 60)        -> 细条 / 小疙瘩
保护: 同一 cls 里最大的那片不并(避免把某类部件整个吃掉)。

用法:
    python merge_tiny_parts.py <work_dir> [min_px] [--dry]
"""
import json
import os
import sys

import numpy as np
from PIL import Image

# 只处理这些"身体结构"类: 五官/眼睛/头发/飘带都有实际用处, 再小也不能并
STRUCT = {"arm_upper", "arm_fore", "arm_hand", "leg", "leg_up", "leg_low",
          "foot", "footwear", "prop", "weapon", "shield", "body", "cloth"}
MIN_PX = 120.0


def _family(cls):
    """把 cls 归到"同一套肢体"里: 碎片的宿主优先选同一套的部件。
    手/臂的 1px 细条并到腿上, 走路时那一条就跟着腿走 —— 必须避免。"""
    c = str(cls or "")
    if c.startswith("arm"):
        return "arm"
    if c.startswith("leg") or c.startswith("foot"):
        return "leg"
    if c in ("prop", "weapon", "shield"):
        return "prop"
    return "body"
SLIVER_SIDE = 3
BLOB_PX = 60


def _load(path):
    return np.array(Image.open(path).convert("RGBA"))


def _is_junk(px, w, h):
    if px == 0:
        return True
    if px >= MIN_PX:
        return False
    return (min(w, h) <= SLIVER_SIDE) or (px <= BLOB_PX)


def _pix_dist(a, b):
    """两片不透明像素之间的最近距离(像素)。"""
    ya, xa = np.nonzero(a[..., 3] > 40)
    yb, xb = np.nonzero(b[..., 3] > 40)
    if not len(ya) or not len(yb):
        return 1e9
    pa = np.column_stack([xa, ya]).astype(np.float32)
    pb = np.column_stack([xb, yb]).astype(np.float32)
    try:
        from scipy.spatial import cKDTree
        return float(cKDTree(pb).query(pa, k=1)[0].min())
    except Exception:                                  # noqa: BLE001
        # 退化: 逐点算(部件小, 慢不到哪去)
        d = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(axis=2)
        return float(np.sqrt(d.min()))


def merge(work, dry=False, verbose=True):
    work = work.rstrip("/\\")
    cp = os.path.join(work, "coords.json")
    pdir = os.path.join(work, "parts")
    if not os.path.isfile(cp):
        return 0
    co = json.load(open(cp, encoding="utf-8"))
    parts = co["parts"]

    imgs, masks, px = {}, {}, {}
    for p in parts:
        f = os.path.join(pdir, p["image"])
        if not os.path.isfile(f):
            continue
        imgs[p["name"]] = _load(f)
        masks[p["name"]] = imgs[p["name"]][..., 3] > 40
        px[p["name"]] = int(masks[p["name"]].sum())

    def biggest_of(p, key):
        grp = [q for q in parts if key(q) == key(p) and q["name"] in px]
        if not grp:
            return True
        return px[p["name"]] >= max(px[q["name"]] for q in grp)

    n_merged = 0
    for p in list(parts):
        nm = p["name"]
        if nm not in px:
            continue
        # 空片没有像素可搬, 直接删(不参与"最大片"判断, 否则单独成类的空片会漏掉)
        if px[nm] == 0:
            for q in parts:
                for key in ("follow", "follow_part"):
                    if q.get(key) == nm:
                        q[key] = None
            print("  空片 %-16s 删除" % nm)
            if not dry:
                f = os.path.join(pdir, p["image"])
                if os.path.isfile(f):
                    os.remove(f)
                parts.remove(p)
            n_merged += 1
            continue
        cls = str(p.get("cls") or "")
        if cls not in STRUCT:
            continue
        if not _is_junk(px[nm], p["width"], p["height"]):
            continue
        # 同一类里最大的那片是"真身", 不许并掉
        if biggest_of(p, lambda q: str(q.get("cls") or "")):
            continue
        # 同侧优先: 用 x 中心和画布中线比
        _cw = co["canvas"].get("w") or co["canvas"].get("width") or 336
        side = 0 if p["x"] + p["width"] / 2 < _cw / 2 else 1

        cands = []
        for q in parts:
            if q is p or q["name"] not in px:
                continue
            if str(q.get("cls") or "") not in STRUCT:
                continue
            # 宿主必须自己不是碎片, 否则会串成 A->B->C
            if px[q["name"]] == 0 or _is_junk(px[q["name"]], q["width"], q["height"]):
                continue
            d = _pix_dist(imgs[nm], imgs[q["name"]])
            pen = 0.0 if _family(q.get("cls")) == _family(cls) else 20.0
            qs = 0 if (q["x"] + q["width"] / 2 < _cw / 2) == (side == 0) else 30.0
            cands.append((d + pen + qs, d, q))
        if not cands:
            continue
        cands.sort(key=lambda t: t[0])
        _, dist, host = cands[0]

        # 把碎片像素补到宿主【透明处】: 宿主原有的像素一个不动, 只填缝。
        # 碎片常常落在宿主 bbox 之外(比如 x1x13 的细条在手臂下方 17px), 所以先按
        # 两者的并集扩一块画布, 再往里贴 —— 只在原数组里贴的话会整条丢掉。
        H = imgs[host["name"]]
        hx, hy = host["x"], host["y"]
        hh, hw = H.shape[:2]
        jx, jy = p["x"], p["y"]
        src = imgs[nm]
        sh, sw = src.shape[:2]
        x0, y0 = min(hx, jx), min(hy, jy)
        x1 = max(hx + hw, jx + sw)
        y1 = max(hy + hh, jy + sh)
        canvas = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        canvas[hy - y0:hy - y0 + hh, hx - x0:hx - x0 + hw] = H
        for yy in range(sh):
            if src[yy, :, 3].max() <= 40:
                continue
            ty = jy + yy - y0
            for xx in range(sw):
                if src[yy, xx, 3] <= 40:
                    continue
                tx = jx + xx - x0
                if canvas[ty, tx, 3] < 40:          # 宿主那里是空的才补
                    canvas[ty, tx] = src[yy, xx]
        ys, xs = np.nonzero(canvas[..., 3] > 40)
        bx0, bx1 = int(xs.min()), int(xs.max()) + 1
        by0, by1 = int(ys.min()), int(ys.max()) + 1
        H2 = canvas[by0:by1, bx0:bx1]
        host["x"] = x0 + bx0
        host["y"] = y0 + by0
        host["width"], host["height"] = H2.shape[1], H2.shape[0]
        imgs[host["name"]] = H2
        masks[host["name"]] = H2[..., 3] > 40
        px[host["name"]] = int(masks[host["name"]].sum())

        # 有部件把这个碎片当成"跟随目标"(武器的 follow_part 常指向某只小臂),
        # 碎片被并掉后引用就成了死指针 —— 魔神的 objects_L 因此失去跟随目标、
        # 退化成"贴躯干", 整把左刀钉在身上不动。这里改指向宿主。
        for q in parts:
            for key in ("follow", "follow_part"):
                if q.get(key) == nm:
                    q[key] = host["name"]
                    if verbose:
                        print("  跟随引用 %-14s.%s: %s -> %s"
                              % (q["name"], key, nm, host["name"]))

        if verbose:
            print("  碎片 %-16s %4dpx %dx%d -> 并入 %-14s (距离%.1f)"
                  % (nm, px[nm], p["width"], p["height"], host["name"], dist))
        if not dry:
            Image.fromarray(imgs[host["name"]], "RGBA").save(
                os.path.join(pdir, host["image"]))
            f = os.path.join(pdir, p["image"])
            if os.path.isfile(f):
                os.remove(f)
            parts.remove(p)
        n_merged += 1

    if n_merged and not dry:
        json.dump(co, open(cp, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print("  碎片合并: %d 个" % n_merged)
    return n_merged


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    work = sys.argv[1]
    dry = "--dry" in sys.argv
    nums = [a for a in sys.argv[2:] if not a.startswith("--")]
    if nums:
        global MIN_PX
        MIN_PX = float(nums[0])
    return 0 if merge(work, dry=dry) >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
