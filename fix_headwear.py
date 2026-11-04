"""给 headwear(头盔/头巾) 从原图补齐被"头/脸"那层啃掉的边(帽檐、垂布)。

拆图链路里做这件事会被后续的合并步骤覆盖掉(实测补完又被还原), 所以独立成一步:
拆完之后跑一次, 直接改 parts/headwear.png 和 coords.json 里的坐标。

用法:
    python fix_headwear.py <素材根目录> <原图.png>
    # 例: python fix_headwear.py output/sprite_suo_white_336 E:/pic/sprite_suo_white_336.png
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image


def amask(p, parts_dir, shape):
    im = np.array(Image.open(os.path.join(parts_dir, p["image"])).convert("RGBA"))
    m = np.zeros(shape, bool)
    m[p["y"]:p["y"] + im.shape[0], p["x"]:p["x"] + im.shape[1]] = im[..., 3] > 12
    return m


def _bbox_of(sel, ox, oy):
    """返回 (x, y, w, h)；sel 是相对裁剪图 out 的布尔掩码。"""
    ys, xs = np.where(sel)
    if len(xs) == 0:
        return ox, oy, 0, 0
    return ox + int(xs.min()), oy + int(ys.min()), \
        int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _write_part(parts_dir, fname, out, sel, ox, oy):
    bx, by, bw, bh = _bbox_of(sel, ox, oy)
    if bw == 0 or bh == 0:
        return None
    rgba = out.copy()
    rgba[..., 3] = np.where(sel, rgba[..., 3], 0)
    sub = rgba[by - oy:by - oy + bh, bx - ox:bx - ox + bw]
    Image.fromarray(sub).save(os.path.join(parts_dir, fname))
    return bx, by, bw, bh


def split_wear_materials(out, head_box=None):
    """把"头盔 + 头巾"拆成"刚性的头饰"和"可飘的垂布"。

    只按颜色分不行: 头巾带压在额头上、和帽檐紧挨, 布上的暗褶子又接近盔色, 分开后
    要么带头盔、要么被头盔盖住(都试过)。这里按【位置】分, 更贴近真实结构:

      - 横跨额头的那圈布是系在头上的 —— 和头盔一样刚性, 不参与飘动;
      - 垂到头部两侧之外的布(画面右那条长的、左边那一小截)才是会飘的部分。

    判据: 高饱和的黄/棕色(布) 且 落在 head 的水平范围之外。返回 (cloth, metal)。
    """
    a = out[..., 3] > 12
    if not a.any():
        return None
    hsv = cv2.cvtColor(out[..., :3].astype(np.uint8), cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(int)
    sat = hsv[..., 1].astype(int)
    cloth_hue = a & (h >= 8) & (h <= 60) & (sat >= 80)
    cloth_hue = cv2.morphologyEx(cloth_hue.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((3, 3), np.uint8)).astype(bool) & a
    if head_box is None:
        return None
    hx0, hx1 = int(head_box[0]), int(head_box[1])
    xs = np.arange(out.shape[1])[None, :]
    outside = (xs < hx0 - 2) | (xs > hx1 + 2)
    cloth = cloth_hue & outside
    if len(head_box) > 3:                       # 只要额头以下的部分: 头盔那层的东西
        y_lo = int(head_box[3])                 # (含剑柄上的皮带)不能跟着飘
        cloth[:max(0, y_lo), :] = False
        # 零散碎块(皮带、描边)整块丢掉, 只留成片的垂布
        n0, lab0, st0, _ = cv2.connectedComponentsWithStats(cloth.astype(np.uint8), 8)
        keep0 = np.zeros_like(cloth)
        for k in range(1, n0):
            ar = int(st0[k, cv2.CC_STAT_AREA])
            if ar < 120:
                continue
            sel = (lab0 == k)
            if float((sel & cloth_hue).sum()) / max(1, ar) < 0.5:
                continue
            keep0 |= sel
        cloth = keep0
    # 只要右侧那片够大就拆(左侧那截是带子末端, 位置还在头的范围内, 跟着头走就行)
    if int(cloth[:, hx1:].sum()) < 120:
        return None
    if int(cloth.sum()) < 400:
        return None
    # 只有"整体以金属为主"才拆: 看这块头饰的饱和度中位数。
    # 头盔灰 S≈20, 纯布头巾金黄 S≈145 —— 比任何阈值/连通块都稳。
    _hsv2 = cv2.cvtColor(out[..., :3].astype(np.uint8), cv2.COLOR_RGB2HSV)
    _sat2 = _hsv2[..., 1][a]
    if float(np.median(_sat2)) >= 60:
        return None
    _strong = a & (_hsv2[..., 1].astype(int) < 60) & (_hsv2[..., 2].astype(int) >= 110)
    if int(_strong.sum()) < 1000:
        return None
    metal = a & ~cloth
    if int(metal.sum()) < 1500:
        return None
    return cloth, metal


def main(root, orig):
    parts_dir = os.path.join(root, "parts")
    cpath = os.path.join(parts_dir, "coords.json")
    coords = json.load(open(cpath, encoding="utf-8"))
    src = np.array(Image.open(orig).convert("RGBA"))
    fg = src[..., 3] > 12
    # 原图是"带底色截图"(RGB, alpha 全 255)时不能拿 alpha 当前景 —— 那等于把整块
    # 背景也算成角色, 头饰就会拖着一大片底色(实测 fa 的头饰带了 4.6 万像素背景)。
    try:
        from build_from_seethrough import strip_bg
        src, fg = strip_bg(src)
        print(f"  原图带底色: 已抠掉背景, 前景 {int(fg.sum())}px")
    except Exception as ex:                             # noqa: BLE001
        print(f"  背景抠除跳过: {ex}")
    parts = {p["name"]: p for p in coords["parts"]}
    hd, fc, hw = parts.get("head"), parts.get("face"), parts.get("headwear")
    if hw is None or hd is None:
        print("没有 headwear/head, 跳过")
        return 1
    h, w = src.shape[:2]

    face_m = amask(fc, parts_dir, (h, w)) if fc else np.zeros((h, w), bool)
    if face_m.any():
        skin = src[..., :3][face_m].reshape(-1, 3).astype(np.float32).mean(0)
        dist = np.linalg.norm(src[..., :3].astype(np.float32) - skin, axis=2)
    else:
        dist = np.full((h, w), 999.0, np.float32)

    # 区域 = 头 ∪ 头饰 再往外扩: 帽檐比头宽得多, 只按头算会漏掉两翼
    x0 = max(0, min(hd["x"], hw["x"]) - 30)
    x1 = min(w, max(hd["x"] + hd["width"], hw["x"] + hw["width"]) + 30)
    y0 = max(0, hw["y"] - 10)
    y1 = min(h, hd["y"] + int(hd["height"] * 0.55))
    region = np.zeros((h, w), bool)
    region[y0:y1, x0:x1] = True

    import cv2 as _cv
    base = amask(hw, parts_dir, (h, w))          # 图层已经给的那部分头饰
    # 图层给的头饰本身就带着脸(前发/后发层把脸也划进去了)。先把"头/脸"的轮廓
    # 从底里挖掉, 再把"压在头上的布/金属"按材质补回来 —— 脸就进不去了。
    _hsil = np.zeros((h, w), bool)
    for _nm in ("head", "face"):
        _p = parts.get(_nm)
        if _p is None:
            continue
        _im = np.array(Image.open(os.path.join(parts_dir, _p["image"])).convert("RGBA"))
        _m = _im[..., 3] > 12
        _hsil[_p["y"]:_p["y"] + _im.shape[0], _p["x"]:_p["x"] + _im.shape[1]] |= _m
    if _hsil.any():
        base &= ~_hsil
    _hsv = _cv.cvtColor(src[..., :3].astype(np.uint8), _cv.COLOR_RGB2HSV)
    _hh = _hsv[..., 0].astype(int)
    _ss = _hsv[..., 1].astype(int)
    _vv = _hsv[..., 2].astype(int)
    _cloth = (_hh >= 20) & (_hh <= 50) & (_ss > 85) & (_vv > 90)     # 黄布/头巾(棕袍 H≈17, 排除)
    _eb = parts.get("eyebrow")
    _brow = _eb["y"] if _eb is not None else hd["y"] + int(hd["height"] * 0.45)
    _above = np.zeros((h, w), bool)
    _above[:max(0, int(_brow) + 2), :] = True
    _metal = (_ss < 55) & (_vv > 70) & _above                        # 金属盔(只在额头以上)
    # 只补"布料/金属"色的像素: 以前按"非肤色前景"补, 会把脸和衣服整块框进来 ——
    # 脸是灰绿的, 离肤色并不远但也不等于肤色, 减不掉。改成只认这两种材质。
    mask = base | (fg & region & (_cloth | _metal))

    # 这片区域是按"非肤色前景"取的, 会把画在脸上的眉/眼/睫一起框进来 ——
    # 头饰部件于是自己带着半张脸。这些像素都有独立部件负责画, 从这里摘掉不影响显示。
    cut = 0
    for nm in ("eyebrow", "irides", "eyelash", "eyes_closed", "nose", "mouth"):
        fp = parts.get(nm)
        if fp is None:
            continue
        fm = amask(fp, parts_dir, (h, w)) & mask
        if int(fm.sum()) < 10:
            continue
        # 只摘"颜色也是五官那个色"的像素: 直接按 mask 减会把压在五官上的黄头巾
        # 一起削出洞(实测眉/闭眼补丁的外框压在头巾带上)。按颜色挑才不会误伤。
        ref = src[..., :3][fm].reshape(-1, 3).astype(np.float32).mean(0)
        d = np.linalg.norm(src[..., :3].astype(np.float32) - ref, axis=2)
        kill = fm & (d < 45)
        n = int(kill.sum())
        if n < 10:
            continue
        mask &= ~kill
        cut += n
        print(f"  摘掉混进来的 {nm}: {n}px")

    # 头巾带下面还挂着一层"额头/眉毛"的皮肤(灰绿), 那是脸不是头巾 —— 摘掉。
    # 判据: 每列找到"黄布"的最低点, 比它更靠下、又不是黄布的前景就不要了
    # (垂下来的布尾本身是黄的, 不受影响)。
    import cv2
    hsv = cv2.cvtColor(src[..., :3].astype(np.uint8), cv2.COLOR_RGB2HSV)
    hh, ss, vv = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    yellow = (hh >= 14) & (hh <= 40) & (ss > 90) & (vv > 90)
    drop = 0
    for cx in range(mask.shape[1]):
        col = np.where(mask[:, cx])[0]
        if col.size == 0:
            continue
        ylow = np.where(yellow[:, cx] & mask[:, cx])[0]
        if ylow.size == 0:
            continue
        yb = int(ylow.max())
        below = mask[yb + 3:, cx] & ~yellow[yb + 3:, cx]
        n = int(below.sum())
        if n:
            mask[yb + 3:, cx] &= yellow[yb + 3:, cx]
            drop += n
    if drop:
        print(f"  摘掉带子下面挂着的面部: {drop}px")
    # 摘五官时留下的 1~3px 小窟窿补回去(闭运算), 免得部件上一堆针眼
    _m8 = mask.astype(np.uint8)
    _m8 = cv2.morphologyEx(_m8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _m8 = cv2.morphologyEx(_m8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = _m8.astype(bool) & fg

    # 头饰里混进来的"脸"直接按皮肤色删: 皮肤色从 head/face 两个部件里取,
    # 而且只在头的轮廓范围内删 —— 布/袍子/金属离皮肤色都够远, 不会被误伤。
    # (试过"只留黄布"的写法: 头巾的暗褶子会被啃成渔网, 所以改成反向删皮肤。)
    _skins = []
    _sil = np.zeros((h, w), bool)
    for _nm in ("head", "face"):
        _p = parts.get(_nm)
        if _p is None:
            continue
        _im = np.array(Image.open(os.path.join(parts_dir, _p["image"])).convert("RGBA"))
        _m = _im[..., 3] > 12
        if not _m.any():
            continue
        _skins.append(_im[..., :3][_m].reshape(-1, 3).astype(np.float32).mean(0))
        _sil[_p["y"]:_p["y"] + _im.shape[0], _p["x"]:_p["x"] + _im.shape[1]] |= _m
    if _skins:
        _d = np.min(np.stack([np.linalg.norm(src[..., :3].astype(np.float32) - c, axis=2)
                              for c in _skins]), axis=0)
        _kill = mask & _sil & (_d < 42)
        _n = int(_kill.sum())
        if _n:
            mask &= ~_kill
            print(f"  头饰去皮肤: {_n}px (脸混进来了)")
    # 带底色的原图: 底色可能被"包在角色轮廓里"而没被泛洪抠掉, 这里按颜色再删一遍
    _raw = np.array(Image.open(orig).convert("RGBA"))
    if (_raw[..., 3] < 12).mean() < 0.02:
        _pts = np.concatenate([_raw[:6, :6, :3].reshape(-1, 3),
                               _raw[:6, -6:, :3].reshape(-1, 3),
                               _raw[-6:, :6, :3].reshape(-1, 3),
                               _raw[-6:, -6:, :3].reshape(-1, 3)])
        _bg = np.median(_pts, axis=0)
        _d = np.linalg.norm(src[..., :3].astype(np.float32) - _bg, axis=2)
        _raw_k = mask & (_d < 34)
        # 同样只删贴边的底色块, 免得把布上的暗褶子当底色削掉
        import cv2 as _c2
        _nl, _ll = _c2.connectedComponents(_raw_k.astype(np.uint8), 8)
        _edge = set(_ll[0, :].tolist()) | set(_ll[-1, :].tolist()) \
            | set(_ll[:, 0].tolist()) | set(_ll[:, -1].tolist())
        _edge.discard(0)
        _kill = np.isin(_ll, list(_edge)) if _edge else np.zeros_like(mask)
        _n = int(_kill.sum())
        if _n:
            mask &= ~_kill
            print(f"  头饰去底色: {_n}px (底色 {_bg.astype(int).tolist()})")

    # 最后清一遍"零散杂块": 每块独立区域算一下"布料占比", 几乎不含布的
    # (残留在下面的袍子、背景碎片)整块丢掉。头巾本身布占比很高, 不会误删。
    import cv2 as _cv2
    _n, _lb, _st, _ = _cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    _drop = np.zeros_like(mask)
    for _k in range(1, _n):
        _comp = (_lb == _k)
        _area = int(_comp.sum())
        if _area < 30:
            _drop |= _comp
            continue
        _frac = float((_comp & _cloth).sum()) / _area
        if _frac < 0.12:
            _drop |= _comp
    _nd = int(_drop.sum())
    if _nd:
        mask &= ~_drop
        print(f"  头饰清杂块: {_nd}px (不含布的碎块)")

    # 混进来的"衣服"(棕袍): 颜色接近 topwear 主色、色相又明显比布低(棕 H≈17,
    # 黄布 H≈30)的整块直接摘掉。衣服总是搭在肩膀/脖子下方, 所以只在该部件外框
    # 的下半部分动手, 免得误伤帽檐边缘的深色描边。
    _tw = parts.get("topwear")
    if _tw is not None:
        _tim = np.array(Image.open(os.path.join(parts_dir, _tw["image"])).convert("RGBA"))
        _tm = _tim[..., 3] > 12
        if _tm.any():
            _tcol = _tim[..., :3][_tm].reshape(-1, 3).astype(np.float32).mean(0)
            _d = np.linalg.norm(src[..., :3].astype(np.float32) - _tcol, axis=2)
            import cv2 as _cv4
            _hsv4 = _cv4.cvtColor(src[..., :3].astype(np.uint8), _cv4.COLOR_RGB2HSV)
            _hh4 = _hsv4[..., 0].astype(int)
            _robe = mask & (_d < 46) & (_hh4 < 22)
            _n = int(_robe.sum())
            if _n > 40:
                mask &= ~_robe
                print(f"  头饰去衣服: {_n}px (棕袍色)")

    # 下巴以下只留布: 头巾垂下来的两条是黄的, 其他(袍子、肩膀、脖子)都不是头饰。
    # 这条比"按衣服色删"可靠 —— 袍子在阴影里时颜色离主色很远, 按色删不掉。
    _mo = parts.get("mouth")
    if _mo is not None:
        _chin = int(_mo["y"] + _mo["height"]) + 2
        if _chin < h:
            import cv2 as _cv5
            _hsv5 = _cv5.cvtColor(src[..., :3].astype(np.uint8), _cv5.COLOR_RGB2HSV)
            _h5 = _hsv5[..., 0].astype(int)
            _s5 = _hsv5[..., 1].astype(int)
            _v5 = _hsv5[..., 2].astype(int)
            _cl = (_h5 >= 16) & (_h5 <= 52) & (_s5 > 70) & (_v5 > 55)
            _sub = mask[_chin:, :] & ~_cl[_chin:, :]
            _n = int(_sub.sum())
            if _n:
                mask[_chin:, :] &= _cl[_chin:, :]
                print(f"  头饰下巴以下去非布: {_n}px")
                # 零散碎点: 整块判断 —— 只保留"从布的主体连下来"的垂布,
                # 也就是这块里含有下巴以上的像素; 完全落在下巴以下的都是
                # 袍子/肩膀的反锯齿碎片, 全丢。
                _nb, _lb, _sb, _ = _cv5.connectedComponentsWithStats(mask.astype(np.uint8), 8)
                _bad = 0
                for _k in range(1, _nb):
                    _comp = (_lb == _k)
                    _top = int(_sb[_k, _cv5.CC_STAT_TOP])
                    _ar = int(_sb[_k, _cv5.CC_STAT_AREA])
                    if _top >= _chin and _ar < 4000:
                        mask &= ~_comp
                        _bad += _ar
                if _bad:
                    print(f"  头饰下巴以下清碎点: {_bad}px")

    # 补"被包在里面的小洞": 布/盔是一整块实心物, 中间不该透出背景。
    # 这些洞是上面几步(挖头脸轮廓、去皮肤、按色补)交接处漏掉的窄条, 比如头盔
    # 帽檐压着的那圈头巾。只补【不与外界连通】且面积小的, 所以脸(与背景连通)、
    # 以及真正镂空的地方都不会被填回去。
    import cv2 as _cv3
    _inv = (~mask).astype(np.uint8)
    _n3, _l3, _s3, _ = _cv3.connectedComponentsWithStats(_inv, 4)
    _fill = np.zeros_like(mask)
    _filled = 0
    for _k in range(1, _n3):
        _ar = int(_s3[_k, _cv3.CC_STAT_AREA])
        if _ar < 4 or _ar > 1200:
            continue
        _x = int(_s3[_k, _cv3.CC_STAT_LEFT]); _y = int(_s3[_k, _cv3.CC_STAT_TOP])
        _ww = int(_s3[_k, _cv3.CC_STAT_WIDTH]); _hh = int(_s3[_k, _cv3.CC_STAT_HEIGHT])
        if _x == 0 or _y == 0 or _x + _ww >= w or _y + _hh >= h:
            continue                              # 贴边 = 和外界连通, 不填
        _mask_k = (_l3 == _k)
        _fill |= _mask_k
        _filled += _ar
    if _filled:
        mask |= _fill
        print(f"  头饰补内洞: {_filled}px ({int(_fill.sum()) and '布面/盔面漏缝'})")

    ys, xs = np.where(mask)
    if len(xs) == 0:
        print("没找到可补的像素")
        return 1
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    out = src[by0:by1, bx0:bx1].copy()
    out[..., 3] = np.where(mask[by0:by1, bx0:bx1], out[..., 3], 0)

    # 头盔 + 头巾连成一块时按材质拆开: 盔是刚性(跟着头走, 一点都不该飘),
    # 头巾垂下来的两条是布(可以飘)。合成一块的话只能二选一 —— 要么盔跟着飘
    # (用户反馈"头盔上部不该飘"), 要么头巾僵死。
    # 不管选的是"头巾"还是"头盔"模式, 只要这块里同时有"金属盔"和"垂布"就拆开 ——
    # 头盔是硬的, 不能因为模式选了"头巾(可飘)"就整顶跟着飘(用户实测踩过)。
    _split = split_wear_materials(
        out, (hd["x"] - bx0, hd["x"] + hd["width"] - bx0,
              hd["y"] - by0, hd["y"] + int(hd["height"] * 0.18) - by0))
    if _split is not None:
        cloth, metal = _split
        # 头巾轮廓上的深色描边/褶子也算布, 否则拆开后会在布上留下一条条盔色硬条
        cloth |= ((out[..., 3] > 12) & (out[..., :3].max(axis=2) < 120)
                  & cv2.dilate(cloth.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
        cloth &= (out[..., 3] > 12)
        metal &= (out[..., 3] > 12) & ~cloth
        # 存一份"合并版"底图: 界面切换头饰模式时会把两块并回一块(只改 coords,
        # 不动 png), 那时再想拆就得有这张原始图。
        try:
            Image.fromarray(out).save(os.path.join(parts_dir, "headwear_full.png"))
        except OSError:
            pass
        _mb = _write_part(parts_dir, hw["image"], out, metal, bx0, by0)
        # 露出金属的这块一律按"刚性头饰"处理: 壳是硬的, 不能跟着布一起飘
        hw.update({"x": _mb[0], "y": _mb[1], "width": _mb[2], "height": _mb[3],
                   "cls": "headgear"})
        sc_name = "headscarf"
        _sp = dict(hw)
        _sp.update({"name": sc_name, "image": sc_name + ".png", "cls": "cloth"})
        _sp.pop("bind_bone", None)
        # 注意: 这块 png 是按"布"自己的外框裁的, x/y 必须用它的外框,
        # 不能沿用 headwear 的(否则垂布会被贴到头盔左上角)
        _cb = _write_part(parts_dir, sc_name + ".png", out, cloth, bx0, by0)
        _sp.update({"x": _cb[0], "y": _cb[1], "width": _cb[2], "height": _cb[3]})
        print(f"  头盔/头巾拆开: 盔 {int(metal.sum())}px(刚性) + 布 {int(cloth.sum())}px(可飘)")
        # headscarf 必须排在 headwear 之前(先画布、再压头盔), 否则布会盖住帽檐
        plist = [q for q in coords["parts"] if q["name"] != sc_name]
        idx = next(i for i, q in enumerate(plist) if q["name"] == "headwear")
        plist.insert(idx, _sp)
        coords["parts"] = plist
    else:
        Image.fromarray(out).save(os.path.join(parts_dir, hw["image"]))
        hw.update({"x": bx0, "y": by0, "width": bx1 - bx0, "height": by1 - by0})
        # 已经有一块独立的 headscarf(布)时, 这块就是"盔", 必须刚性 ——
        # 切换模式会把它的 cls 改成 cloth, 这里纠正回来。
        if any(q["name"] == "headscarf" for q in coords["parts"]):
            hw["cls"] = "headgear"
    json.dump(coords, open(cpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"头饰补齐 -> {hw['image']} {bx1 - bx0}x{by1 - by0}, 不透明 {int(mask.sum())}px")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2]))
