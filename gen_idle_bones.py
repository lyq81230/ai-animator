#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_idle_bones.py — 全骨骼驱动的 Idle 待机生成器 (Spine 3.8)。

与 gen_idle2.py(逐帧 mesh deform 烘焙) 的区别:
  * 自动生成骨架(bones) + 加权蒙皮 mesh(weighted vertices)
  * 动画驱动的是【骨骼】的 rotate/translate/scale 时间线, 不是顶点 deform
  * 文件体积小, 且可在 Spine 编辑器里直接拖骨骼二次编辑

骨架设计(自动按部件分类/几何放置):
  root
   └ torso           (呼吸: scaleY + translateY; 底部锁定靠权重而非骨)
      ├ neck         (随头轻动)
      │  └ head      (2.5D 转头: rotate + 多段竖骨近似柱面)
      │     ├ eye_L/R  (眨眼: scaleY 压扁)
      │     ├ brow     (随眨眼微动)
      │     └ facefeat (鼻/嘴跟随 head)
      ├ armL_up→armL_fore→armL_hand  (FK 摆动)
      ├ armR_up→armR_fore→armR_hand
      ├ hair_k_0→hair_k_1            (每片头发一条 2 段骨链, 风吹)
      └ ribbon_k_0→ribbon_k_1        (每片飘带一条骨链, 风吹)

加权蒙皮约定(关键, 保证零误差还原 setup pose):
  所有骨 setup rotation=0, 仅有世界平移。则顶点对某骨的"局部坐标"=
  顶点世界坐标 − 骨世界坐标。单骨权重=1 时, 骨不动则顶点不动, 完全等价静态 mesh。
  多骨混合时按权重线性插值各骨贡献(Spine 标准 linear blend skinning)。

用法同 gen_idle2.py, 输出 result.json (--out)。
"""
import argparse
import json
import math
import os
import shutil

import numpy as np
from PIL import Image

from gen_idle2 import (_alpha_bbox, classify, spring_follow)
from video2spine import (build_grid_mesh, build_shape_mesh, video_to_spine,
                         decimate, trim_deform)


def _decimate_vec(flat, fps, tol):
    """向量帧序列(F,D)抽稀, 返回保留帧索引。复用 video2spine.decimate。"""
    return decimate(flat, tol)


def _trim_deform(flat):
    """去首尾全零偏移, 返回 (offset, trimmed_list) 或 (None, None)。"""
    return trim_deform(flat)

EPS = 1e-6
_LAST_ANIM = {}


# ----------------------------------------------------------------------------
# 骨架数据结构
# ----------------------------------------------------------------------------
class Bone:
    """一根骨。world(wx,wy) 为 setup 世界坐标(Spine y 向上)。
    setup rotation 一律 0 -> 顶点局部坐标 = 世界 - 骨世界, 还原零误差。
    动画里写它的 rotate/translate/scale 时间线。"""
    __slots__ = ("name", "parent", "wx", "wy", "rot0", "children", "index",
                 "sx", "sy")

    def __init__(self, name, parent, wx, wy):
        self.name = name
        self.parent = parent          # Bone or None
        self.wx = wx
        self.wy = wy
        self.rot0 = 0.0
        self.sx = 1.0                 # setup 缩放: 闭眼补丁骨用 0.001 常驻隐藏
        self.sy = 1.0
        self.children = []
        self.index = 0                # 在 bones 列表中的序号(蒙皮引用)

    def local_xy(self):
        """相对父骨的 setup 偏移(父 rotation=0, 故就是世界差)。"""
        if self.parent is None:
            return self.wx, self.wy
        return self.wx - self.parent.wx, self.wy - self.parent.wy


class Skeleton:
    def __init__(self):
        self.root = Bone("root", None, 0.0, 0.0)
        self.bones = [self.root]
        self._byname = {"root": self.root}
        self.transforms = []          # transform constraints (2.5D 转面等)

    def add(self, name, parent, wx, wy):
        b = Bone(name, parent, wx, wy)
        b.index = len(self.bones)
        parent.children.append(b)
        self.bones.append(b)
        self._byname[name] = b
        return b

    def add_transform(self, name, bones, target, translate_mix=0.0,
                      rotate_mix=0.0, local=True, relative=True):
        """加 transform constraint。relative+local: target 的局部变换增量
        按 mix 叠加到受约束骨 -> 实现 2.5D 视差(各五官 mix 不同=深度不同)。"""
        self.transforms.append(dict(
            name=name, order=len(self.transforms), bones=list(bones),
            target=target, translateMix=translate_mix, rotateMix=rotate_mix,
            scaleMix=0.0, shearMix=0.0, local=local, relative=relative))

    def get(self, name):
        return self._byname.get(name)

    def export_transforms(self):
        """导出 Spine transform 约束数组(仅含非零 mix 的项)。"""
        out = []
        for c in self.transforms:
            d = {"name": c["name"], "order": c["order"],
                 "bones": c["bones"], "target": c["target"],
                 "rotateMix": round(c["rotateMix"], 4),
                 "translateMix": round(c["translateMix"], 4),
                 "scaleMix": 0, "shearMix": 0}
            if c.get("local"):
                d["local"] = True
            if c.get("relative"):
                d["relative"] = True
            out.append(d)
        return out

    def export(self):
        """导出 Spine bones 列表 (含 length 仅为可视, 不影响计算)。"""
        out = []
        for b in self.bones:
            d = {"name": b.name}
            if b.parent is not None:
                d["parent"] = b.parent.name
                lx, ly = b.local_xy()
                if abs(lx) > EPS:
                    d["x"] = round(lx, 2)
                if abs(ly) > EPS:
                    d["y"] = round(ly, 2)
            # setup 缩放: 不写时引擎按 1 处理, 闭眼补丁会在绑定姿态就整块显出来
            if abs(b.sx - 1.0) > 1e-6:
                d["scaleX"] = round(b.sx, 4)
            if abs(b.sy - 1.0) > 1e-6:
                d["scaleY"] = round(b.sy, 4)
            out.append(d)
        return out


# ----------------------------------------------------------------------------
# 加权蒙皮: 把 mesh 顶点绑定到骨骼
# ----------------------------------------------------------------------------
def skin_vertices(verts_spine, bone_weights):
    """生成 Spine 加权 mesh 的 vertices 数组。
    verts_spine: [(sx, sy), ...] 顶点世界坐标(y 向上)
    bone_weights: [ [(bone, w), ...], ... ] 每顶点的(骨,权重)列表(权重已归一)
    返回 flat list: 每顶点 = [n, boneIdx, lx, ly, w, ...] (n 为绑定骨数)。
    局部坐标 lx,ly = 世界 - 骨世界 (因所有骨 setup rotation=0)。"""
    out = []
    for (sx, sy), bws in zip(verts_spine, bone_weights):
        out.append(len(bws))
        for bone, w in bws:
            lx = sx - bone.wx
            ly = sy - bone.wy
            out += [bone.index, round(lx, 2), round(ly, 2), round(w, 4)]
    return out


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------
# 部件加载 (复用 gen_idle2 的满画布 bbox 还原 + 分类)
# ----------------------------------------------------------------------------
def _split_foot_pair(im):
    """把"两只鞋连在一张图"的 footwear 按连通域拆成左右两块。

    返回 [(side, PIL.Image, (x0, y0, w, h))] 或 [] (只有一只/分不开)。
    老拆图结果里鞋是整块且没 bind_bone, 生成时整块落到 root 上 -> 腿动鞋不动。
    """
    try:
        import cv2
    except Exception:                                   # noqa: BLE001
        return []
    rgba = np.asarray(im.convert("RGBA"))
    a = (rgba[..., 3] > 24).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(a, 8)
    comps = [(i, float(cents[i][0])) for i in range(1, n)
             if int(stats[i, cv2.CC_STAT_AREA]) >= 40]
    if len(comps) < 2:
        return []
    comps.sort(key=lambda t: -int(stats[t[0], cv2.CC_STAT_AREA]))
    comps = sorted(comps[:2], key=lambda t: t[1])        # 按 x 排: 左 -> 右
    out = []
    for k, (lab, _cx) in enumerate(comps):
        x0 = int(stats[lab, cv2.CC_STAT_LEFT])
        y0 = int(stats[lab, cv2.CC_STAT_TOP])
        w0 = int(stats[lab, cv2.CC_STAT_WIDTH])
        h0 = int(stats[lab, cv2.CC_STAT_HEIGHT])
        sub = rgba.copy()
        sub[labels != lab] = 0
        out.append(("L" if k == 0 else "R",
                    Image.fromarray(sub[y0:y0 + h0, x0:x0 + w0]),
                    (x0, y0, w0, h0)))
    return out


def punch_head_wear(parts, img_dir, cw, ch):
    """把"头饰/头盔/头巾"的轮廓从头部各层里挖掉。

    为什么需要: slot 数组里 headwear 是最后一个(Spine 语义=最上层), 但龙骨这类编辑器
    导入时会按【骨骼】把 slot 重新排队 —— 头饰挂在 head 骨上就排到 b_face / b_head
    之前, 头盔被脸皮盖住(用户实测)。挖掉之后不管谁先画都能看见头盔; 头饰和头部各层
    都是同骨刚性运动, 不会因为挖洞露缝。只挖头部各层, 不碰躯干/手臂(骨头不同, 挖了
    转头会露洞)。
    """
    # 头饰可能被拆成多块(头盔 + 垂布), 挖洞要按【并集】算, 否则只挖了其中一块,
    # 另一块底下还留着面部像素, 编辑器里按骨骼重排后又会被盖住。
    srcs = []
    for q in parts:
        nm = str(q["name"]).lower()
        if q["cls"] in ("headgear", "cloth") and nm.startswith(
                ("head", "hat", "helmet", "cap", "tou", "mao")):
            srcs.append(q)
    if not srcs:
        return
    targets = [t for t in parts if t not in srcs and
               t["cls"] in ("face", "eye", "brow", "lash", "eye_closed")]
    if not targets:
        return
    # 头饰已经排在脸【下面】时不要挖 —— 那是"脸在头盔开口里"的角色(魔神这类:
    # 小脸 + 大盔壳, 原画里头盔四周包边、中间留出完整一张脸)。这种角色的 z 序
    # 本身就是对的, 挖洞只会把脸按头盔轮廓啃掉一圈, 只剩盔上镂空处那点残脸。
    # 只有头饰在上层(头盔包住头、脸只露一条缝)时才需要挖, 让下层头盔透出来。
    idx = {id(p): i for i, p in enumerate(parts)}
    if max(idx[id(s)] for s in srcs) < min(idx[id(t)] for t in targets):
        print(f"  头饰在脸下层(z 序已正确), 跳过挖洞")
        return
    mask = np.zeros((ch, cw), bool)
    for src in srcs:
        sim = np.array(Image.open(os.path.join(img_dir, src["image"])).convert("RGBA"))
        sh, sw = sim.shape[:2]
        y0, x0 = int(round(src["y0"])), int(round(src["x0"]))
        y1, x1 = min(ch, y0 + sh), min(cw, x0 + sw)
        if y1 <= y0 or x1 <= x0:
            continue
        mask[y0:y1, x0:x1] |= sim[:y1 - y0, :x1 - x0, 3] > 12
    if not mask.any():
        return

    total = 0
    for t in targets:
        fp = os.path.join(img_dir, t["image"])
        if not os.path.isfile(fp):
            continue
        im = np.array(Image.open(fp).convert("RGBA"))
        h, w = im.shape[:2]
        ty, tx = int(round(t["y0"])), int(round(t["x0"]))
        sy0, sx0 = max(0, ty), max(0, tx)
        sy1, sx1 = min(ch, ty + h), min(cw, tx + w)
        if sy1 <= sy0 or sx1 <= sx0:
            continue
        m = mask[sy0:sy1, sx0:sx1]
        sub = im[sy0 - ty:sy1 - ty, sx0 - tx:sx1 - tx]
        hit = m & (sub[..., 3] > 12)
        n = int(hit.sum())
        if n < 20:
            continue
        sub[..., 3] = np.where(hit, 0, sub[..., 3])
        Image.fromarray(im).save(fp)
        total += n
        print(f"    挖洞: {t['name']:12s} -{n}px")
    if total:
        print(f"  头饰挖洞: 从头部各层挖掉 {total}px "
              f"(编辑器按骨骼重排 slot 时头盔也不会被脸盖住)")


def _split_helmet_wear(im, px, py, head_box):
    """把"头盔 + 头巾连成一块"的头饰拆成 [刚性盔] + [可飘垂布]。

    在生成阶段再兜一次底: 界面上切换"头饰模式"会把部件重新合并成一整块,
    并给它 cls=cloth -> 整顶头盔跟着飘(用户实测)。这里不管上游怎么合并, 只按
    像素判断: 低饱和亮灰 = 盔(刚性), 头两侧垂下的高饱和布 = 可飘垂布。

    参数 head_box = (x0, x1, y0, h)；返回 (metal_rgba, m_box, cloth_rgba, c_box)
    或 None(不是"盔+布"的组合, 保持原样)。
    """
    import cv2
    a = np.asarray(im)[..., 3] > 12
    if not a.any():
        return None
    arr = np.asarray(im).astype(np.uint8)
    hsv = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(int)
    sat = hsv[..., 1].astype(int)
    hx0, hx1, hy0, hh = head_box
    hx0 -= px
    hx1 -= px
    y_lo = max(0, (hy0 - py) + int(hh * 0.18))
    xs = np.arange(arr.shape[1])[None, :]
    outside = (xs < hx0 - 2) | (xs > hx1 + 2)
    cloth = a & (h >= 8) & (h <= 60) & (sat >= 80) & outside
    cloth[:y_lo, :] = False
    if int(cloth.sum()) < 400:
        return None
    # 只在"整体以金属为主"时才拆(饱和度中位数 < 60): 纯布头巾(金黄 S≈145)不拆
    if float(np.median(sat[a])) >= 60:
        return None
    strong = a & (sat < 60) & (hsv[..., 2].astype(int) >= 110)
    if int(strong.sum()) < 1000:
        return None
    metal = a & ~cloth
    if int(metal.sum()) < 1500:
        return None

    def _cut(sel):
        ys, xs2 = np.where(sel)
        if len(xs2) == 0:
            return None
        x0, x1 = int(xs2.min()), int(xs2.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        out = arr[y0:y1, x0:x1].copy()
        out[..., 3] = np.where(sel[y0:y1, x0:x1], out[..., 3], 0)
        return out, (px + x0, py + y0, x1 - x0, y1 - y0)

    m = _cut(metal)
    c = _cut(cloth)
    if m is None or c is None:
        return None
    return m[0], m[1], c[0], c[1]


def load_parts(coords, parts_dir, img_dir, ch, square_mesh=False):
    """返回 parts 列表, 每项含 mesh + 顶点世界坐标(spine y上) + cls + 几何信息。"""
    parts = []

    # 头盔 + 头巾若被上游合并成一整块(cls 还是 cloth), 这里按像素拆开:
    # 盔 -> headwear(cls=headgear, 刚性跟头), 垂布 -> headscarf(cls=cloth, 走飘动链)。
    # 界面切换"头饰模式"时会把它们合回去, 只靠拆图阶段处理挡不住。
    # 已经拆成两块(有 headscarf)时, 金属那块必须按刚性处理 —— 界面上切换模式会
    # 用 set_head_wear 把所有 head* 部件统一改成 cloth, 于是"盔"又跟着布飘。
    if any(q["name"] == "headscarf" for q in coords["parts"]):
        for q in coords["parts"]:
            if q["name"] == "headwear":
                q["cls"] = "headgear"
            elif q["name"] == "headscarf":
                q["cls"] = "cloth"
            elif "head" in q["name"] and q.get("cls") in ("cloth", "headgear"):
                q["cls"] = "headgear"
        coords["_wear_split"] = True      # 让 main 把纠正后的 cls 落盘

    _hd = next((q for q in coords["parts"] if q["name"] == "head"), None)
    if _hd is not None and not any(q["name"] == "headscarf"
                                   for q in coords["parts"]):
        _box = (_hd["x"], _hd["x"] + _hd["width"], _hd["y"], _hd["height"])
        _out = []
        for q in coords["parts"]:
            if (q["name"] == "headwear"
                    and str(q.get("cls") or "") in ("cloth", "headgear")):
                _fp = os.path.join(parts_dir, q.get("image", "headwear.png"))
                # 拆分是不可逆的(会把 headwear.png 改写成"只有盔")。留一份原始合并图,
                # 之后每次都从它拆 —— 否则第二次生成时找不到布, 垂布就丢了。
                _full = os.path.join(parts_dir, "headwear_full.png")
                if not os.path.isfile(_full) and os.path.isfile(_fp):
                    shutil.copyfile(_fp, _full)
                _src = _full if os.path.isfile(_full) else _fp
                if os.path.isfile(_src):
                    _im = Image.open(_src).convert("RGBA")
                    _r = _split_helmet_wear(_im, q["x"], q["y"], _box)
                    if _r is not None:
                        _mr, _mb, _cr, _cb = _r
                        Image.fromarray(_mr).save(_fp)
                        for _d in (parts_dir, img_dir):
                            if os.path.isdir(_d):
                                Image.fromarray(_cr).save(
                                    os.path.join(_d, "headscarf.png"))
                        q.update({"x": _mb[0], "y": _mb[1],
                                  "width": _mb[2], "height": _mb[3],
                                  "cls": "headgear"})
                        _sp = dict(q)
                        _sp.update({"name": "headscarf", "image": "headscarf.png",
                                    "x": _cb[0], "y": _cb[1],
                                    "width": _cb[2], "height": _cb[3],
                                    "cls": "cloth"})
                        _sp.pop("bind_bone", None)
                        _out.append(_sp)                 # 先画布, 再压头盔
                        coords["_wear_split"] = True
                        print(f"  头饰再拆分: 盔 {int(_mb[2])}x{int(_mb[3])} 刚性"
                              f" + 垂布 {int(_cb[2])}x{int(_cb[3])} 可飘")
            _out.append(q)
        coords["parts"] = _out

    # 最后一道保险: 万一没法拆(缺合并底图等), 只要这块"以金属为主", 就按刚性处理 ——
    # 宁可垂布不飘, 也不能让整顶头盔跟着飘。
    for q in coords["parts"]:
        if q["name"] != "headwear":
            continue
        if str(q.get("cls") or "") == "headgear":
            continue
        _fp = os.path.join(parts_dir, q.get("image", "headwear.png"))
        if not os.path.isfile(_fp):
            continue
        try:
            import cv2 as _cv
            _arr = np.asarray(Image.open(_fp).convert("RGBA"))
            _m = _arr[..., 3] > 12
            if int(_m.sum()) < 200:
                continue
            _sat = _cv.cvtColor(_arr[..., :3], _cv.COLOR_RGB2HSV)[..., 1][_m]
            if float(np.median(_sat)) < 60:
                q["cls"] = "headgear"
                coords["_wear_split"] = True
                print(f"  头饰按刚性处理: headwear 以金属为主"
                      f"(S中位 {float(np.median(_sat)):.0f}) -> headgear")
        except Exception:                        # noqa: BLE001
            pass

    for p in coords["parts"]:
        name = p["name"]
        png = p.get("image", name + ".png")
        sp = os.path.join(parts_dir, png)
        im = Image.open(sp)
        iw, ih = im.size
        x, y = float(p["x"]), float(p["y"])
        w = float(p.get("width", iw))
        h = float(p.get("height", ih))
        gx, gy = p.get("grid", [8, 8])
        is_bird = bool(p.get("bird", False))

        # 满画布 PNG 真实 bbox 还原 + UV 重映射
        # _alpha_bbox 返回 PNG 内部像素坐标(相对 PNG 左上); 部件画布位置 = x,y。
        # 故还原后的画布坐标 = (x + bl, y + bt), 不能直接用 bl,bt 覆盖 x,y
        # (否则裁切过的 PSD 小图会丢失画布偏移 -> 部件飞到左上角)。
        uv_box = None
        bl = bt = 0
        sub_box = None        # (PNG内 l,t,r,b) 供形状网格切 alpha
        bb = _alpha_bbox(im)
        if bb is not None:
            bl, bt, br, bbm = bb
            tw, tht = br - bl, bbm - bt
            # 仅当 PNG 内主体明显小于标称(满画布/带噪) 才还原; 裁切小图(alpha
            # 几乎填满)不触发还原但仍可用 PNG 内 bbox 做形状网格切片。
            if tw < 0.92 * w or tht < 0.92 * h or (x == 0 and y == 0):
                x, y, w, h = x + float(bl), y + float(bt), float(tw), float(tht)
                uv_box = (bl / iw, bt / ih, br / iw, bbm / ih)
                sub_box = (bl, bt, br, bbm)
            else:
                sub_box = (bl, bt, br, bbm)
        # 兜底: 旧拆图里 footwear 是"两只鞋连在一张图", 且没有 bind_bone ——
        # 整块会绑到 root, 腿怎么迈鞋都钉在原地。这里就地拆成左右, 各 100% 绑脚骨。
        if name == "footwear" and not p.get("bind_bone") and not square_mesh:
            pair = _split_foot_pair(im)
            if pair:
                for side, sub_img, (sx0, sy0, sw, sh) in pair:
                    sub_name = f"footwear_{side}"
                    sub_img.save(os.path.join(img_dir, sub_name + ".png"))
                    bx = x + (sx0 - bl if uv_box else sx0)
                    by = y + (sy0 - bt if uv_box else sy0)
                    a_sub = np.asarray(sub_img.convert("RGBA"))[..., 3]
                    shp = build_shape_mesh(bx, by, sw, sh, int(gx), int(gy), a_sub)
                    if shp is not None:
                        sv, su, stri, shu, sed = shp
                    else:
                        sv, su, stri, shu, sed = build_grid_mesh(
                            bx, by, sw, sh, int(gx), int(gy))
                    v_vid = np.array(sv, float)
                    v_spine = np.column_stack([v_vid[:, 0], ch - v_vid[:, 1]])
                    parts.append(dict(
                        name=sub_name, image=sub_name + ".png",
                        attachment=sub_name, uvs=su,
                        triangles=stri, hull=shu, edges=sed,
                        v_vid=v_vid, v_spine=v_spine, cls="foot", bird=False,
                        follow=None, bind_bone=f"leg_{side}_foot",
                        iw=int(sw), ih=int(sh),
                        x0=float(v_vid[:, 0].min()), x1=float(v_vid[:, 0].max()),
                        y0=float(v_vid[:, 1].min()), y1=float(v_vid[:, 1].max()),
                        cx=float(v_vid[:, 0].mean()), cy=float(v_vid[:, 1].mean())))
                print(f"  鞋自动拆分: footwear -> L/R (各绑 leg_?_foot)")
                continue

        # 形状自适应网格: 用 PNG 内 bbox 区 alpha 沿轮廓布点, 透明角不被拉伸;
        # 失败(无 cv2/scipy 或形状异常)回退正方形规则网格。
        shape = None
        if not square_mesh and sub_box is not None:
            a_full = np.asarray(im.convert("RGBA"))[:, :, 3]
            l_, t_, r_, b_ = sub_box
            sub = a_full[max(0, t_):max(1, b_), max(0, l_):max(1, r_)]
            if sub.size:
                shape = build_shape_mesh(x, y, w, h, gx, gy, sub)
        if shape is not None:
            verts, uvs, tris, hull, edges = shape
        else:
            verts, uvs, tris, hull, edges = build_grid_mesh(x, y, w, h, gx, gy)
        if uv_box is not None:
            u0, v0u, u1, v1u = uv_box
            for k in range(0, len(uvs), 2):
                uvs[k] = u0 + uvs[k] * (u1 - u0)
                uvs[k + 1] = v0u + uvs[k + 1] * (v1u - v0u)

        cls = p.get("cls") or classify(name, is_bird)
        if is_bird:
            cls = "bird"
        shutil.copy(sp, os.path.join(img_dir, png))

        v_vid = np.array(verts, float)                 # 视频像素(y下)
        v_spine = np.column_stack([v_vid[:, 0], ch - v_vid[:, 1]])  # y上
        parts.append(dict(
            name=name, image=png, attachment=os.path.splitext(png)[0],
            uvs=uvs, triangles=tris, hull=hull, edges=edges,
            v_vid=v_vid, v_spine=v_spine, cls=cls, bird=is_bird,
            follow=p.get("follow_part"),
            bind_bone=p.get("bind_bone"),
            iw=iw, ih=ih,
            x0=float(v_vid[:, 0].min()), x1=float(v_vid[:, 0].max()),
            y0=float(v_vid[:, 1].min()), y1=float(v_vid[:, 1].max()),
            cx=float(v_vid[:, 0].mean()), cy=float(v_vid[:, 1].mean())))
    punch_head_wear(parts, img_dir,
                    int((coords.get("canvas") or {}).get("w", 336)),
                    int((coords.get("canvas") or {}).get("h", 336)))
    return parts


# ----------------------------------------------------------------------------
# 手标骨链加载 (bones.json, 像素坐标 -> spine 世界坐标)
# ----------------------------------------------------------------------------
def load_bone_markers(bones_json, coords_path, ch):
    """读手标骨链。格式见 IDLE_ALGORITHM.md 8.7。
    返回 dict: {'arm':{'L':[pts],'R':[pts]}, 'hair':[[pts],...],
               'ribbon':[[pts],...], 'spine':[pts]}。点已转 spine 坐标(y上)。
    任意缺失项 -> 调用方退回几何自动算法。文件不存在 -> 返回 {}。"""
    path = bones_json.strip()
    if not path:
        cand = os.path.join(os.path.dirname(os.path.abspath(coords_path)),
                            "bones.json")
        path = cand if os.path.isfile(cand) else ""
    if not path or not os.path.isfile(path):
        return {}
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as ex:                            # noqa: BLE001
        print(f"  bones.json 解析失败, 退回自动算法: {ex}")
        return {}

    def to_spine(pts):
        # 输入像素 [[px,py],...] (y下) -> spine [[x, ch-py],...] (y上)
        out = []
        for q in pts:
            out.append([float(q[0]), float(ch - q[1])])
        return np.array(out, np.float32) if out else None

    res = {"arm": {}, "hair": [], "ribbon": [], "spine": None,
           "chain_defs": []}
    for ci, ch_def in enumerate(data.get("chains", [])):
        typ = ch_def.get("type", "")
        pts = ch_def.get("points", [])
        if len(pts) < 2:
            res["chain_defs"].append(None)     # 占位保持索引对齐
            continue
        sp = to_spine(pts)
        # 父子延伸: parent=父链在 chains 数组的索引(-1=独立挂默认锚),
        # parent_pt=父链上的节点索引(默认末点 -1)。
        res["chain_defs"].append(dict(
            type=typ, side=ch_def.get("side", "L").upper(),
            pts=sp, parent=int(ch_def.get("parent", -1)),
            parent_pt=int(ch_def.get("parent_pt", -1))))
        if typ == "arm":
            side = ch_def.get("side", "L").upper()
            res["arm"][side] = sp
        elif typ == "hair":
            res["hair"].append(sp)
        elif typ == "ribbon":
            res["ribbon"].append(sp)
        elif typ in ("spine", "torso"):
            res["spine"] = sp
    n = len(res["arm"]) + len(res["hair"]) + len(res["ribbon"]) + \
        (1 if res["spine"] is not None else 0)
    if n:
        print(f"  手标骨链已加载: arm={list(res['arm'])} "
              f"hair={len(res['hair'])} ribbon={len(res['ribbon'])} "
              f"spine={'Y' if res['spine'] is not None else 'N'}")
    return res


# ----------------------------------------------------------------------------
# 骨架构建 (按分类与几何自动放置)
# ----------------------------------------------------------------------------
def build_skeleton(parts, cw, ch, args):
    """自动放置骨骼, 返回 (Skeleton, info dict)。坐标全用 spine 世界(y 上)。"""
    sk = Skeleton()

    def grp(c):
        return [p for p in parts if p["cls"] == c]

    body = grp("body")
    faces = grp("face")
    eyes = grp("eye")
    brows = grp("brow")
    lashes = grp("lash")
    blink_eyes = grp("blink_eye")      # 闭眼图层(有则切换, 无则挤压)
    closed_eyes = grp("eye_closed")    # 合成出来的闭眼补丁(倒地/昏迷时盖住眼睛)
    hairs = grp("hair")
    ribbons = grp("ribbon")
    cloths = grp("cloth")          # 头巾/披巾: 一头钉在头上, 垂下的部分会飘
    arms_up = grp("arm_upper")
    arms_fore = grp("arm_fore")
    arms_hand = grp("arm_hand")

    # 全身 bbox (spine 世界)
    allv = np.vstack([p["v_spine"] for p in parts if not p["bird"]])
    bx0, bx1 = allv[:, 0].min(), allv[:, 0].max()
    by0, by1 = allv[:, 1].min(), allv[:, 1].max()    # by0=底(y小) by1=顶(y大)
    cx = (bx0 + bx1) / 2

    # 头部基准: face/eye/brow/lash 的 bbox
    head_src = faces + eyes + brows + lashes
    if head_src:
        hv = np.vstack([p["v_spine"] for p in head_src])
        head_cx = float((hv[:, 0].min() + hv[:, 0].max()) / 2)
        head_topY = float(hv[:, 1].max())            # 头顶(y大)
        head_botY = float(hv[:, 1].min())            # 下巴(y小)
        head_half = max(1.0, (hv[:, 0].max() - hv[:, 0].min()) / 2)
    else:
        head_cx = cx
        head_topY = by1
        head_botY = by0 + 0.6 * (by1 - by0)
        head_half = (bx1 - bx0) / 2
    head_h = max(1.0, head_topY - head_botY)
    neck_y = head_botY

    # 手标骨链(缺失项退回几何自动); 各部位按需取用
    markers = getattr(args, "_markers", {}) or {}
    chain_defs = markers.get("chain_defs") or []

    # ---- 选脊椎主干: 在所有 spine 类型链里挑最像"躯干中轴"的一条 ----
    # 其余 spine 链 + 所有其它链一律当【通用骨链】(纯骨骼), 按区域绑部件,
    # 运动效果不看链类型, 改看绑到该链的部件 cls (见下方区域绑定)。
    # 故用户即便把所有链都标成 spine, 也只有 1 条成主干, 余者照样生成骨。
    trunk_ci = -1
    spine_cands = [ci for ci, cd in enumerate(chain_defs)
                   if cd is not None and cd["type"] in ("spine", "torso")
                   and cd["pts"] is not None and len(cd["pts"]) >= 2]

    def _trunk_score(cd):
        p = cd["pts"]                                  # 根->梢(y上, 根 y小)
        span = float(p[:, 1].max() - p[:, 1].min())    # 竖向跨度大=像主干
        root = p[0]
        return span - abs(float(root[0]) - cx) * 0.6 - (float(root[1]) - by0) * 0.4
    if spine_cands:
        trunk_ci = max(spine_cands, key=lambda ci: _trunk_score(chain_defs[ci]))

    # torso 骨: 放在身体中部偏下(呼吸锚点); root 在脚底(y=by0)
    if trunk_ci >= 0:
        sp = chain_defs[trunk_ci]["pts"]               # 根->梢(y上故梢y大)
        torso_pt, head_pt = sp[0], sp[-1]
        neck_pt = sp[len(sp) // 2] if len(sp) >= 3 else (sp[0] + sp[-1]) / 2
        torso_y = float(torso_pt[1])
        torso = sk.add("torso", sk.root, float(torso_pt[0]), torso_y)
        neck = sk.add("neck", torso, float(neck_pt[0]), float(neck_pt[1]))
        head = sk.add("head", neck, float(head_pt[0]), float(head_pt[1]))
    else:
        torso_y = by0 + 0.45 * (by1 - by0)
        torso = sk.add("torso", sk.root, cx, torso_y)
        neck = sk.add("neck", torso, head_cx, neck_y)
        head = sk.add("head", neck, head_cx, head_botY + 0.35 * head_h)

    # 五官骨: 各自挂 head, 放在部件中心 (含闭眼层, 随头转)
    feat_bones = {}
    for p in eyes + brows + lashes + faces + blink_eyes + closed_eyes:
        bn = "b_" + p["name"]
        c = p["v_spine"].mean(axis=0)
        feat_bones[p["name"]] = sk.add(bn, head, float(c[0]), float(c[1]))

    # ---- 2.5D 转面: head_ctrl 控制骨 + 五官视差 transform 约束 (Spine 原生) ----
    # 原理: head_ctrl(挂 head)动画里左右平移(转面信号); 各五官骨建 transform
    # constraint(target=head_ctrl, relative+local), translateMix 按"凸出深度"
    # 设值 -> 鼻最凸 mix 最大、眼最平 mix~0。控制骨横移时五官按深度产生视差,
    # 即 Spine 标准 2.5D 转面, 且可在编辑器里调每个约束的 mix 微调立体感。
    # 注意: head_ctrl 世界坐标必须 == head(setup local translate=0)! 否则
    # Spine 原生 relative 约束在 setup pose 就把 head_ctrl 的非零 local translate
    # ×mix 叠加到五官 -> 脸在 setup 即错位(GIF 预览只用动画增量看不出, Spine 会暴露)。
    head_ctrl = sk.add("head_ctrl", head, head.wx, head.wy)
    info_turn = []                                     # [(feat_bone, depth)]

    def feat_depth(p):
        n = p["name"].lower()
        cls = p["cls"]
        # 深度系数(越凸视差越大): 鼻>眉>嘴>脸>眼。眼≈0(随脸转不自滑)
        if cls in ("eye", "blink_eye", "lash"):
            return 0.04
        if cls == "brow":
            return 0.55
        # face 内细分: 嘴有一点视差
        if any(k in n for k in ("mouth", "zui", "lip", "kou")):
            return 0.35
        # 鼻子/耳朵: 是"长在脸上"的, 必须跟着头刚性走 —— 给视差会让它们
        # 在转头时相对脸滑动, 看起来像自己飘。深度 0 = 不参与 2.5D 视差。
        if any(k in n for k in ("nose", "bizi", "bi_", "_bi", "ear", "erduo", "er_")):
            return 0.0
        return 0.18                                    # 脸/头基面
    for p in eyes + brows + lashes + faces + blink_eyes + closed_eyes:
        b = feat_bones[p["name"]]
        d = feat_depth(p)
        if abs(d) > 1e-3:
            info_turn.append((b, d))

    # ---- 手标链父子延伸: 统一预建所有手标链骨 (拓扑序解析 parent) ----
    # 每条链可声明 parent=另一条链索引, parent_pt=父链节点序号 -> 子链根骨
    # 父级 = 父链对应骨 (而非默认 head/torso), 实现任意链接任意链。
    # 注: chain_defs 已在上方读取; 主干链(trunk_ci)不在此建骨(由 torso/neck/head 承担)。
    chain_bones = [None] * len(chain_defs)     # ci -> [Bone,...]

    def _default_anchor(typ):
        return head if typ in ("hair",) else torso

    def _build_chain(ci, stack):
        if chain_bones[ci] is not None:
            return chain_bones[ci]
        cd = chain_defs[ci]
        if cd is None or ci == trunk_ci or cd["pts"] is None \
                or len(cd["pts"]) < 2:
            # 主干由 torso/neck/head 单独处理, 不另建骨; 被引用为父则子链锚到 head。
            chain_bones[ci] = []
            return []
        if ci in stack:                        # 环引用保护
            anchor = _default_anchor(cd["type"])
        else:
            par = cd["parent"]
            if 0 <= par < len(chain_defs) and chain_defs[par] is not None \
                    and par != trunk_ci:
                pbones = _build_chain(par, stack | {ci})
                if pbones:
                    ppt = cd["parent_pt"]
                    ppt = ppt if 0 <= ppt < len(pbones) else len(pbones) - 1
                    anchor = pbones[ppt]
                else:
                    anchor = _default_anchor(cd["type"])
            elif par == trunk_ci:              # 父级是主干 -> 锚到 head/neck/torso
                anchor = head
            else:
                anchor = _default_anchor(cd["type"])
        bones = []
        prev = anchor
        pref = cd["type"] + (("_" + cd["side"]) if cd["type"] == "arm" else "")
        for s, q in enumerate(cd["pts"]):
            b = sk.add(f"mk{ci}_{pref}_{s}", prev, float(q[0]), float(q[1]))
            bones.append(b)
            prev = b
        chain_bones[ci] = bones
        return bones

    for _ci in range(len(chain_defs)):
        _build_chain(_ci, set())

    # ---- 通用骨链池: 除主干外所有链都建骨, 当【纯骨骼】(不看链类型) ----
    # 部件按【空间区域】就近绑到某条链 -> 同区域多层(含背面不显示层)共用一条
    # 链, 运动一致; 运动效果只看部件 cls(hair/ribbon=风吹, arm=FK), 不看链类型。
    # 故用户把所有链标成 spine 也照常生成: 只 1 条成主干, 余者皆通用链。
    generic_chains = []     # [{'pts':Nx2, 'bones':[Bone,...]}]
    for ci, cd in enumerate(chain_defs):
        if cd is None or ci == trunk_ci or not chain_bones[ci]:
            continue
        generic_chains.append(dict(pts=cd["pts"], bones=chain_bones[ci]))

    def fit_cost(pts, part):
        pv = part["v_spine"]                           # (N,2)
        tot = 0.0                                      # 折线点到顶点云最近距离均值
        for q in pts:
            d2 = np.min(np.sum((pv - q) ** 2, axis=1))
            tot += math.sqrt(d2)
        return tot / max(1, len(pts))

    def part_size(p):
        v = p["v_spine"]
        return float(max(np.ptp(v[:, 0]), np.ptp(v[:, 1]), 1.0))

    def nearest_chain(p):
        """返回 (best_chain or None, cost)。"""
        best, bc = None, 1e18
        for gc in generic_chains:
            c = fit_cost(gc["pts"], p)
            if c < bc:
                bc, best = c, gc
        return best, bc

    # 头发/飘带: 每片一条多段骨链 (参考官方 luna: 3-4 段, 逐节摆动递增更柔软)
    wind_chains = []        # [(part, [b0, b1, ...], 'hair'/'ribbon')]

    def build_wind_chain_auto(p, tag, idx):
        v = p["v_spine"]
        topY = float(v[:, 1].max())
        botY = float(v[:, 1].min())
        length = max(1.0, topY - botY)
        n_seg = int(np.clip(round(length / 90.0), 2, 4))
        anchor = head if tag == "hair" else torso
        chain = []
        prev = anchor
        for s in range(n_seg):
            yy = topY - (s / n_seg) * length
            band = np.abs(v[:, 1] - yy) < (length / n_seg)
            bx = float(v[band, 0].mean()) if band.any() else \
                float(v[v[:, 1].argmax(), 0])
            b = sk.add(f"auto_{tag}{idx}_{s}", prev, bx, yy)
            chain.append(b)
            prev = b
        return chain

    # 区域就近绑链(可多片共用同链); 该区域没标链(最近链都太远) -> 自动几何补
    n_auto_wind = 0
    for grp_parts, tag in ((hairs + cloths, "hair"), (ribbons, "ribbon")):
        for idx, p in enumerate(grp_parts):
            gc, cost = nearest_chain(p)
            thr = 0.6 * part_size(p)               # 链离这片太远 = 该区无标链
            if gc is not None and cost <= thr:
                chain = gc["bones"]                # 共用该区域链(含父子延伸)
            else:
                chain = build_wind_chain_auto(p, tag, idx)   # 自动补缺
                n_auto_wind += 1
            wind_chains.append((p, chain, tag))

    # 手臂 FK: 区域就近绑链(部件投影到链最近段); 缺失退几何自动(肩/肘/腕猜测)
    arm_chains = []

    def arm_side(side_up, side_fore, side_hand, label):
        seq = []                                       # 有序 (part, joint_name)
        for fp in side_up[:1]:
            seq.append((fp, f"arm_{label}_up"))
        for fp in side_fore[:1]:
            seq.append((fp, f"arm_{label}_fore"))
        for fp in side_hand[:1]:
            seq.append((fp, f"arm_{label}_hand"))
        if not seq:
            return
        chain = []
        prev = torso
        prev_v = None          # 上一部件顶点
        for (fp, jname) in seq:
            v = fp["v_spine"]
            if prev_v is None:
                # 肩点 = 部件【上沿 28%】里最靠身体中线的那一点。
                # 旧写法是"离 (中线, 躯干中部) 最近的顶点", 躯干中部在大臂的下半,
                # 于是肩点被放到肘上, 后面算肘/腕用的参照点全跟着错位 ——
                # 实测肩骨落在肘(画布 y151)、肘骨跑到肩上方(y105), 大小臂绕错的
                # 支点转, 看着就是"手臂上下两截没绑在一起"。
                _y1, _y0 = float(v[:, 1].max()), float(v[:, 1].min())
                band = v[v[:, 1] >= _y1 - 0.28 * max(1.0, _y1 - _y0)]
                joint = band[np.argmin(np.abs(band[:, 0] - cx))]
            else:
                # 肘 / 腕 = 两块部件顶点云里【最近的一对点】的中点 = 真实的接缝。
                # 旧写法拿"离肩最远的点"和"离肩最近的点"取中点, 部件一重叠
                # (护腕套在袖子上)就会把接缝算到大臂中间去。
                joint = _closest_pair_joint(prev_v, v)
            b = sk.add(jname, prev, float(joint[0]), float(joint[1]))
            chain.append((fp, b))
            prev = b
            prev_v = v
        arm_chains.append((label, chain))

    arm_parts = arms_up + arms_fore + arms_hand
    chain_arm = {}                                     # id(gc) -> (gc, [parts])
    chain_arm_order = []
    unassigned_arm = []
    for p in arm_parts:
        gc, cost = nearest_chain(p)
        thr = 0.7 * part_size(p)
        if gc is not None and cost <= thr:
            key = id(gc)
            if key not in chain_arm:
                chain_arm[key] = (gc, [])
                chain_arm_order.append(key)
            chain_arm[key][1].append(p)
        else:
            unassigned_arm.append(p)
    # 已绑链的手臂部件: 各自投影到链最近段, 绑该段骨
    for key in chain_arm_order:
        gc, plist = chain_arm[key]
        pts, bones = gc["pts"], gc["bones"]
        label = "L" if float(pts[:, 0].mean()) < cx else "R"
        chain = []
        for fp in plist:
            c = fp["v_spine"].mean(axis=0)
            best, bd = 0, 1e18
            for k in range(len(pts) - 1):
                a, bpt = pts[k], pts[k + 1]
                ab = bpt - a
                t = np.clip(np.dot(c - a, ab) / max(1e-6, np.dot(ab, ab)), 0, 1)
                proj = a + t * ab
                dd = float(np.sum((c - proj) ** 2))
                if dd < bd:
                    bd, best = dd, k
            chain.append((fp, bones[min(best, len(bones) - 1)]))
        arm_chains.append((label, chain))
    # 未绑链的手臂: 按 L/R 几何自动放骨
    if unassigned_arm:
        def split_lr(lst):
            return ([p for p in lst if p["cx"] < cx],
                    [p for p in lst if p["cx"] >= cx])
        ua_up = [p for p in unassigned_arm if p["cls"] == "arm_upper"]
        ua_fo = [p for p in unassigned_arm if p["cls"] == "arm_fore"]
        ua_ha = [p for p in unassigned_arm if p["cls"] == "arm_hand"]
        upL, upR = split_lr(ua_up)
        foL, foR = split_lr(ua_fo)
        haL, haR = split_lr(ua_ha)
        if upL or foL or haL:
            arm_side(upL, foL, haL, "L")
        if upR or foR or haR:
            arm_side(upR, foR, haR, "R")

    # ---- 道具(武器/盾)跟随手: coords 里 follow_part 指向某只小臂/手 ----
    # 道具**不共用**小臂那根骨, 而是挂一根独立子骨:
    #   ① 共用骨的话, 手臂靠 scaleY 缩短时刀会被一起压扁(8% 缩短 = 136px 的刀短
    #      11px, 一动一缩像橡皮);
    #   ② 挂到【手骨】位置上而不是小臂位置 —— 刀握在手里, 得跟着拳头走。上一版
    #      把它放在肘部, 结果手随缩短往上收、刀还留在肘那儿, 看着就是"武器没跟手"。
    # 位置放在手骨(wx,wy)、父骨仍是小臂, 是为了继承小臂的旋转与位移; 缩放由
    # 动画里写字骨 scale = 1/父骨链缩放 抵消。
    for p in [x for x in parts if x.get("follow")]:
        target = p["follow"]
        for (label, chain) in arm_chains:
            hit = next((bd for (pp, bd) in chain if pp["name"] == target), None)
            if hit is None:
                continue
            grip = sk.get(f"arm_{label}_hand") or hit     # 握点: 优先挂在手骨处
            pb = sk.add(f"prop_{p['name']}", hit, grip.wx, grip.wy)
            chain.append((p, pb))       # 权重按 setup 世界坐标算, 位置不变形
            print(f"  道具 {p['name']} 跟随 {target}（独立骨, 挂在 {grip.name} 处, 不随手臂缩放）")
            break
        else:
            print(f"  道具 {p['name']}: 找不到 {target}, 保持贴躯干")

    # ---- 腿: 每条腿 髋 -> 膝 -> 踝 三节骨, 髋挂 torso 下 ----
    # 一条腿是一整块网格, 沿高度做权重渐变就能在膝盖处平滑弯, 不需要切片。
    legs = grp("leg")
    feet = grp("foot")                 # 脚片单独绑踝骨(cls=foot), 但**算腿长要带上**
    leg_chains = []                    # [(label, [hip,knee,foot], topY, botY, parts)]
    for label, lst in (("L", [p for p in legs if p["cx"] < cx]),
                       ("R", [p for p in legs if p["cx"] >= cx])):
        if not lst:
            continue
        # 腿的上下沿必须算到【脚底】。切出 foot 之后小腿片只到踝, 只按 cls=="leg"
        # 量范围的话腿长凭空少了 16px, 踝骨跟着往上跑 20px(实测 283 -> 263) ——
        # 脚片离踝骨 20px 远, 转起来是"被一根悬空的骨头甩着", 不是绕脚踝转。
        span = lst + [f for f in feet
                      if (f["cx"] < cx) == (label == "L")]
        v = np.vstack([p["v_spine"] for p in span])
        topY, botY = float(v[:, 1].max()), float(v[:, 1].min())
        hipx = float(np.vstack([p["v_spine"] for p in lst])[:, 0].mean())
        hgt = max(1.0, topY - botY)
        hip = sk.add(f"leg_{label}_up", torso, hipx, topY)
        knee = sk.add(f"leg_{label}_fore", hip, hipx, topY - 0.45 * hgt)
        foot = sk.add(f"leg_{label}_foot", knee, hipx, topY - 0.88 * hgt)
        leg_chains.append((label, [hip, knee, foot], topY, botY, lst))
    if leg_chains:
        print(f"  腿骨: {len(leg_chains)} 条 × 3 节 (髋/膝/踝)")

    # 睁眼<->闭眼 配对 (按 x 中心就近): 有闭眼层则动画用切换, 否则挤压
    blink_pairs = []        # [(open_part, close_part)]
    if blink_eyes:
        remaining = list(blink_eyes)
        for oe in eyes:
            if not remaining:
                break
            be = min(remaining, key=lambda b: abs(b["cx"] - oe["cx"]))
            remaining.remove(be)
            blink_pairs.append((oe, be))

    # ---- 建 2.5D 转面 transform 约束 (turn 强度缩放每个约束的 translateMix) ----
    # turn=0 -> 全 mix=0 不转面; turn=1 -> 较大视差; 0.5 默认微转。
    turn = float(np.clip(getattr(args, "head_turn", 0.5), 0.0, 1.0))
    for b, depth in info_turn:
        mix = depth * turn * 0.5                       # 0.5: 基准缩放(turn=1时鼻 mix=0.5)
        sk.add_transform(f"turn_{b.name}", [b.name], "head_ctrl",
                         translate_mix=mix, rotate_mix=0.0,
                         local=True, relative=True)

    eye_closed_bone = (feat_bones.get(closed_eyes[0]["name"])
                       if closed_eyes else None)
    # 不写骨骼的 setup 缩放: 回到"闭眼补丁"最初的行为(补丁只在动画里被缩放),
    # 加了 setup=0.001 会让某些编辑器/预览在未播动画时看不到补丁。

    info = dict(
        torso=torso, neck=neck, head=head, feat_bones=feat_bones,
        eye_closed_bone=eye_closed_bone,
        head_ctrl=head_ctrl, info_turn=info_turn, turn=turn,
        wind_chains=wind_chains, arm_chains=arm_chains, leg_chains=leg_chains,
        head_cx=head_cx, head_topY=head_topY, head_botY=head_botY,
        head_h=head_h, head_half=head_half, neck_y=neck_y,
        by0=by0, by1=by1, torso_y=torso_y, cx=cx,
        eyes=eyes, brows=brows, lashes=lashes, faces=faces,
        blink_eyes=blink_eyes, blink_pairs=blink_pairs,
        hairs=hairs, ribbons=ribbons,
        n_generic=len(generic_chains), n_auto_wind=n_auto_wind,
        trunk_marked=(trunk_ci >= 0))
    return sk, info


# ----------------------------------------------------------------------------
# 蒙皮权重: 每部件顶点 -> (骨, 权重)
# ----------------------------------------------------------------------------
def assign_weights(parts, sk, info, args):
    """为每个部件计算 bone_weights: [[(bone,w),...], ...] (与 v_spine 对齐)。"""
    torso = info["torso"]
    neck = info["neck"]
    head = info["head"]
    by0, by1 = info["by0"], info["by1"]
    head_botY, head_h = info["head_botY"], info["head_h"]
    root = sk.root
    lock_h = max(1.0, args.foot_lock * (by1 - by0))   # 自底向上锁定过渡带

    for p in parts:
        v = p["v_spine"]
        cls = p["cls"]
        bw = []
        if p["bird"]:
            for _ in v:
                bw.append([(root, 1.0)])
            p["bone_weights"] = bw
            continue
        # coords 里显式指定绑定骨(如切开的大腿/小腿): 100% 单骨, 不做权重渐变。
        # 这样刚性蒙皮的编辑器也能靠"两片分别跟髋/膝"实现屈膝。
        _bb = p.get("bind_bone")
        if _bb:
            _b = sk.get(_bb)
            if _b is not None:
                for _ in v:
                    bw.append([(_b, 1.0)])
                p["bone_weights"] = bw
                continue

        if cls == "body":
            # 三段加权(参考官方脊椎链): 脚底锁 root -> 躯干随 torso ->
            # 贴脖上沿向 neck 混合, 使脖子区随头转而不脱节。
            neck_y = info["neck_y"]
            torso_y = info["torso_y"]
            # 上半身向 neck 过渡带: 从胸口(torso_y)到脖(neck_y)
            up_h = max(1.0, neck_y - torso_y)
            for (sx, sy) in v:
                t = smoothstep((sy - by0) / lock_h)    # 0=脚底(锁) 1=带顶(自由)
                # 贴脖系数: 顶点越接近/超过脖高, 越偏 neck
                nu = smoothstep(np.clip((sy - torso_y) / up_h, 0.0, 1.0))
                if t <= EPS:
                    bw.append([(root, 1.0)])
                elif nu > EPS:
                    # torso 与 neck 混合 (上沿贴脖随头转)
                    wn = nu * 0.7                       # 最高 70% 跟 neck
                    bw.append([(torso, 1 - wn), (neck, wn)])
                elif t >= 1 - EPS:
                    bw.append([(torso, 1.0)])
                else:
                    bw.append([(root, 1 - t), (torso, t)])
        elif cls == "eye_closed":
            # 闭眼补丁: 100% 挂在它自己的骨上(缩放它来决定露不露)
            fb = info.get("eye_closed_bone") or head
            for _ in v:
                bw.append([(fb, 1.0)])
        elif cls in ("eye", "lash", "blink_eye"):
            # 眼/睫毛/闭眼层: 强制单骨绑定(feat 骨或 head), 随头转;
            # 睁眼眨眼走 deform 挤压(无闭眼层时)或 attachment 切换(有闭眼层)
            fb = info["feat_bones"].get(p["name"], head)
            for _ in v:
                bw.append([(fb, 1.0)])
        elif cls == "leg":
            lc = next((c for c in info.get("leg_chains", [])
                       if any(pp is p for pp in c[4])), None)
            if lc is None:
                for _ in v:
                    bw.append([(torso, 1.0)])
            else:
                _lab, (b_hip, b_knee, b_foot), topY, botY, _ps = lc
                hgt = max(1.0, topY - botY)
                for (_sx, _sy) in v:
                    u = (topY - _sy) / hgt          # 0=髋 1=脚
                    if u <= 0.45:                   # 大腿段: 髋 -> 膝
                        t = smoothstep(np.clip(u / 0.45, 0.0, 1.0))
                        bw.append([(b_hip, 1 - t), (b_knee, t)])
                    else:                           # 小腿段: 膝 -> 踝
                        t = smoothstep(np.clip((u - 0.45) / 0.50, 0.0, 1.0))
                        bw.append([(b_knee, 1 - t), (b_foot, t)])
        elif cls == "headgear":
            # 头盔/头巾(无头发角色): 整块 100% 跟"头部画面"那根骨(b_head), 不是 head。
            # 1) 不做脸/颈混合, 否则垂下来的布会被 neck 拉走、转头时整片被扯歪;
            # 2) 龙骨这类编辑器导入时会按【骨骼】给 slot 重新排队, 挂在 head(第 3 根)
            #    上会排到 b_face / b_head 之前 -> 头盔被脸皮盖住。挂 b_head 就排在它们后面。
            fb = info.get("feat_bones", {}).get("head") or head
            for _v in v:
                bw.append([(fb, 1.0)])
        elif cls == "brow" or cls == "face":
            # 眉/脸: 绑各自特征骨(继承 head 转); 下半脸(贴脖)向 neck 渐变,
            # 与 body 脖区的 neck 权重衔接 -> 整条下巴-脖-胸连续转, 不脱节。
            fb = info["feat_bones"].get(p["name"], head)
            # 分层经常把**整个上半身**都塞进 head 件(胸甲/肚子一起)。那些像素如果
            # 100% 绑头骨, 头一转胸甲就跟着扭。这里按高度把它们交还给躯干:
            # 从脖子往下 body_h 的 12% 之内, 权重从 0 平滑过渡到 100% torso。
            band = max(1.0, 0.12 * (by1 - by0))
            for (sx, sy) in v:
                u = (sy - head_botY) / head_h          # 0=下巴 1=头顶
                # 从 u=0.22 开始向下渐增 neck 权重, 下巴以下满混
                w_neck = smoothstep(np.clip((0.22 - u) / 0.42, 0.0, 1.0)) * 0.6
                # 下巴以下(衣领/脖影)几乎整块交给 neck: 头转的时候这圈基本不动,
                # 网格在中间拉伸过渡, 不会翻出一条硬缝(切口式裁头的最大毛病)。
                if u < 0:
                    w_neck = max(w_neck, 0.6 + 0.4 * smoothstep(np.clip(-u / 0.35, 0.0, 1.0)))
                down = smoothstep(np.clip((neck_y - sy) / band, 0.0, 1.0))
                if down > EPS:
                    rest = 1.0 - down
                    if w_neck > EPS:
                        bw.append([(torso, down), (fb, rest * (1 - w_neck)),
                                   (neck, rest * w_neck)])
                    else:
                        bw.append([(torso, down), (fb, rest)])
                elif w_neck > EPS:
                    bw.append([(fb, 1 - w_neck), (neck, w_neck)])
                else:
                    bw.append([(fb, 1.0)])
        elif cls == "cloth":
            # 头巾: 权重按【离头部多远】分, 不按高度分。
            # 包在头上那圈(离头近)必须 100% 跟头 —— 按高度分会把下半圈算进"会飘"的
            # 骨, 一转就把贴着头的那圈布带走, 看着像脸在动。
            chain = None
            for (pp, bones, tag) in info["wind_chains"]:
                if pp is p:
                    chain = bones
                    break
            chain = chain or []
            hx0 = info["head_cx"] - info["head_half"]
            hx1 = info["head_cx"] + info["head_half"]
            hy0, hy1 = info["head_botY"], info["head_topY"]
            m0 = 0.16 * info["head_h"]          # 这个距离以内 = 贴着头, 完全不动
            m1 = 0.78 * info["head_h"]          # 这个距离以外 = 完全交给摆动的骨
            nseg = len(chain)
            for (sx, sy) in v:
                dx = max(hx0 - sx, 0.0, sx - hx1)
                dy = max(hy0 - sy, 0.0, sy - hy1)
                d = math.hypot(dx, dy)
                t = smoothstep(np.clip((d - m0) / max(1e-3, m1 - m0), 0.0, 1.0))
                if nseg < 2 or t <= EPS:
                    bw.append([((chain[0] if chain else head), 1.0)])
                    continue
                # 沿整条链平滑分布(以前只有首/末两根骨): 只给首末两骨时,
                # 中间那段布要么完全钉死要么整块跟着末骨甩, 看着像"整条头巾在晃"。
                fpos = t * (nseg - 1)
                i0 = int(math.floor(fpos))
                i1 = min(i0 + 1, nseg - 1)
                f = fpos - i0
                if i0 == i1 or f <= EPS:
                    bw.append([(chain[i0], 1.0)])
                else:
                    bw.append([(chain[i0], 1 - f), (chain[i1], f)])
        elif cls in ("hair", "ribbon"):
            chain = None
            for (pp, bones, tag) in info["wind_chains"]:
                if pp is p:
                    chain = bones
                    break
            topY = p["y1_spine"] = float(v[:, 1].max())
            botY = float(v[:, 1].min())
            span = max(1.0, topY - botY)
            lock = args.root_lock
            nseg = len(chain)
            for (sx, sy) in v:
                u = (topY - sy) / span                 # 0=发根 1=发梢
                # root-lock 段锁第一骨; 其余沿链按 tip-gamma 分布到多段
                t = np.clip((u - lock) / max(1e-3, 1 - lock), 0, 1) ** args.tip_gamma
                # t∈[0,1] 映射到链段位置 [0, nseg-1]
                fpos = t * (nseg - 1)
                i0 = int(np.floor(fpos))
                i0 = min(i0, nseg - 2) if nseg >= 2 else 0
                frac = fpos - i0
                if nseg == 1:
                    bw.append([(chain[0], 1.0)])
                elif frac <= EPS:
                    bw.append([(chain[i0], 1.0)])
                elif frac >= 1 - EPS:
                    bw.append([(chain[i0 + 1], 1.0)])
                else:
                    bw.append([(chain[i0], 1 - frac), (chain[i0 + 1], frac)])
        elif cls in ("arm_upper", "arm_fore", "arm_hand", "prop"):
            # 手臂: 各段绑各自骨(纯 FK 旋转, 不动骨位移)。衔接靠子骨枢轴
            # 落在与上级部件的交界处(见 build_skeleton arm_side)。
            # prop(武器/盾): 已挂进同一条手臂链, 100% 跟随那只手/小臂的骨。
            b, prev_b, next_b = _find_arm_bone(p, info)
            tb = b if b else torso
            for _ in v:
                bw.append([(tb, 1.0)])
        else:
            for _ in v:
                bw.append([(torso, 1.0)])
        p["bone_weights"] = bw


def _closest_pair_joint(a, b):
    """两组轮廓顶点里距离最近的一对的中点 —— 用来找两段部件之间的接缝。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return (a.mean(axis=0) if len(a) else b.mean(axis=0))
    # 顶点是轮廓采样, 数量不大(几十~几百), 直接算; 先按 x 预筛减一半
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
    return (a[i] + b[j]) * 0.5


def _find_arm_bone(p, info):
    """返回 (该部件骨, 链上上级骨 or None, 链上下级骨 or None)。
    上级=更靠肩(chain 前一个), 下级=更靠手(chain 后一个)。"""
    for (label, chain) in info["arm_chains"]:
        for k, (pp, b) in enumerate(chain):
            if pp is p:
                prev_b = chain[k - 1][1] if k > 0 else None
                next_b = chain[k + 1][1] if k + 1 < len(chain) else None
                return b, prev_b, next_b
    return None, None, None


# ----------------------------------------------------------------------------
# 动画时间线生成 (驱动骨骼 rotate/scale/translate)
# ----------------------------------------------------------------------------
def _tl_write(bones_tl, fps, tol=1.0):
    """返回 (set_rotate, set_translate, set_scale) 三个时间线写入器。

    写入 bones_tl[bone][prop] = [关键帧...]; 标量序列先做抽稀。
    tol 是抽稀容差倍率: 1.0 保持原有精度(idle 不变), 动作类可放到 2~3
    以减少关键帧(正弦信号逐帧记录会让 Spine 编辑器很重)。
    四个动作(idle/walk/hit/death)共用同一套写入逻辑, 保证导出格式一致。
    """
    tol_r = 0.05 * tol
    tol_t = 0.05 * tol
    tol_s = 0.002 * tol

    def set_rotate(bone, angles_deg):
        keys = _decimate_scalar(angles_deg, fps, tol=tol_r)
        if keys:
            bones_tl.setdefault(bone.name, {})["rotate"] = [
                {"time": round(i / fps, 5), "angle": round(float(angles_deg[i]), 3)}
                for i in keys]

    def set_translate(bone, dx, dy):
        keysx = set(_decimate_scalar(dx, fps, tol_t))
        keysy = set(_decimate_scalar(dy, fps, tol_t))
        ks = sorted(keysx | keysy)
        if ks:
            bones_tl.setdefault(bone.name, {})["translate"] = [
                {"time": round(i / fps, 5),
                 "x": round(float(dx[i]), 3), "y": round(float(dy[i]), 3)}
                for i in ks]

    def set_scale(bone, sx, sy):
        ks = sorted(set(_decimate_scalar(sx, fps, tol_s))
                    | set(_decimate_scalar(sy, fps, tol_s)))
        if ks:
            bones_tl.setdefault(bone.name, {})["scale"] = [
                {"time": round(i / fps, 5),
                 "x": round(float(sx[i]), 4), "y": round(float(sy[i]), 4)}
                for i in ks]

    return set_rotate, set_translate, set_scale


def _blink_tracks(info, args, fps, F, tt):
    """眨眼: 有闭眼图层 -> slot attachment 切换; 无 -> 眼/睫 mesh deform 挤压。
    返回 (blink_slots, blink_deform, closed)。closed: 0=睁 1=全闭。"""
    dur = float(tt[-1]) if len(tt) else 0.0
    act = getattr(args, "action", "idle")
    if act == "walk":
        # 行走: 每步眨一次, 落在两个落脚点之间(避开重心起伏的同一相位)
        blink_times = [dur * f for f in (0.28, 0.63, 0.88)]
    elif act == "hit":
        # 受击: 挨打瞬间闭一下眼(护眼条件反射), 之后睁开
        blink_times = [max(0.08, 0.14 * dur)]
    elif act == "death":
        blink_times = []                     # 死亡靠闭眼补丁, 不在这里眨
    else:
        blink_times = [float(x) for x in args.blink_at.split(",") if x.strip()]
    # 动作时长可能比默认眨眼时刻短, 超出的丢弃, 否则永远不触发
    blink_times = [b for b in blink_times if b < dur - 2 * args.blink_dur]
    if not blink_times:
        return {}, {}, np.zeros(F + 1)

    def blink_factor(t):
        k = 1.0
        for bt in blink_times:
            d = abs(t - bt)
            if d < args.blink_dur:
                bb = 0.5 * (1 + math.cos(math.pi * d / args.blink_dur))
                k = min(k, 1 - args.blink_depth * bb)
        return k
    kf = np.array([blink_factor(t) for t in tt])      # 1=睁 ->压扁比例
    closed = (1 - kf) / max(1e-3, args.blink_depth)   # 0=睁 1=全闭

    pairs = info.get("blink_pairs") or []
    switched = set()                                  # 用切换处理过的睁眼层名
    blink_slots = {}                                  # {slot: attachment 时间线}
    if pairs:
        # 闭眼阈值: closed>0.5 显示闭眼层, 否则睁眼层。生成离散切换关键帧
        thr = 0.5
        state = closed > thr                          # bool 数组(True=闭)
        for (oe, be) in pairs:
            switched.add(oe["name"])
            o_tl, c_tl = [], []
            prev = None
            for fi in range(F + 1):
                cur = bool(state[fi])
                if cur != prev:
                    t = round(fi / fps, 5)
                    o_tl.append({"time": t,
                                 "name": None if cur else oe["attachment"]})
                    c_tl.append({"time": t,
                                 "name": be["attachment"] if cur else None})
                    prev = cur
            blink_slots[oe["name"]] = o_tl
            blink_slots[be["name"]] = c_tl

    # 未被切换处理的睁眼层(及睫毛) -> deform 挤压
    blink_deform = {}                                  # {part_name: keyframes}
    squeeze_parts = [p for p in info["eyes"] if p["name"] not in switched]
    squeeze_parts += info["lashes"]
    for p in squeeze_parts:
        v = p["v_spine"]                               # (nv,2) spine 世界
        ey_c = float(v[:, 1].mean())                   # 眼中心 y(挤压基准)
        nv = len(v)
        flat_frames = []
        for fi in range(F + 1):
            s = 1.0 - kf[fi]
            dy = (ey_c - v[:, 1]) * s
            off = np.zeros(nv * 2)
            off[1::2] = dy                             # spine deform: x偏移,y偏移
            flat_frames.append(off)
        flat = np.array(flat_frames)                   # (F+1, nv*2)
        keep = _decimate_vec(flat, fps, tol=0.3)
        kfs = []
        for fi in keep:
            offset, trimmed = _trim_deform(flat[fi].tolist())
            entry = {"time": round(fi / fps, 5)}
            if trimmed is not None:
                entry["offset"] = offset
                entry["vertices"] = [round(x, 3) for x in trimmed]
            kfs.append(entry)
        blink_deform[p["name"]] = kfs
    return blink_slots, blink_deform, closed


def gen_idle(sk, info, args, fps, dur):
    """待机: 呼吸 + 2.5D 转头 + 眨眼 + 风吹 + 摆臂。整周期无缝循环。"""
    F = int(round(dur * fps))
    T = dur
    tt = np.arange(F + 1) / fps
    w = 2 * math.pi / T                       # 基础角频率(整周期)

    bones_tl = {}
    set_rotate, set_translate, set_scale = _tl_write(bones_tl, fps)
    wb = w * args.breath_cycles
    breath = args.breath * np.sin(wb * tt)
    bob = args.bob * np.sin(wb * tt)
    set_scale(info["torso"], np.ones(F + 1), 1 + breath)
    set_translate(info["torso"], np.zeros(F + 1), bob)

    # ---- 头部: 上下微起伏(点头呼吸)为主 + 2.5D 转面(transform 约束驱动) ----
    # 左右转面改由 head_ctrl 横移驱动五官视差约束(Spine 原生 2.5D), 不再直接
    # 大幅 rotate head。head_ctrl 的 local translateX = 转面信号, 经各约束按
    # 深度放大成视差。head 仍保留极小 rotate 让整头略微跟转, 防生硬。
    wh = w * args.head_cycles
    base = np.sin(wh * tt + 0.3)
    turn = float(info.get("turn", 0.5))
    # 转面横移幅度: 随 turn 0..1 缩放; 以头宽为尺度, turn=1 时约 head_half*0.5
    turn_amp = turn * info["head_half"] * 0.5
    set_translate(info["head_ctrl"], turn_amp * base, np.zeros(F + 1))
    # 整头极轻 rotate(让脖-头有连贯转动感, 不喧宾夺主)
    yaw = args.head_yaw * 0.18 * turn
    set_rotate(info["neck"], yaw * 0.60 * base)
    set_rotate(info["head"], yaw * 0.30 * base)
    set_rotate(info["torso"], yaw * 0.12 * base)
    # 头部上下微起伏(点头呼吸感): 与身体呼吸同相, 头随胸腔轻轻起伏
    nod = args.bob * 0.6 * np.sin(wb * tt) + 0.4 * args.head_yaw * (
        0.5 - 0.5 * np.cos(wh * tt))                    # 缓慢点头叠加呼吸浮动
    set_translate(info["head"], np.zeros(F + 1), nod)

    # ---- 眨眼: 有闭眼图层 -> attachment 切换; 无 -> deform 挤压 ----
    blink_slots, blink_deform, closed = _blink_tracks(info, args, fps, F, tt)
    info["blink_slots"] = blink_slots
    info["blink_deform"] = blink_deform
    # 眉毛: 仍用骨 translate 轻随眨眼下移
    for p in info["brows"]:
        b = info["feat_bones"].get(p["name"])
        if b:
            set_translate(b, np.zeros(F + 1),
                          -args.brow_follow * 12.0 * closed)

    # ---- 风吹: 头发/飘带骨链逐节 rotate (弹簧 + 梢滞后) ----
    chains = info["wind_chains"]
    n_wind = max(1, len(chains))
    order = {id(pp): i for i, (pp, _, _) in enumerate(
        sorted(chains, key=lambda c: c[0]["cx"]))}
    soft = np.clip(args.soft, 0.0, 1.0)
    # 风场信号(整周期)
    wind_main = (np.sin(args.wind_base * w * tt)
                 + 0.4 * np.sin(2 * args.wind_base * w * tt + 0.7)) / 1.4
    gust = 1.0 + args.gust * (0.5 - 0.5 * np.cos(w * tt))

    def roll_loop(arr, shift):
        """对首尾相等(F+1长)的周期数组做循环移位, 保持首尾仍相等。"""
        if shift == 0 or len(arr) <= 2:
            return arr
        per = np.roll(arr[:-1], shift)        # 去末点(=首)后循环移位
        return np.append(per, per[0])         # 重新闭合
    for (pp, bones, tag) in chains:
        base_deg = args.ribbon_deg if tag == "ribbon" else args.wind_deg
        amp = base_deg * args.wind_amp
        kk = (args.ribbon_k if tag == "ribbon" else args.hair_k) * (1 - 0.6 * soft)
        cc = (args.ribbon_c if tag == "ribbon" else args.hair_c) * (1 - 0.45 * soft)
        idx = order.get(id(pp), 0)
        phase = (idx / n_wind) * 2 * math.pi * args.wind_stagger
        shift = int(round(phase / w * fps)) if w > 0 else 0
        target = math.radians(amp) * wind_main * gust
        target = roll_loop(target, shift)
        ang = spring_follow(target, fps, kk, cc)          # 稳态周期, 首尾闭合
        ang_deg = np.degrees(ang)
        # 多段链逐节摆动: 参考官方(发根~21° -> 发梢~47°), 越往梢幅度越大 + 滞后越多
        # -> 鞭梢效应, 头发更柔软。各节摆幅 = 基础 * (0.5 + 0.6*节序), 滞后递增。
        nseg = len(bones)
        # 头巾(cloth): 根段摆幅必须为 0 —— 包在头上的那圈是钉死的, 只有下摆飘。
        # 否则整块布(含压在脸上的部分)一起摆, 看起来像脸在动。
        is_cloth = (pp.get("cls") == "cloth")
        for s, b in enumerate(bones):
            frac = s / max(1, nseg - 1)                # 0=根 1=梢
            # 头巾只有"垂下来的部分"该飘, 包在头上的那圈是钉死的:
            # frac^2.2 让前两段几乎不动, 再乘 0.55 把整体摆幅压低(用户反馈"太大")。
            seg_amp = ((frac ** 2.2) * 0.55) if is_cloth else (0.45 + 0.85 * frac)
            seg_lag = int(soft * 0.10 * fps * frac)    # 梢滞后更多(鞭梢)
            seg_ang = roll_loop(ang_deg, seg_lag) * seg_amp
            set_rotate(b, seg_ang)

    # ---- 手臂 FK: 各关节 rotate, 左右反向摆 (纯旋转, 不动位移避免大小臂断开) ----
    # 左臂往左时右臂往右 -> 镜像对称。仅旋转, 不加 translate(位移会让大小臂脱节)。
    # 三段的角度必须【沿链条递减】(肩 > 肘 > 腕): Spine 里子骨是相对父骨的,
    # 世界角 = 肩+肘+腕。以前反过来(2.2/3.5/5.0), 世界角就成了 2.2/5.7/10.7 ——
    # 肩甲几乎不动, 手却甩了 5 倍, 看着就是"肩膀和小臂各走各的"。
    wa = w * args.arm_cycles
    for (label, chain) in info["arm_chains"]:
        sign = 1.0 if label == "L" else -1.0          # 左右反向
        k_arm = _arm_k(args, label)                   # 左右分开的摆幅倍率
        for j, (pp, b) in enumerate(_arm_seq(chain)):
            if b.name.endswith("_up"):
                deg = args.sh_deg
            elif b.name.endswith("_fore"):
                deg = args.el_deg
            else:
                deg = args.wr_deg
            ph = args.arm_phase * 0.45 * j             # 关节递进相位(连锁)
            a = sign * deg * k_arm * np.sin(wa * tt + ph)
            set_rotate(b, a)

    anim = {}
    if bones_tl:
        anim["bones"] = bones_tl
    return anim, F, T


# ----------------------------------------------------------------------------
# 动作: 行走 / 受击 / 倒地
#   三者和 idle 共用同一套骨架与次级运动参数(风/呼吸/摆臂/转头),
#   只是驱动力换成"身体运动"而不是"风场"。
# ----------------------------------------------------------------------------
def _arm_seq(chain):
    """链里跳过道具(武器/盾)。道具和手共用同一根骨, 若也参与写旋转,
    会把那根骨的角度覆盖掉(实测右小臂 2.5° 被写成 0.36°)。
    道具不写角度也会跟着走 —— 它的权重就绑在那根骨上。"""
    return [(pp, b) for (pp, b) in chain if not pp.get("follow")]


def _arm_k(args, label):
    """左右臂摆幅倍率(界面滑块)。L=左臂 R=右臂, 默认 1.0。"""
    return max(0.0, float(getattr(args, "arm_l" if label == "L" else "arm_r", 1.0)))


def _amp_of(args):
    return max(0.0, float(getattr(args, "act_amp", 1.0)))


# 速度基准: 滑块/参数 100% 对应"以前播放速度的 75%"。
# 也就是所有动作整体放慢 1/3, 步子/呼吸/头发都跟着慢下来。
# 速度基准: 时长统一 ×1/0.75(见 _dur_of), 播放速度 = 原始 75%
SPEED_BASE = 0.75


# 行走摆臂的"透视缩短"系数(前摆 / 后摆), 大臂和小臂各一组。
# 提到模块级是因为**道具的缩放补偿要用同一组数**(刀骨要除回 1/父骨缩放),
# 分成两处写迟早会对不上 —— 一不一致, 刀就会被拉长或压扁。
ARM_UP_SY = (0.055, 0.026)      # 大臂: 前摆压 5.5%, 后摆拉长 2.6%
ARM_FORE_SY = (0.115, 0.050)    # 小臂: 实际还要乘 base(1-0.3*1=0.7) -> 8% / 3.5%
ARM_FORE_BASE = 0.70            # 小臂那一段 k 的折扣(见 gen_walk 的 base)
# 行走再额外放慢一档: 步数不变、时间轴 ×1/0.75。
# 为什么单独一项: 早期实现把 2.0 步/秒吸附成"2.67 秒走 6 步"(实际 2.25 步/秒),
# 用户据此提"再慢到 75%", 于是按"同样的 6 步用 4/3 的时间"来做 = 1.69 步/秒。
WALK_SLOW = 0.75


def _speed_of(args):
    """用户倍率(滑块), 不含 SPEED_BASE。

    行走的"整体放慢"不在这里做: 步数必须保持整数周期(否则循环接不上), 所以
    做法是【步数不变、时间轴拉长】(见 gen_walk 里的 T = dur / SPEED_BASE)。
    只拉长时长而不改相位率, 结果只是"一循环里多走几步", 速度并没变慢 ——
    那个坑踩过一次。
    """
    return max(0.05, float(getattr(args, "speed", 1.0)))


def _loop_freq(dur, freq):
    """把频率吸附到整数个周期, 保证循环首尾严格闭合。"""
    n = max(1, int(round(dur * freq)))
    return n / dur


def _drive_secondary(info, args, fps, tgt_deg, set_rotate, loop=True, only=None):
    """用任意目标角序列驱动头发/飘带链: 弹簧跟随 + 逐节振幅递增 + 梢滞后。
    与 idle 风吹共用 hair-k/hair-c/soft, 但信号来自身体运动(走路甩、受击鞭、倒地拖)。"""
    chains = info.get("wind_chains") or []
    if not chains:
        return
    soft = np.clip(args.soft, 0.0, 1.0)
    tgt = np.radians(np.asarray(tgt_deg, dtype=float))
    for (pp, bones, tag) in chains:
        _is_c = (pp.get("cls") == "cloth")
        if only == "cloth" and not _is_c:
            continue
        if only == "hair" and _is_c:
            continue
        kk = (args.ribbon_k if tag == "ribbon" else args.hair_k) * (1 - 0.6 * soft)
        cc = (args.ribbon_c if tag == "ribbon" else args.hair_c) * (1 - 0.45 * soft)
        ang = spring_follow(tgt, fps, kk, cc, loop=loop)
        nseg = len(bones)
        is_cloth = (pp.get("cls") == "cloth")
        # 头巾: 摆幅要按【骨链长度】归一 —— 同样角度, 链越长末端甩得越远。
        # 以前那批头巾垂布只有短短一截, 0.55 的系数刚好; 现在垂布长了一倍多,
        # 同样的角度末端位移就翻倍, 看着"甩得厉害"。这里改成"末端位移上限"
        # 反算角度: 短链不受影响(仍取 0.55), 长链自动收小。
        k_cloth = 0.55
        if is_cloth and nseg >= 2:
            _tot = 0.0
            for _i in range(1, nseg):
                _tot += math.hypot(bones[_i].wx - bones[_i - 1].wx,
                                   bones[_i].wy - bones[_i - 1].wy)
            if _tot > 1e-3:
                _max_deg = math.degrees(math.atan2(7.0, _tot))   # 末端最多走 7px
                k_cloth = min(0.75, _max_deg / 6.0)
        for s, b in enumerate(bones):
            frac = s / max(1, nseg - 1)              # 0=根 1=梢
            # 头巾: 根段完全不摆(包在头上的那圈必须钉死), 只有末端飘;
            # 头发: 根段也带一点, 否则发根会僵。
            seg_amp = ((frac ** 2.2) * k_cloth) if is_cloth else (0.45 + 0.85 * frac)
            lag = int(0.05 * fps * frac)
            arr = np.roll(ang, lag) if lag else ang
            if loop and len(arr) > 2:                # roll 会破坏闭合, 重新闭合
                arr = np.append(arr[:-1], arr[0])
            # 头巾是【被头部甩动】的: 头往左, 布梢因为惯性留在右边 -> 反向。
            # 取负号, 否则末端的角度和头的旋转同相, 看着就是"布跟着头一起转"。
            sgn = -1.0 if is_cloth else 1.0
            if is_cloth:                              # 布不能一直歪向一边, 去掉直流
                arr = arr - arr.mean()
            set_rotate(b, sgn * np.degrees(arr) * seg_amp)


def gen_walk(sk, info, args, fps, dur):
    """行走(循环): 重心双频起伏 + 左右移重心 + 躯干前倾反扭 + 摆臂 + 发丝滞后。
    原地走, 位移交给引擎; 频率吸附整数周期保证无缝。"""
    bones_tl = {}
    # 行走是长循环且信号平滑, 容差放宽到 2.5 倍 -> 关键帧减半, 肉眼看不出差别
    sr, st, ss = _tl_write(bones_tl, fps, tol=2.5)
    amp = _amp_of(args)
    spd = _speed_of(args)

    cad = max(0.2, args.cadence * spd)               # 步/秒(决定"几步")
    cycles = max(1, int(round(dur * cad / 2.0)))     # 完整左右周期(两步一个周期)
    # 速度基准: 步数保持不变, 时间轴整体拉长 1/SPEED_BASE ——
    # 这样每步走得更久(= 变慢), 而且周期仍是整数个, 循环首尾照样闭合。
    T = dur / WALK_SLOW
    F = int(round(T * fps))
    # 用 linspace 精确铺到 T: 原来 tt = arange(F+1)/fps 的末帧时间
    # (F/fps) 恰好差一点到不了 T, 相位少走零点几度 —— 循环首尾接不上,
    # 实测 leg_R_foot 首帧 0° 末帧 0.86°, 播放时每圈在脚上跳 1.5px。
    tt = np.linspace(0.0, T, F + 1)
    f = cycles / T                                   # 每秒周期数(相位率)
    ph = 2 * math.pi * f * tt
    s = np.sin(ph)
    step = 0.5 - 0.5 * np.cos(2 * ph)                # 0..1, 每周期两次下踩

    # 重心: 落脚下沉 / 抬脚回升, 左右移重心
    bob = args.step_bob * amp * (0.5 - step)
    sway = args.sway * amp * s
    st(sk.root, sway, bob)

    # 躯干: 恒定微前倾 + 随重心反向扭一下(腰胯带动)
    lean = args.lean * amp
    twist = args.twist * amp
    sr(info["torso"], lean + twist * s)
    ss(info["torso"], np.ones(F + 1),
       1 + 0.5 * args.breath * np.sin(2 * ph))
    # 颈/头: 反向补偿抵消扭转(视线稳), 同时叠上"随步点头 + 侧倾"——
    # 只有反向补偿的话头几乎不动, 看着像"头挂在身上不参与走路"。
    nod = args.head_nod * amp * np.sin(2 * ph + 0.55)   # 每次落脚点一下头
    tilt = args.head_tilt * amp * s                      # 随重心左右微倾
    sr(info["neck"], -0.30 * twist * s + 0.40 * nod)
    sr(info["head"], -0.18 * twist * s + 0.75 * nod + tilt)
    st(info["head"], 0.22 * sway, -0.55 * bob)
    # 2.5D 转面(复用 head-turn), 与重心错开半拍
    turn = float(info.get("turn", 0.5))
    st(info["head_ctrl"],
       turn * info["head_half"] * 0.28 * np.sin(ph + 0.5), np.zeros(F + 1))

    # 摆臂: 这批角色正面朝镜头, 手臂是【前后】摆, 不是左右甩。
    # 屏幕上能表达"前后"的只有三样:
    #   ① 一点点横向角(前后甩在平面上的投影) —— 只保留 28%, 给多了就是划水;
    #   ② 前摆那条臂朝镜头: 略粗、略短、手略高(近大远小 + 透视缩短);
    #   ③ 左右反相: 一前一后。
    # --arm-axis side 回到老做法(绕肩左右摆, 适合侧身角色)。
    arm_front = (getattr(args, "arm_axis", "front") != "side")
    for label, chain in info["arm_chains"]:
        sign = 1.0 if label == "L" else -1.0
        k_arm = _arm_k(args, label)
        # 与【对侧腿】严格同相: 左腿迈前时 sin(ph)>0 -> 右臂前摆;
        # 右腿迈前时 sin(ph)<0 -> 左臂前摆。
        # 关键是"前"和"后"要用同一个带符号的信号分开 —— 以前用 |sin| 表示抬臂,
        # 前摆后摆都抬, 于是手臂的节奏是腿的两倍, 看着跟腿对不上。
        p_arm = ph if label == "R" else ph + math.pi
        s_arm = np.sin(p_arm)
        fwd = np.clip(s_arm, 0.0, None)               # 1=前摆到头
        bwd = np.clip(-s_arm, 0.0, None)              # 1=后摆到头
        # 段号 j 必须看【部件自己是哪一段】, 不能看它在链里排第几:
        # 手臂链有两种建法 —— 几何 auto 建出来的顺序是 肩→肘→手, 但"区域就近绑链"
        # 是按部件表顺序 append 的, 顺序可能是反的。按位置取 j 会让肩甲套上"手"的
        # 动作(几乎不转)、手却套上肩的位移 —— 实测华雄右臂就是这样, 左右不对称。
        _SEG_J = {"arm_upper": 0, "arm_fore": 1, "arm_hand": 2}
        for _i, (pp, b) in enumerate(_arm_seq(chain)):
            j = _SEG_J.get(str(pp.get("cls") or ""), _i)
            base = 1.0 - 0.30 * j
            k = amp * k_arm * base
            if arm_front:
                # 前摆: 手抬高 + 手臂透视缩短 + 往身体中线收;
                # 后摆: 手放低 + 手臂拉直(略长) + 往外撇一点。
                # 两者都跟对侧腿同相, 不再用 |sin| 双倍频。
                # 每一段的角都【各写各的、都乘 sign】—— 这里不再先写一条无符号的
                # 底角再覆盖, 那种"先写后盖"就是之前左右不对称的来源。
                if j == 0:
                    # 大臂这段的原则: **几乎不左右摇, 主要靠"缩短 + 抬升 + 略内收"**。
                    # 之前 5.8° 的旋转 + 3px 横移, 结果手部横向行程 11.8px、竖向才
                    # 12.8px —— 看着就是"整条胳膊在左右摆", 而且护肩甲跟着在胸口上
                    # 滑来滑去(用户: "手上半部分与身体连接处动作太大")。
                    # 现在: 旋转 ~1.4°, 横移 1.6px(往身体中线收一点点), 竖移 2px,
                    # scaleY 5.5% —— 大臂自己也要看得见地"压扁"。
                    # 0.16*swing*sin 是"侧向摆"，这里砍到 0.06 —— 侧向只留一点点意思。
                    # 整个肩角都乘 sign: 左右必须【互为镜像】。
                    sr(b, sign * (args.arm_swing * k * 0.03 * np.sin(p_arm)
                                  + (0.7 * fwd - 0.4 * bwd) * k))
                    st(b, sign * (1.2 * fwd - 0.6 * bwd) * k,
                           (1.2 * fwd - 0.5 * bwd) * k)
                    ss(b, 1.0 + (0.010 * fwd - 0.005 * bwd) * k,
                           1.0 - (ARM_UP_SY[0] * fwd - ARM_UP_SY[1] * bwd) * k)
                elif j == 1:
                    # 小臂【不许写位移】。肘骨的位置就是"大臂末端和小臂顶端的接缝"，
                    # 给小臂加 translate 等于把护腕从袖口上揭下来 —— 实测最大
                    # (2.8, 4.2)px，走动时肘部一条 4px 的缝忽开忽合。
                    #
                    # 缩短主要写在这里: scaleY 的锚点在【肘】, 一缩就是护腕+手整个
                    # 往上收, 接缝纹丝不动 —— 这正是"手往前伸、看着变短"的效果。
                    # 系数写在 ARM_FORE_SY 里, 乘了 base(0.7) 之后实际 8%。
                    # 旋转压到 1.2°: 小臂一转, 刀就跟着横扫, 那是"左右摆"的来源。
                    ss(b, 1.0 + (0.008 * fwd - 0.005 * bwd) * k,
                           1.0 - (ARM_FORE_SY[0] * fwd - ARM_FORE_SY[1] * bwd) * k)
                    # 肘角 = 肩角的同向延伸，必须也乘 sign（世界角 = 肩角 + 肘角）。
                    # 另外肘【只能往一个方向弯】：后摆那半程如果算出负角，小臂会
                    # 反向折过直线(=反关节)。取 max(0, ·) 掐掉。
                    _elb = (args.arm_swing * k * 0.04 * np.sin(p_arm + 0.30 * j)
                            + (2.0 * fwd - 1.0 * bwd + 0.4) * k)
                    sr(b, sign * np.maximum(_elb, 0.0))
                else:
                    # 手: 跟着小臂走, 只在自己的关节上再补一点点, 同样要镜像
                    sr(b, sign * (args.arm_swing * k * 0.03 * np.sin(p_arm + 0.30 * j)
                                  + 0.4 * fwd * k))
            else:
                sr(b, sign * args.arm_swing * k * np.sin(ph + 0.30 * j))

        # 道具(刀/盾): 挂在【手】位置上的子骨, 父骨是小臂。
        # 这里**不再**抵消手臂的 scaleY —— 以前为了"刀不被压扁"写了 1/父骨缩放,
        # 结果手和手臂一起缩了 8%、刀却保持原长, 看着就是"武器没跟着手臂收",
        # 握把那里对不上。现在让它完全继承父骨链: 手臂缩多少, 刀就缩多少。
        # (想要"刀不变形"就把下面这行打开, 但必须同时接受手变小手、刀不变小。)

    # 腿: 默认按【正前方】迈步 —— 这批角色是正面朝向镜头, 腿若绕髋左右摆动,
    # 看起来就是"横着挪"。所以前后用 3D 感的组合来表现:
    #   迈出去的腿 -> 大腿透视缩短(膝盖抬高) + 整条腿抬起 + 脚掌下垂一点;
    #   支撑腿     -> 保持原长踩在地上;
    #   左右摆角只留一点点(不然像劈叉)。
    # --leg-axis side 则回到"绕髋左右摆"(适合侧身角色)。
    leg_front = (getattr(args, "leg_axis", "front") != "side")
    for (label, (b_hip, b_knee, b_foot), _top, _bot, _ps) in info.get("leg_chains", []):
        phs = ph if label == "L" else ph + math.pi
        s1 = np.sin(phs)                            # +1 迈前 / -1 在后
        fwd = np.clip(s1, 0.0, None)                # 0=支撑 1=完全迈出
        a_knee = -args.knee_bend * amp * (0.5 + 0.5 * s1) ** 1.6
        if leg_front:
            a_hip = 0.30 * args.leg_swing * amp * s1
            sr(b_hip, a_hip)
            sr(b_knee, a_knee * 0.45)               # 膝盖只微弯, 主靠透视缩短
            sr(b_foot, 0.45 * args.knee_bend * amp * fwd)   # 脚尖下垂, 露出鞋底
            # 透视缩短: 迈出去的那条腿按 sin 缩短, 膝盖因此抬起来
            ss(b_hip, np.ones(F + 1), 1.0 - args.leg_squash * amp * fwd)
            st(b_hip, np.zeros(F + 1), args.leg_lift * amp * fwd)
        else:
            a_hip = args.leg_swing * amp * s1
            sr(b_hip, a_hip)
            sr(b_knee, a_knee)
            sr(b_foot, -(a_hip + a_knee) * 0.85)    # 脚掌尽量踩平
            st(b_hip, np.zeros(F + 1), args.leg_lift * amp * fwd)

    # 发丝: 被身体左右甩 -> 反向滞后
    # 次级运动: 身体左右甩(低频) + 步子带来的高频抖动。只给低频时,
    # 头巾末端几乎不动(实测 2 段链只有末端有权重), 所以补一个 2 倍频分量。
    flut = (-args.hair_lag * 1.6 * amp * s
            + 4.5 * amp * np.sin(2 * ph + 1.15))
    # 布(头巾垂布)不能跟"位置"走, 要跟"速度"走: 阻尼弹簧在高频下与激励速度反相,
    # 也就是相位差 90°(用 cos)。用 sin 时布和头同相, 看着就是"布长在头上一起转"。
    # 关键: 布梢的父骨是 head, 头一转布就整块跟着转 —— 那是"刚体", 不是布。
    # 这里把 head 自己的转角反向减回去(留 20% 跟随), 布梢才会在头转的瞬间"留在原地",
    # 形成惯性滞后/反向摆动。
    head_ang = -0.18 * twist * s + 0.75 * nod + tilt
    # 布梢 = 头部运动的【反向惯性】+ 一点自身抖动。
    # 主项直接取 head 转角的反号(1.4 倍), 头甩到哪布就甩到反方向、并且因惯性慢半拍。
    flut_cloth = (-9.0 * head_ang
                  + 1.0 * amp * np.sin(2 * ph + 1.15))
    _drive_secondary(info, args, fps, flut, sr, loop=True, only="hair")
    _drive_secondary(info, args, fps, flut_cloth, sr, loop=True, only="cloth")

    blink_slots, blink_deform, closed = _blink_tracks(info, args, fps, F, tt)
    info["blink_slots"] = blink_slots
    info["blink_deform"] = blink_deform
    for p in info["brows"]:
        b = info["feat_bones"].get(p["name"])
        if b:
            st(b, np.zeros(F + 1), -args.brow_follow * 12.0 * closed)

    anim = {}
    if bones_tl:
        anim["bones"] = bones_tl
    return anim, F, T


def gen_hit(sk, info, args, fps, dur):
    """受击(一次性): 瞬间后仰 + 衰减回弹 + 头/发滞后, 末帧归零便于接回待机。"""
    F = int(round(dur * fps))
    T = dur
    tt = np.arange(F + 1) / fps
    bones_tl = {}
    sr, st, ss = _tl_write(bones_tl, fps)
    amp = _amp_of(args)
    spd = _speed_of(args)

    t0 = 0.08 * T                                    # 受击发生时刻
    k = np.clip((tt - t0) * spd, 0.0, None)          # 受击后经过的动作时间
    attack = np.clip(k / 0.045, 0.0, 1.0)            # 3 帧急起, 保留冲击感
    env = attack * np.exp(-4.2 * k) * np.cos(2 * math.pi * 2.1 * k)
    env = np.where(tt >= t0, env, 0.0)
    fade = np.clip((T - tt) / max(1e-3, 0.30 * T), 0.0, 1.0)   # 末段归零
    env = env * fade

    # 头/发比身体晚 50ms(惯性)
    lag = np.interp(tt - 0.05, tt, env, left=0.0)
    lag = np.where(tt - 0.05 < 0, 0.0, lag) * fade

    hit_dir = 1.0 if args.impact_dir >= 0 else -1.0

    # 整体也要被打退, 不能只转躯干 —— 只转躯干的话腿是一根不动的柱子,
    # 上半身和下半身像断开的两截, 看着就是"脚不配套"。
    # 这里让 root 绕【脚底中点】后仰一点点(脚基本不动, 身子倒), 再叠加击退平移。
    lean_deg = -args.hit_lean * amp * env        # 负 = 上身往受击方向倒
    th = np.radians(lean_deg)
    pvx, pvy = float(info["cx"]), float(info["by0"])
    c, sn = np.cos(th), np.sin(th)
    st(sk.root,
       pvx - (c * pvx - sn * pvy) - hit_dir * args.impact * amp * env,
       pvy - (sn * pvx + c * pvy) - 0.30 * args.impact * amp * np.abs(env))
    sr(sk.root, lean_deg)
    sr(info["torso"], -0.75 * args.impact_deg * amp * env)
    sr(info["neck"], -0.55 * args.impact_deg * amp * lag)
    sr(info["head"], -0.70 * args.impact_deg * amp * lag)
    st(info["head"], np.zeros(F + 1), -0.25 * args.impact * amp * np.abs(lag))

    # 手臂被震开
    for label, chain in info["arm_chains"]:
        sign = 1.0 if label == "L" else -1.0
        k_arm = _arm_k(args, label)
        for j, (pp, b) in enumerate(_arm_seq(chain)):
            # 甩幅 1.4 -> 0.7: 护肩是绑在大臂上的, 支点搬到肩以后 12.6° 能把
            # 整块肩甲甩到耳朵以上 —— 表现就是"头盔/头被盖住"。
            sr(b, sign * k_arm * (1.0 - 0.25 * j) * args.arm_swing * 0.7
               * amp * lag)

    # 腿: 受击不能只有上身演 —— 下盘要卸力, 否则人被整体推走、脚像钉在地上滑,
    # 看着就"不配套"。做法: 退方向那条腿往后撤半步(抬脚->落地->撑住),
    # 另一条腿微屈顶住; 每条腿都向外张开一点, 重心压下去(屈膝)。
    # 再用 foot-plant 把腿相对 root 反向平移, 脚在地上蹭住而不是整体滑走。
    # step_env 是单峰包络(不带 cos 振荡), 所以脚只抬一次就落下, 不会抖。
    step_env = np.where(tt >= t0, attack * np.exp(-3.2 * k), 0.0) * fade
    step_label = "L" if hit_dir > 0 else "R"         # 退方向那侧的脚先撤
    for (label, (b_hip, b_knee, b_foot), _top, _bot, _ps) in \
            info.get("leg_chains", []):
        side = -1.0 if b_hip.wx < info["cx"] else 1.0   # 画面左腿 -1 / 右腿 +1
        is_step = (label == step_label)                 # 往退方向撤的那条腿
        lead = 1.0 if is_step else 0.45
        a_hip = -side * args.hit_step * amp * lead * step_env   # 向外撑开/撤步
        # 撤步腿的膝盖只微弯: 弯多了大腿往前、小腿往后, 两个转角互相抵消,
        # 鞋就原地不动了(实测脚间距反而变小, 看着像腿在打摆子)。
        a_knee = -args.hit_knee * amp * (0.35 if is_step else 1.0) * step_env
        sr(b_hip, a_hip)
        sr(b_knee, a_knee)
        sr(b_foot, -(a_hip + a_knee) * 0.85)            # 鞋底尽量踩平
        st(b_hip,
           hit_dir * args.hit_plant * args.impact * amp * step_env,
           args.hit_lift * amp * (1.0 if is_step else 0.12) * step_env)

    # 发丝被"甩"向受击方向, 随后回落
    _drive_secondary(info, args, fps, args.hair_lag * 1.8 * amp * lag, sr,
                     loop=False)

    anim = {}
    if bones_tl:
        anim["bones"] = bones_tl
    return anim, F, T


def gen_death(sk, info, args, fps, dur):
    """死亡(一次性): 最后一僵 -> 膝盖卸力 -> 加速塌下 -> 触地压扁 -> 四肢瘫软 -> 定住。

    和原来那条"倒地(fall)"的区别 —— 它只是"摔了一跤", 这条是"死了":
      · **不回弹**。倒地在触地后会弹一下(那口气还没咽), 死亡只塌一次; 触地后的
        过冲一过就彻底不动, 末段 12% 完全静止 —— 引擎钳住最后一帧就是"躺尸"。
      · 时间轴更长(默认 2.0s): 前 60% 倒、后 40% 沉降, 落地有余韵才显得"重"。
      · 头是【跟着】身体倒, 触地后继续往下垂; 倒地时头是反向撑着的(还想抬头)。
      · 手臂先被身体带下去, 触地后垂到地上, 不再张开乱挥。
      · 落地后才闭眼, 然后一直闭着。

    root 在原点, 直接 root.rotate 会绕画布原点转(人整个甩出去),
    所以按"绕脚底中点 P 旋转"反算 root 的平移补偿:
        T = P - R(θ)·S·P
    """
    F = int(round(dur * fps))
    T = dur
    tt = np.arange(F + 1) / fps
    bones_tl = {}
    sr, st, ss = _tl_write(bones_tl, fps)
    amp = _amp_of(args)
    spd = _speed_of(args)

    deg = float(np.clip(args.death_deg * amp, 0.0, 115.0))
    # 实测: 正角度在 spine 里是往【画面左】倒, 所以 dir=1(默认, 右后方) 要用负角度。
    fall_deg = -deg if args.death_dir >= 0 else deg

    # ---- 时间轴(动作时间; 除以 spd 让速度滑块对整段生效) ----
    t_stiff = 0.07 * T / spd            # 生命最后一僵(轻微挺身/抽搐)
    t_loose = 0.28 * T / spd            # 从这一刻起整个人往倒地方向塌
    t_land = 0.52 * T / spd             # 触地
    t_settle = t_land + 0.26 * T / spd  # 四肢完全瘫软(0.78T) -> 之后不再动

    stiff = np.exp(-((tt - t_stiff) / max(1e-3, 0.06 * T)) ** 2)
    # 卸力: 膝盖一软, 身子先矮下去一截。
    # 这段要在【前半秒】里做完(0.05T~0.34T): 拖到最后才降, 前 1/3 会像静止画面,
    # 看着就是"站了半秒然后突然摔倒"。
    loose = smoothstep((tt - 0.05 * T) / max(1e-3, 0.29 * T))
    u = np.clip((tt - t_loose) / max(1e-3, t_land - t_loose), 0.0, 1.0)
    ease = u ** 1.6                     # 末段加速 = 重力; 指数太大会变成"突然摔倒"
    lt = np.clip(tt - t_land, 0.0, None)
    hit_k = np.exp(-9.0 * lt)           # 触地冲击包络: 1 -> 0
    # 触地后一次过冲: 先往地里再沉一点再回位, 但衰减很快 —— 不许弹回"想爬起来"
    over = -0.05 * np.sin(2 * math.pi * 3.2 * lt) * np.exp(-6.0 * lt)
    settle = smoothstep(lt / max(1e-3, t_settle - t_land))

    theta = np.radians(fall_deg * (ease + over))

    # ---- 整体: 侧倾 + 后倒压缩 + 往倒地方向滑一点 ----
    # 正面角色只转 86° 会像"被推倒的木板"; 真倒下时人是往斜后方塌的,
    # 所以侧倾角留一点给"纵向压缩"(躺下后正对镜头的投影变短)。
    back = np.clip(args.death_back * amp, 0.0, 0.6)
    sy0 = 1.0 - back * ease - 0.055 * hit_k               # 触地那一下再压扁
    sx0 = (1.0 - 0.55 * back * ease + args.death_spread * amp * ease
           + 0.030 * hit_k)                                # 压扁时横向撑开
    px, py = float(info["cx"]), float(info["by0"])
    c, sn = np.cos(theta), np.sin(theta)
    bcx, bcy = float(info["cx"]), 0.5 * (float(info["by0"]) + float(info["by1"]))

    # ---- 消失(死亡是终结态, 没有复活, 尸体不能一直躺在场上) ----
    # 龙骨 5.5 只认 bone 的 translate/rotate/scale —— 插槽透明度不在导出链路里,
    # 所以用【整个 root 缩到 1%】来消失: 缩放两条链路都支持, 引擎里最后一帧
    # 就是"什么都没有", 播完直接销毁节点即可。
    # 缩放的锚点默认在脚点 P, 直接缩会把整个人吸到脚下; 这里补一段平移, 让缩放
    # 发生在【身体中心】—— 看起来是"化掉/沉下去", 不是"被吸走"。
    _vd = max(0.0, float(getattr(args, "death_vanish", 0.20)))
    t_van = T - _vd * T / max(1e-6, spd)
    # 收尾提前 3% 收完: 最后几帧真的"什么都没有", 而不是留一个 1px 的小点
    _span = max(1e-3, (T - 0.03 * T) - t_van)
    van = (smoothstep((tt - t_van) / _span)
           if t_van < T - 1e-6 else np.zeros(F + 1))
    # 横向比纵向慢半拍: 先是"贴地压扁"(化开), 最后一小段才收成一点。
    # 等比缩成小人的话像"角色被缩小", 压扁再收才像"身子化进地里"。
    k_van = 1.0 - 0.998 * van
    k_van_x = 1.0 - 0.998 * (van ** 1.7)
    sx, sy = sx0 * k_van_x, sy0 * k_van
    mx = c * sx * px - sn * sy * py
    my = sn * sx * px + c * sy * py
    # Δ = (1-k)·M0·(B-P): 把缩放中心从脚点挪到身体中心
    _dbx, _dby = bcx - px, bcy - py
    ddx = (1.0 - k_van_x) * (c * sx0 * _dbx - sn * sy0 * _dby)
    ddy = (1.0 - k_van) * (sn * sx0 * _dbx + c * sy0 * _dby)
    slide = (args.death_slide * amp * ease
             * (1.0 if args.death_dir >= 0 else -1.0))
    # 绕脚点转会把重心甩出画外, 按"身体中心偏移"往回拉一部分, 留一点倾斜感
    mbcx = c * sx0 * bcx - sn * sy0 * bcy
    drift = (px - mx + slide + mbcx) - bcx
    pull = -args.death_center * drift * ease
    # 下沉有两截: ① 屈膝时身子先矮下去(buckle), ② 瘫软后再沉一点(settle)。
    # **不下沉 root**: 脚踩在 root 的原点上, root 一降脚就陷进地面。
    # 只把躯干降下去、再把两条腿的髋骨反向抬回来 —— 视觉上就是"膝盖软了、
    # 上半身塌下来、脚还踩在地上"。
    drop = (args.death_buckle * amp * loose * (1.0 - 0.65 * ease)
            + args.death_sink * amp * settle)
    st(sk.root, px - mx + slide + pull + ddx,
       py - my + 1.0 * amp * stiff + ddy - 9.0 * amp * van)
    sr(sk.root, np.degrees(theta))
    ss(sk.root, sx, sy)

    # ---- 上身: 死亡时头是"跟着倒"的, 触地后继续垂 ----
    # 倒地那版是反向抵消(头撑着不肯倒), 那是"昏过去"的演法。
    # 这里让 躯干+颈+头 的局部角同向累加, 世界角到 1.3 倍左右: 头最后枕在地上。
    tdeg = np.degrees(theta)
    st(info["torso"], np.zeros(F + 1), -drop)         # 上半身塌下去
    sr(info["torso"], -0.06 * tdeg)
    sr(info["neck"], 0.10 * tdeg + 0.05 * tdeg * settle + 4.0 * amp * loose)
    sr(info["head"], (0.20 * tdeg + 0.14 * tdeg * settle + 9.0 * amp * loose
                      - 2.0 * amp * stiff))

    # ---- 腿: 先卸力屈膝(人不是木板, 是跪下去的), 触地再被压一下 ----
    lead_label = "R" if args.death_dir >= 0 else "L"
    for (label, (b_hip, b_knee, b_foot), _top, _bot, _ps) in \
            info.get("leg_chains", []):
        side = -1.0 if b_hip.wx < info["cx"] else 1.0
        lead = 1.0 if label == lead_label else 0.62
        a_knee = (-args.death_knee * amp * lead * (0.45 * loose + 0.55 * ease)
                  - args.death_knee * amp * lead * 0.20 * hit_k)
        a_hip = -side * args.death_leg * amp * lead * ease
        sr(b_hip, a_hip)
        sr(b_knee, a_knee)
        sr(b_foot, -(a_hip + a_knee) * 0.75)          # 鞋底尽量贴地
        st(b_hip, np.zeros(F + 1), drop)              # 抵消躯干的下沉, 脚留在地上

    # ---- 手臂: 被身体带下去 -> 触地后垂到地上, 不乱挥 ----
    for label, chain in info["arm_chains"]:
        sign = 1.0 if label == "L" else -1.0
        k_arm = _arm_k(args, label)
        for j, (pp, b) in enumerate(_arm_seq(chain)):
            # 各段沿链条略减(手比肩松), 总角只有倒地那版的 1/3 —— 死人不会甩手
            a = (0.12 * loose + 0.30 * ease + 0.60 * settle) * (1.0 - 0.12 * j)
            sr(b, sign * k_arm * args.arm_swing * 0.9 * amp * a)

    # ---- 落地后闭眼, 一直闭着 ----
    # 用【骨缩放】而不是网格 deform: 龙骨 5.5 不支持形变, 只有这样压出来的两份
    # 数据里眼睛才真的闭上。
    if args.death_close_eye:
        t_close = t_land + 0.08 * T
        info["_eye_close_t"] = t_close
        if info.get("eye_closed_bone") is None:
            # 没有闭眼补丁(素材没拆到眼睛): 退而求其次, 把眼睛层压扁成一条缝
            k = smoothstep(np.clip((tt - t_close) / max(0.06, 0.08 * T), 0.0, 1.0))
            for bn in ("b_eyewhite", "b_irides", "b_eyelash"):
                b = sk.get(bn)
                if b is None:
                    continue
                ss(b, np.ones(F + 1), 1.0 - 0.999 * k)
            print("  闭眼: 无补丁, 用压扁眼睛层代替(龙骨里看不到)")
        else:
            print("  闭眼: 落地后盖上闭眼补丁(骨缩放, 龙骨同样有效)")

    # ---- 发丝/头巾: 比身体慢半拍, 触地后拖一下再停 ----
    _drive_secondary(info, args, fps, -0.30 * tdeg, sr, loop=False)

    anim = {}
    if bones_tl:
        anim["bones"] = bones_tl
    return anim, F, T


def _squash_eye_track(anim, info, args, fps, F, action, dur):
    """把眼睛层垂直压扁成一条缝来做"闭眼"。

    为什么不用闭眼补丁: 这套素材里 AI 分层给的 `eyes_closed` 其实是**睁眼重绘**
    (有眼白和红瞳, 实测 5 个角色里 4 个能测出 >12% 眼白), 拿它当补丁,
    眨眼瞬间是把"睁着的眼睛"再盖一层上去 —— 看起来就是重影/没闭上。
    压扁自己的眼睛层反而干净: 眼白+瞳孔+睫毛一起压成线, 视觉上就是闭眼。
    """
    feat = info.get("feat_bones") or {}
    names = [n for n in ("eyewhite", "irides", "eyelash") if n in feat]
    if not names:
        return False
    tt = np.arange(F + 1) / fps
    if action == "walk":
        times = [dur * f for f in (0.28, 0.63, 0.88)]
    elif action == "hit":
        times = [max(0.08, 0.14 * dur)]
    elif action == "death" and getattr(args, "death_close_eye", 1) \
            and info.get("_eye_close_t"):
        times = [float(info["_eye_close_t"])]
    else:
        times = [float(x) for x in str(getattr(args, "blink_at", "")).split(",")
                 if x.strip()]
    times = [t for t in times if 0.02 < t < dur]

    def closed_at(t):
        c = 0.0
        for bt in times:
            d = abs(t - bt)
            if d < args.blink_dur:
                c = max(c, 0.5 * (1 + math.cos(math.pi * d / args.blink_dur)))
        return min(1.0, c * float(args.blink_depth) / 0.72)

    # 压到 8% 高: 眼白/瞳孔/睫毛一起变成一条缝
    depth = 0.92
    if action == "death" and times:
        # 死亡闭眼要"闭住不睁"
        for n in names:
            b = feat[n]
            kf = [{"time": 0.0, "x": 1.0, "y": 1.0}, ]
            t0 = max(0.0, times[0] - 0.06)
            kf.append({"time": round(t0, 5), "x": 1.0, "y": 1.0})
            kf.append({"time": round(times[0], 5), "x": 1.0, "y": round(1 - depth, 4)})
            kf.append({"time": round(F / fps, 5), "x": 1.0, "y": round(1 - depth, 4)})
            anim.setdefault("bones", {}).setdefault(b.name, {})["scale"] = kf
        return True

    for n in names:
        b = feat[n]
        kf = []
        for fi in range(F + 1):
            c = closed_at(tt[fi]) if times else 0.0
            kf.append({"time": round(fi / fps, 5), "x": 1.0,
                       "y": round(1.0 - depth * c, 4)})
        anim.setdefault("bones", {}).setdefault(b.name, {})["scale"] = kf
    return True


def _eye_closed_track(anim, info, args, fps, F, action, dur):
    """闭眼补丁的缩放轨道: 平时缩到 0(看不见), 倒地落地后放大到 1(盖住眼睛)。

    用【骨的缩放】而不是网格 deform / 插槽切换, 因为这两样龙骨 5.5 都不支持
    (龙骨只认 bone 的 translate/rotate/scale)。
    """
    if getattr(args, "no_blink", False):
        return                     # --no-blink: 不做任何眨眼/闭眼
    b = info.get("eye_closed_bone")
    if b is None:
        return
    hid = 0.001
    t_end = round(F / fps, 5)
    if action == "death" and getattr(args, "death_close_eye", 1) and info.get("_eye_close_t"):
        # 关键帧必须落在整帧上: 转龙骨时时间是按 round(t*fps) 取整的,
        # 落在半帧上的话两边插值点错开, 缩放会有几像素偏差(实测 4.1px)。
        f0 = max(0, int(round(float(info["_eye_close_t"]) * fps)))
        f1 = min(F, max(f0 + 1, int(round((float(info["_eye_close_t"])
                                           + max(0.06, 0.08 * dur)) * fps))))
        kf = [{"time": 0.0, "x": hid, "y": hid},
              {"time": round(f0 / fps, 5), "x": hid, "y": hid},
              {"time": round(f1 / fps, 5), "x": 1.0, "y": 1.0},
              {"time": t_end, "x": 1.0, "y": 1.0}]
        anim.setdefault("bones", {}).setdefault(b.name, {})["scale"] = kf
        return

    # 非倒地动作: 这块补丁就是"眨眼"用的 —— 平时缩到 0.001 看不见,
    # 眨眼的瞬间张开盖住眼睛。头部图层本身把眼睛画死了, 只压扁眼睛层是没用的。
    tt = np.arange(F + 1) / fps
    if action == "walk":
        times = [dur * f for f in (0.28, 0.63, 0.88)]     # 每步眨一次
    elif action == "hit":
        times = [max(0.08, 0.14 * dur)]                    # 挨打瞬间闭眼
    else:
        times = [float(x) for x in args.blink_at.split(",") if x.strip()]
    times = [t for t in times if 0.02 < t < dur]

    def closed_at(t):
        c = 0.0
        for bt in times:
            d = abs(t - bt)
            if d < args.blink_dur:
                c = max(c, 0.5 * (1 + math.cos(math.pi * d / args.blink_dur)))
        # 24fps 采样很难正好落在眨眼的峰值上, 按 0.72 归一化让"接近闭合"就算全闭,
        # 否则补丁只张到 0.87 倍, 眼睛还会露出一点。
        return min(1.0, c * float(args.blink_depth) / 0.72)

    kf = []
    for fi in range(F + 1):
        c = closed_at(tt[fi]) if times else 0.0
        v = round(hid + (1.0 - hid) * c, 4)
        kf.append({"time": round(fi / fps, 5), "x": v, "y": v})
    anim.setdefault("bones", {}).setdefault(b.name, {})["scale"] = kf


def gen_animation(sk, info, args, fps, dur):
    """按 --action 分派。idle 走原逻辑, 保证既有产出不变。"""
    action = getattr(args, "action", "idle")
    if action == "walk":
        anim, F, T = gen_walk(sk, info, args, fps, dur)
    elif action == "hit":
        anim, F, T = gen_hit(sk, info, args, fps, dur)
    elif action == "death":
        anim, F, T = gen_death(sk, info, args, fps, dur)
    else:
        anim, F, T = gen_idle(sk, info, args, fps, dur)
    _eye_closed_track(anim, info, args, fps, F, action, dur)
    _root_reset_guard(anim, sk, action)
    return anim, F, T


def _root_reset_guard(anim, sk, action):
    """给"非死亡"动作补一条 root 的 静止关键帧(只在原本没有该通道时补)。

    死亡是靠"把 root 缩到 0"来消失的。Spine 的动画只影响【自己有轨道】的属性:
    播完死亡再切行走/待机, 这两条动画原本没写 root 的 scale(待机连 rotate 都没有),
    就会沿用死亡最后一帧的 0 缩放 / -74° 旋转 —— 人直接不见了, 或者躺着走路。
    这里在 t=0 补一条 bind 值, 切过去就会被立刻拉回正常姿态。
    """
    if action == "death":
        return
    rt = anim.setdefault("bones", {}).setdefault(sk.root.name, {})
    if "translate" not in rt:
        rt["translate"] = [{"time": 0.0, "x": 0.0, "y": 0.0}]
    if "rotate" not in rt:
        rt["rotate"] = [{"time": 0.0, "angle": 0.0}]
    if "scale" not in rt:
        rt["scale"] = [{"time": 0.0, "x": 1.0, "y": 1.0}]


def _decimate_scalar(arr, fps, tol):
    """标量序列关键帧抽稀, 返回保留帧索引(含起止)。近常量则只留端点。"""
    F = len(arr)
    if F <= 2:
        return list(range(F))
    if float(np.ptp(arr)) < tol:
        return [0, F - 1] if abs(arr[0]) > tol or abs(arr[-1]) > tol else []
    kept = [0]
    anchor = 0
    i = 1
    while i < F:
        a, b = arr[anchor], arr[i]
        span = i - anchor
        fits = True
        for j in range(anchor + 1, i):
            interp = a + (b - a) * (j - anchor) / span
            if abs(arr[j] - interp) > tol:
                fits = False
                break
        if not fits:
            kept.append(i - 1)
            anchor = i - 1
        else:
            i += 1
    if kept[-1] != F - 1:
        kept.append(F - 1)
    return kept


# ----------------------------------------------------------------------------
# 预览: 用骨骼 FK + 线性蒙皮在 Python 里复现每帧顶点, 渲染校验
# ----------------------------------------------------------------------------
def _sample_tl(tl_list, F, fps, keys, default):
    """把抽稀的关键帧时间线还原为逐帧数组(线性插值)。keys=要取的字段名列表。"""
    out = {k: np.full(F + 1, default.get(k, 0.0)) for k in keys}
    if not tl_list:
        return out
    times = [kf["time"] for kf in tl_list]
    for k in keys:
        vals = [kf.get(k, default.get(k, 0.0)) for kf in tl_list]
        frames = [int(round(t * fps)) for t in times]
        out[k] = np.interp(np.arange(F + 1), frames, vals)
    return out


def _bone_local_anim(anim, sk, F, fps):
    """解析 animation, 返回每骨逐帧 local (rot_deg, tx, ty, sx, sy)。"""
    bt = anim.get("bones", {})
    res = {}
    for b in sk.bones:
        d = bt.get(b.name, {})
        rot = _sample_tl(d.get("rotate"), F, fps, ["angle"], {"angle": 0.0})["angle"]
        tr = _sample_tl(d.get("translate"), F, fps, ["x", "y"], {"x": 0, "y": 0})
        # 没有 scale 轨道时用骨骼的 setup 缩放(闭眼补丁是 0.001)
        sc = _sample_tl(d.get("scale"), F, fps, ["x", "y"],
                        {"x": b.sx, "y": b.sy})
        res[b.name] = (rot, tr["x"], tr["y"], sc["x"], sc["y"])
    return res


def _bone_world_at(sk, locals_f, fi):
    """计算第 fi 帧每骨的世界变换矩阵(2x2)+平移。返回 {name:(M(2x2), T(2,))}。
    Spine: child_world = parent_world ∘ (translate · rotate · scale)。
    transform 约束(relative+local): 受约束骨 local translate 叠加 target 的
    local translate 增量 × translateMix -> 复现 2.5D 视差预览。"""
    # 预计算约束叠加: {bone_name: (add_tx, add_ty)}
    add_tr = {}
    for c in getattr(sk, "transforms", []):
        tgt = sk.get(c["target"])
        if tgt is None:
            continue
        _, ttx, tty, _, _ = locals_f[tgt.name]
        mix = c["translateMix"]
        for bn in c["bones"]:
            ax, ay = add_tr.get(bn, (0.0, 0.0))
            add_tr[bn] = (ax + ttx[fi] * mix, ay + tty[fi] * mix)
    world = {}
    for b in sk.bones:
        rot, tx, ty, sx, sy = locals_f[b.name]
        ang = math.radians(b.rot0 + rot[fi])
        ca, sa = math.cos(ang), math.sin(ang)
        # 本地: 先缩放再旋转
        M_local = np.array([[ca * sx[fi], -sa * sy[fi]],
                            [sa * sx[fi], ca * sy[fi]]])
        lx, ly = b.local_xy()
        atx, aty = add_tr.get(b.name, (0.0, 0.0))
        T_local = np.array([lx + tx[fi] + atx, ly + ty[fi] + aty])
        if b.parent is None:
            M = M_local
            T = T_local
        else:
            PM, PT = world[b.parent.name]
            M = PM @ M_local
            T = PM @ T_local + PT
        world[b.name] = (M, T)
    return world


def render_preview_bones(parts, sk, info, args, F, cw, ch, out_dir,
                         action="idle"):
    import cv2

    def warp(img, uvs, tris, dst_xy, canvas, ga=1.0):
        h, w = img.shape[:2]
        n = len(uvs) // 2
        src = np.array([[uvs[2 * i] * w, uvs[2 * i + 1] * h]
                        for i in range(n)], np.float32)
        for t in range(0, len(tris), 3):
            ia, ib, ic = tris[t], tris[t + 1], tris[t + 2]
            s = src[[ia, ib, ic]]
            d = np.array([dst_xy[ia], dst_xy[ib], dst_xy[ic]], np.float32)
            r = cv2.boundingRect(d)
            if r[2] <= 0 or r[3] <= 0:
                continue
            x0, y0, ww, hh = r
            d2 = (d - np.array([x0, y0], np.float32)).astype(np.float32)
            M = cv2.getAffineTransform(s.astype(np.float32), d2)
            patch = cv2.warpAffine(img, M, (ww, hh), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_TRANSPARENT)
            mask = np.zeros((hh, ww), np.uint8)
            cv2.fillConvexPoly(mask, d2.astype(np.int32), 255)
            xa, ya = max(0, x0), max(0, y0)
            Hc, Wc = canvas.shape[:2]
            xb, yb = min(Wc, x0 + ww), min(Hc, y0 + hh)
            if xb <= xa or yb <= ya:
                continue
            syl, sxl = ya - y0, xa - x0
            pa = patch[syl:syl + (yb - ya), sxl:sxl + (xb - xa)]
            ma = mask[syl:syl + (yb - ya), sxl:sxl + (xb - xa)]
            if pa.shape[2] == 4:
                al = (pa[:, :, 3:4] / 255.0) * (ma[:, :, None] / 255.0) * ga
                canvas[ya:yb, xa:xb] = (canvas[ya:yb, xa:xb] * (1 - al)
                                        + pa[:, :, :3] * al)

    imgs = {}
    for p in parts:
        cand = os.path.join(args.parts, p["attachment"] + ".png")
        if not os.path.isfile(cand):
            cand = os.path.join(out_dir, "images", p["attachment"] + ".png")
        imgs[p["name"]] = np.array(Image.open(cand).convert("RGBA"))

    locals_f = _bone_local_anim(_LAST_ANIM, sk, F, args.fps)

    # 眨眼 deform: 预解析为逐帧顶点偏移 {part_name: (F+1, nv, 2)}
    blink_off = {}
    bd = info.get("blink_deform") or {}
    for p in parts:
        kfs = bd.get(p["name"])
        if not kfs:
            continue
        nv = len(p["v_spine"])
        full = np.zeros((F + 1, nv * 2))
        kept = [int(round(k["time"] * args.fps)) for k in kfs]
        mats = []
        for k in kfs:
            arr = np.zeros(nv * 2)
            if k.get("vertices"):
                off = k.get("offset", 0)
                vs = k["vertices"]
                arr[off:off + len(vs)] = vs
            mats.append(arr)
        mats = np.array(mats)
        for d in range(nv * 2):
            full[:, d] = np.interp(np.arange(F + 1), kept, mats[:, d])
        blink_off[p["name"]] = full.reshape(F + 1, nv, 2)

    # attachment 切换可见性: {part_name: bool[F+1]}
    visible = {}
    bs = info.get("blink_slots") or {}
    for p in parts:
        if p["cls"] == "blink_eye":
            vis = np.zeros(F + 1, bool)         # 闭眼层默认隐藏
        else:
            vis = np.ones(F + 1, bool)
        tl = bs.get(p["name"])
        if tl:
            cur = (p["cls"] != "blink_eye")     # 睁眼层默认显示, 闭眼默认隐藏
            ti = 0
            for fi in range(F + 1):
                while ti < len(tl) and int(round(tl[ti]["time"] * args.fps)) <= fi:
                    cur = tl[ti]["name"] is not None
                    ti += 1
                vis[fi] = cur
        visible[p["name"]] = vis

    # 预计算每顶点的骨绑定 (索引 + 局部坐标 + 权重)
    for p in parts:
        binds = []
        for (sx, sy), bws in zip(p["v_spine"], p["bone_weights"]):
            entry = [(bw[0].name, sx - bw[0].wx, sy - bw[0].wy, bw[1])
                     for bw in bws]
            binds.append(entry)
        p["_binds"] = binds

    gif = []
    sh = 200
    # 倒地是绕脚点旋转, 身体会转到画面左侧; 预览需要额外左边距才看得全。
    # 注意这只影响预览取景, 不影响导出的 Spine 坐标(引擎里由相机决定)。
    # 死亡是绕脚点倒向一侧的: 只在倒下去的那一侧留白, 不然最后几帧被画布边缘切掉。
    # (两侧都留白会把画面拉得很扁, 预览里角色只剩一条)
    _ox = int((info["by1"] - info["by0"]) * 0.55) if action == "death" else 0
    _dirr = getattr(args, "death_dir", 1)
    ox_l = _ox if (_ox and _dirr < 0) else 0
    ox_r = _ox if (_ox and _dirr >= 0) else 0
    cw_v = cw + ox_l + ox_r
    strip_idx = [int(F * k / 7) for k in range(8)]
    strip = []
    for fi in range(F):
        world = _bone_world_at(sk, locals_f, fi)
        canvas = np.full((ch, cw_v, 3), 60, float)
        for p in parts:
            if not visible[p["name"]][fi]:        # attachment 切换隐藏的跳过
                continue
            dst = []
            doff = blink_off.get(p["name"])
            for vi, entry in enumerate(p["_binds"]):
                wx = wy = 0.0
                for (bname, lx, ly, wgt) in entry:
                    M, T = world[bname]
                    px = M[0, 0] * lx + M[0, 1] * ly + T[0]
                    py = M[1, 0] * lx + M[1, 1] * ly + T[1]
                    wx += px * wgt
                    wy += py * wgt
                if doff is not None:                  # 叠加眨眼 deform (spine y上)
                    wx += doff[fi, vi, 0]
                    wy += doff[fi, vi, 1]
                dst.append([wx + ox_l, ch - wy])  # spine y上 -> 画布 y下
            warp(imgs[p["name"]], p["uvs"], p["triangles"],
                 np.array(dst, float), canvas)
        full = canvas.astype(np.uint8)
        sc = 400.0 / max(cw_v, ch)
        gif.append(Image.fromarray(full).resize(
            (max(1, int(cw_v * sc)), max(1, int(ch * sc)))))
        if fi in strip_idx:
            sw = max(1, int(cw_v * sh / ch))
            strip.append(cv2.resize(full, (sw, sh)))
    suffix = "loop" if action in ("idle", "walk") else "once"
    gp = os.path.join(out_dir, f"_{action}_{suffix}.gif")
    gif[0].save(gp, save_all=True, append_images=gif[1:],
                duration=1000 // 30, loop=0, disposal=2)
    if strip:
        Image.fromarray(np.hstack(strip)).save(
            os.path.join(out_dir, f"_{action}_strip.png"))
    print(f"  预览 -> {gp} + _{action}_strip.png")



# ----------------------------------------------------------------------------
# Spine JSON 组装 + 主流程
# ----------------------------------------------------------------------------
def build_spine_json(parts, sk, info, anim, cw, ch, F, T, action="idle", fps=30):
    spine = {
        "skeleton": {"spine": "3.8.86", "x": 0, "y": 0,
                     "width": cw, "height": ch,
                     "fps": fps,
                     "images": "./images/", "audio": ""},
        "bones": sk.export(),
        "slots": [],
        "skins": [{"name": "default", "attachments": {}}],
        "animations": {},
    }
    tcons = sk.export_transforms()
    if tcons:
        spine["transform"] = tcons
    att = spine["skins"][0]["attachments"]
    for p in parts:
        # slot 要挂到"主导骨"上, 不能全写 root: 龙骨这类编辑器导入后会按骨骼给
        # slot 重新排队, 全挂 root 就等于没告诉它谁属于谁 —— 实测头饰被排到最底、
        # 头盔被脸盖住。spine 官方导出也总是写具体骨骼。
        _bw = p.get("bone_weights") or []
        _acc = {}
        for _vtx in _bw:                      # 每个顶点一组 [(bone, w), ...]
            for _e in _vtx:
                _b = _e[0]
                _w = float(_e[1]) if len(_e) > 1 else 1.0
                _acc[_b] = _acc.get(_b, 0.0) + _w
        _main = max(_acc, key=_acc.get) if _acc else None
        slot = {"name": p["name"], "bone": (_main.name if _main else "root"),
                "attachment": p["attachment"]}
        if p["bird"]:
            slot["color"] = "ffffff00"
        # 闭眼层: setup 初始不挂图(睁眼态), 动画切换显示
        if p["cls"] == "blink_eye":
            slot.pop("attachment", None)
        spine["slots"].append(slot)
        verts = skin_vertices(
            [(float(x), float(y)) for (x, y) in p["v_spine"]],
            p["bone_weights"])
        att[p["name"]] = {p["attachment"]: {
            "type": "mesh", "uvs": p["uvs"], "triangles": p["triangles"],
            "vertices": verts, "hull": p["hull"], "edges": p["edges"],
            "width": p["iw"], "height": p["ih"]}}
    # 眨眼 deform 时间线 (眼/睫毛网格挤压, 不经骨骼)
    bd = info.get("blink_deform") or {}
    if bd:
        deform_skin = {}
        name2att = {p["name"]: p["attachment"] for p in parts}
        for pname, kfs in bd.items():
            if not any(k.get("vertices") for k in kfs):
                continue
            deform_skin.setdefault(pname, {})[name2att[pname]] = kfs
        if deform_skin:
            anim.setdefault("deform", {})["default"] = deform_skin
    # 眨眼 attachment 切换时间线 (有闭眼图层时)
    bs = info.get("blink_slots") or {}
    if bs:
        slot_tl = anim.setdefault("slots", {})
        for sname, tl in bs.items():
            slot_tl.setdefault(sname, {})["attachment"] = tl
    spine["animations"][action] = anim
    return spine


# 各动作的默认时长(秒)。行走要够一个完整步态循环, 受击短促,
# 死亡要留出"倒下 + 沉降 + 定住"三段, 比原来的倒地(1.6)长。
_ACTION_DUR = {"idle": 5.0, "walk": 2.0, "hit": 0.9, "death": 2.0}
_ACTIONS = ("idle", "walk", "hit", "death")
# 旧名字: 以前这条动作叫 fall(倒地)。留着当别名, 老命令/老脚本不用改。
_ACTION_ALIAS = {"fall": "death", "die": "death", "dead": "death"}


def _parse_actions(s):
    """'walk,hit,death' -> ['walk','hit','death']; 去重保序, 未知动作报错。

    fall(旧名) 会自动映射成 death。
    """
    out = []
    for k in str(s or "").replace(" ", "").split(","):
        if not k:
            continue
        k = _ACTION_ALIAS.get(k, k)
        if k not in _ACTIONS:
            raise SystemExit(f"未知动作: {k} (可选: {', '.join(_ACTIONS)})")
        if k not in out:
            out.append(k)
    return out or ["idle"]


def _parse_map(s):
    out = {}
    for kv in str(s or "").replace(" ", "").split(","):
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def _dur_of(args, action):
    """动作时长。界面上填的是"以前的速度", 实际生成时拉长 1/0.75 ——
    同一步数用更长时间走完 = 播放速度是以前的 75%。"""
    if args.duration is not None:
        base = float(args.duration)
    else:
        base = float(_parse_map(args.duration_map).get(
            action, _ACTION_DUR.get(action, 5.0)))
    return base / SPEED_BASE


def _resolve_targets(args, actions):
    """解析输出路径 -> (输出根目录, {动作: result.json 路径})。

    单动作: --out 就是最终文件(旧行为不变)。
    多动作: --out-dir/<动作>/result.json; 若只想给 --out, 且它的目录名本身就是
            动作名(如 .../walk/result.json), 会退一层当作根目录。
    """
    if args.out_dir:
        base = os.path.abspath(args.out_dir)
        return base, {a: os.path.join(base, a, "result.json") for a in actions}
    if not args.out:
        raise SystemExit("需要 --out (单动作) 或 --out-dir (多动作)")
    out = os.path.abspath(args.out)
    if len(actions) == 1:
        return os.path.dirname(out), {actions[0]: out}
    d = os.path.dirname(out)
    base = os.path.dirname(d) if os.path.basename(d) in actions else d
    return base, {a: os.path.join(base, a, "result.json") for a in actions}


def _gen_one(args, action, out_path, dur, coords, cw, ch):
    """生成单个动作的 Spine JSON + 预览, 返回该动作的产物信息。"""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    parts = load_parts(coords, args.parts, img_dir, ch, args.square_mesh)
    if coords.pop("_wear_split", False):
        # 头饰被拆成了 headwear + headscarf: 把结果写回 coords, 让界面预览、
        # 后续几次生成看到的是同一个状态(否则每次都在内存里拆, 磁盘上还是合并的)
        try:
            with open(args.coords, "w", encoding="utf-8") as _f:
                json.dump(coords, _f, ensure_ascii=False, indent=2)
            print(f"  coords 已更新(含 headscarf): {args.coords}")
        except OSError as _ex:                       # noqa: BLE001
            print(f"  coords 写回失败: {_ex}")

    # 拆图结果体检: 早期版本只有 16 个部件(没有腿、没有道具), 用它生成的动画会
    # 出现"腿没了/刀盾不跟手"。以前这里按**部件总数**(<20)拦, 但头发/五官并进
    # head 之后总数会掉到 18, 该有的一样不缺 —— 改成按功能角色查。
    from parts_check import check_parts
    _miss = check_parts(parts)
    if _miss:
        print("  ⚠ 这份拆图结果像是旧版: 缺 " + "、".join(_miss))
        print("    请重跑一次 ① 拆图(或 build_from_seethrough.py --head-wear scarf --leg-halves)")
    sk, info = build_skeleton(parts, cw, ch, args)
    assign_weights(parts, sk, info, args)
    anim, F, T = gen_animation(sk, info, args, args.fps, dur)
    global _LAST_ANIM
    _LAST_ANIM = anim
    spine = build_spine_json(parts, sk, info, anim, cw, ch, F, T,
                             action=action, fps=args.fps)
    json.dump(spine, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    print(f"完成(骨骼驱动) -> {out_path}")
    print(f"  骨骼 {len(sk.bones)} 根, 部件 {len(parts)}, "
          f"动画骨时间线 {len(anim.get('bones', {}))} 条, 时长 {T}s")
    print(f"  head 2.5D yaw={args.head_yaw} 风吹链 {len(info['wind_chains'])} "
          f"手臂链 {len(info['arm_chains'])}")
    print(f"  主干={'手标' if info.get('trunk_marked') else '几何自动'} "
          f"通用骨链 {info.get('n_generic', 0)} 条 "
          f"(区域无链自动补 {info.get('n_auto_wind', 0)} 片飘动)")

    if args.preview:
        try:
            render_preview_bones(parts, sk, info, args, F, cw, ch, out_dir,
                                 action=action)
        except Exception as ex:                       # noqa: BLE001
            print(f"  预览跳过: {ex}")
    return {"action": action, "out": out_path, "images": img_dir, "dur": dur}


def main():
    ap = argparse.ArgumentParser(
        description="全骨骼驱动动作生成器 (可一次生成多个动作)")
    ap.add_argument("--parts", required=True)
    ap.add_argument("--coords", required=True)
    ap.add_argument("--out", default="",
                    help="单动作输出 result.json; 多动作可改用 --out-dir")
    ap.add_argument("--duration", type=float, default=None,
                    help="所有动作统一时长(秒); 不给则按动作取默认/--duration-map")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--db-keep", action="store_true",
                    help="Spine 包里保留之前已生成的其他动作(默认只放这次勾选的)")
    ap.add_argument("--db-weighted", action="store_true",
                    help="(已废弃, 保留兼容) 不再使用")
    ap.add_argument("--dragonbones", action="store_true",
                    help="(已废弃, 保留兼容) 现在总是只导出 Spine 包")
    ap.add_argument("--cocos", action="store_true",
                    help="额外导出一份 Cocos 用的 DragonBones 5.5 包(<名字>_ske/_tex.json + _tex.png)")
    ap.add_argument("--db-version", default="5.5",
                    help="DragonBones 数据版本; Cocos 只认 5.5(6.0 的 animation 是数组, 会导入失败)")
    ap.add_argument("--db-plain", action="store_true",
                    help="DragonBones 用「每个部件一张图贴骨」的写法(无网格形变, 兼容性最高)")
    ap.add_argument("--no-db-mesh", dest="db_mesh_also", action="store_false",
                    default=True,
                    help="不再额外导出 cocos_mesh/(网格版, 对照用)")
    ap.add_argument("--db-rigid", action="store_true",
                    help="DragonBones 用刚性四边形网格(无权重, 与官方 Pro 导出同形)")
    ap.add_argument("--db-rigid-slots", default="",
                    help="只把这几个插槽导成刚性四边形(逗号分隔), 其余仍用带权重的形状网格。"
                         "编辑器重建网格三角时, 小部件(头/五官/头饰)用四边形最稳")
    # 动作类型与总体调节
    ap.add_argument("--action", default="idle",
                    help="idle 待机 / walk 行走 / hit 受击 / death 死亡; "
                         "多个动作用逗号隔开(如 walk,hit,death) 一次全出, "
                         "导出时合并成一个 Spine 骨架的多条动画")
    ap.add_argument("--out-dir", default="",
                    help="多动作时的输出根目录: <out-dir>/<动作>/result.json; "
                         "不给则从 --out 推断")
    ap.add_argument("--duration-map", default="",
                    help="分动作时长, 如 idle=5,walk=2,hit=0.9,death=2; "
                         "没写的动作用内置默认值")
    ap.add_argument("--act-amp", type=float, default=1.0,
                    help="动作幅度倍率 0~2, 缩放该动作的主位移与主角度")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="速度倍率 0.2~3: 行走改步频, 受击/倒地按时间轴缩放")
    # 行走
    ap.add_argument("--step-bob", type=float, default=3.5, help="行走重心起伏 px")
    ap.add_argument("--sway", type=float, default=4.0, help="行走左右移重心 px")
    ap.add_argument("--lean", type=float, default=2.0, help="行走躯干前倾 deg")
    ap.add_argument("--twist", type=float, default=2.5, help="行走扭身 deg")
    ap.add_argument("--cadence", type=float, default=2.0, help="步频 步/秒")
    ap.add_argument("--arm-swing", type=float, default=9.0, help="摆臂幅度 deg")
    ap.add_argument("--arm-axis", default="front", choices=["front", "side"],
                    help="摆臂轴向: front=前后摆(正面角色, 默认) / side=左右摆(侧身角色)")
    ap.add_argument("--arm-l", type=float, default=1.0,
                    help="左臂摆幅倍率(0~2): 界面「左臂摆动」滑块")
    ap.add_argument("--arm-r", type=float, default=1.0,
                    help="右臂摆幅倍率(0~2): 界面「右臂摆动」滑块")
    ap.add_argument("--leg-swing", type=float, default=24.0,
                    help="行走: 髋部前后摆角 deg (需要左右腿是分开的部件)")
    ap.add_argument("--knee-bend", type=float, default=26.0,
                    help="行走: 膝盖弯曲 deg")
    ap.add_argument("--leg-lift", type=float, default=6.0,
                    help="行走: 摆动相抬脚 px")
    ap.add_argument("--leg-squash", type=float, default=0.15,
                    help="行走: 迈出腿的大腿透视缩短比例(0~0.3)")
    ap.add_argument("--leg-axis", default="front", choices=("front", "side"),
                    help="行走迈步方式: front=向正前方(正面角色, 默认), side=绕髋左右摆")
    ap.add_argument("--head-nod", type=float, default=1.8,
                    help="行走: 头部随步点头幅度 deg (0=不点头)")
    ap.add_argument("--head-tilt", type=float, default=0.7,
                    help="行走: 头部左右侧倾幅度 deg")
    ap.add_argument("--hair-lag", type=float, default=6.0,
                    help="发丝被身体带动的滞后幅度 deg")
    # 受击
    ap.add_argument("--impact", type=float, default=12.0, help="受击后退 px")
    ap.add_argument("--impact-deg", type=float, default=14.0, help="受击后仰 deg")
    ap.add_argument("--impact-dir", type=int, default=1,
                    help="受击方向 1=向画面左退(默认), -1=向画面右退")
    ap.add_argument("--hit-lean", type=float, default=5.0,
                    help="受击: 整体绕脚点后仰 deg (0=只有躯干动, 腿会显得没跟上)")
    ap.add_argument("--hit-knee", type=float, default=24.0,
                    help="受击: 屈膝卸力 deg (0=腿不动)")
    ap.add_argument("--hit-step", type=float, default=16.0,
                    help="受击: 撤步腿髋外张 deg (撑开下盘)")
    ap.add_argument("--hit-lift", type=float, default=4.0,
                    help="受击: 撤步抬脚 px")
    ap.add_argument("--hit-plant", type=float, default=0.35,
                    help="受击: 脚相对身体的反向补偿 0~1 (越大脚越像踩在地上, 不是被推着滑)")
    # 死亡 (旧名 fall/倒地, --fall-* 仍作别名保留)
    ap.add_argument("--death-deg", "--fall-deg", dest="death_deg", type=float,
                    default=74.0,
                    help="死亡侧倾角 deg (左右方向的分量; 正面角色不需要 90° 横倒)")
    ap.add_argument("--death-back", "--fall-back", dest="death_back", type=float,
                    default=0.18,
                    help="死亡后倒压缩 0~0.6: 纵向压扁表示向画面【后方】塌下去")
    ap.add_argument("--death-spread", "--fall-spread", dest="death_spread",
                    type=float, default=0.05,
                    help="死亡横向放宽 0~0.3 (躺下后投影变宽)")
    ap.add_argument("--death-slide", "--fall-slide", dest="death_slide",
                    type=float, default=12.0,
                    help="死亡时向倒地方向的水平滑移 px")
    ap.add_argument("--death-center", "--fall-center", dest="death_center",
                    type=float, default=0.6,
                    help="死亡后把身体拉回原位的比例 0~1 (0=纯物理, 会明显偏向倒地方向; "
                         "越大越贴近原地, 但脚会带一点滑)")
    ap.add_argument("--death-close-eye", "--fall-close-eye", dest="death_close_eye",
                    type=int, default=1,
                    help="落地后闭眼并一直闭着, 1=开 0=关")
    ap.add_argument("--death-knee", "--fall-knee", dest="death_knee", type=float,
                    default=40.0, help="死亡屈膝 deg (先卸力跪下, 再塌)")
    ap.add_argument("--death-leg", "--fall-leg", dest="death_leg", type=float,
                    default=10.0, help="死亡双腿外张 deg")
    ap.add_argument("--death-dir", "--fall-dir", dest="death_dir", type=int, default=1,
                    help="倒地方向 1=右后(默认), -1=左后")
    ap.add_argument("--death-sink", type=float, default=2.0,
                    help="瘫软后整体再下沉 px (重量感; 0=不动)")
    ap.add_argument("--death-buckle", type=float, default=11.0,
                    help="屈膝卸力时身子先矮下去多少 px (0=看不出腿软, 直接倒)")
    ap.add_argument("--death-vanish", type=float, default=0.20,
                    help="结尾消失占整段的比例 0~0.5 (0=不消失, 尸体留在地上; "
                         "0.24=最后 1/4 缩到看不见 -> 化掉/沉下去)")
    # 呼吸
    ap.add_argument("--breath", type=float, default=0.016)
    ap.add_argument("--bob", type=float, default=2.5)
    ap.add_argument("--breath-cycles", type=float, default=2.0)
    ap.add_argument("--foot-lock", type=float, default=0.42,
                    help="自底向上锁定过渡带比例(脚踩地)")
    # 转头
    ap.add_argument("--head-yaw", type=float, default=5.0)
    ap.add_argument("--head-cycles", type=float, default=1.0)
    ap.add_argument("--head-turn", type=float, default=0.5,
                    help="2.5D 转面强度 0-1 (0=不转, 0.5=微转默认, 1=大幅)")
    # 眨眼
    ap.add_argument("--blink-at", default="1.6,3.9")
    ap.add_argument("--blink-dur", type=float, default=0.13)
    ap.add_argument("--blink-depth", type=float, default=0.88)
    ap.add_argument("--no-blink", action="store_true",
                    help="完全不生成眨眼/闭眼轨道(角色始终睁眼)。"
                         "素材的 eyes_closed 是睁眼重绘, 补丁方式会成重影时用这个")
    ap.add_argument("--brow-follow", type=float, default=0.10)
    # 风吹
    ap.add_argument("--wind-deg", type=float, default=7.0)
    ap.add_argument("--ribbon-deg", type=float, default=16.0)
    ap.add_argument("--wind-amp", type=float, default=0.5)
    ap.add_argument("--wind-base", type=float, default=1.0)
    ap.add_argument("--gust", type=float, default=0.6)
    ap.add_argument("--wind-stagger", type=float, default=0.6)
    ap.add_argument("--soft", type=float, default=0.5)
    ap.add_argument("--hair-k", type=float, default=85.0)
    ap.add_argument("--hair-c", type=float, default=10.0)
    ap.add_argument("--ribbon-k", type=float, default=55.0)
    ap.add_argument("--ribbon-c", type=float, default=7.0)
    ap.add_argument("--root-lock", type=float, default=0.28)
    ap.add_argument("--tip-gamma", type=float, default=1.5)
    # 手臂
    ap.add_argument("--sh-deg", type=float, default=3.4)
    ap.add_argument("--el-deg", type=float, default=1.5)
    ap.add_argument("--wr-deg", type=float, default=0.7)
    ap.add_argument("--arm-cycles", type=float, default=1.0)
    ap.add_argument("--arm-phase", type=float, default=1.9)
    # 兼容 gen_idle2 多余参数(忽略)
    ap.add_argument("--wind-mask", default="")
    ap.add_argument("--blink-eye", default="")
    ap.add_argument("--wave", type=float, default=1.4)
    ap.add_argument("--wind-base-deg", type=float, default=0)
    ap.add_argument("--arm-bob", type=float, default=2.0)
    ap.add_argument("--tol", type=float, default=0.4)
    ap.add_argument("--bones-json", default="",
                    help="手标骨链 JSON(像素坐标); 空则自动找 coords 同目录 bones.json")
    ap.add_argument("--square-mesh", action="store_true",
                    help="强制用正方形规则网格(关闭按 alpha 形状自适应布点)")
    args, _unknown = ap.parse_known_args()

    coords = json.load(open(args.coords, encoding="utf-8"))
    canv = coords.get("canvas", {"w": 1280, "h": 720})
    cw = canv["w"] if isinstance(canv, dict) else canv[0]
    ch = canv["h"] if isinstance(canv, dict) else canv[1]

    args._markers = load_bone_markers(args.bones_json, args.coords, ch)

    actions = _parse_actions(args.action)
    base_dir, targets = _resolve_targets(args, actions)
    if len(actions) > 1:
        print(f"多动作生成: {' / '.join(actions)}  ->  {base_dir}")
        print("  (拆图部件与骨架每个动作各建一次; 最后合并导出成一个龙骨骨架)")

    made = []
    for act in actions:
        if len(actions) > 1:
            print(f"\n--- {act} ---")
        args.action = act
        made.append(_gen_one(args, act, targets[act], _dur_of(args, act),
                             coords, cw, ch))

    # ---- 导出 Spine 包(唯一导出方式) ----
    # 只出 Spine 3.8 三件套(png + atlas + json): LoongBones 只认这个;
    # DragonBones 的 _ske.json 它根本导入不了, 已全部取消, 不再生成 dragonbones*。
    try:
        import make_spine_pack as MSP
        root_dir = base_dir
        if (len(made) == 1 and os.path.basename(root_dir.rstrip("\\/"))
                == made[0]["action"]):
            root_dir = os.path.dirname(root_dir)
        db_name = os.path.basename(root_dir.rstrip("\\/")) or made[0]["action"]

        # 文件里只放**这次勾选的动作**; --db-keep 才把之前生成过的并回来
        _items = list(made)
        if getattr(args, "db_keep", False):
            _have = {m["action"] for m in _items}
            for _act in _ACTIONS:
                if _act in _have:
                    continue
                _p = os.path.join(root_dir, _act, "result.json")
                if os.path.isfile(_p):
                    _items.append({"action": _act, "out": _p,
                                   "images": os.path.join(root_dir, _act, "images")})
            if len(_items) > len(made):
                print(f"      并入已有动作(--db-keep): "
                      f"[{' '.join(m['action'] for m in _items[len(made):])}]")

        MSP.build([(m["out"], m["images"], m["action"]) for m in _items],
                  os.path.join(root_dir, "spine"), db_name)

        # ---- Cocos 用的 DragonBones 5.5 ----
        # Cocos Creator 的龙骨运行时有版本闸门: 数据里 "version" 必须是 5.x。
        # 6.0 的数据结构有两处不一样 —— animation 是**数组**、并且带一批 6.0 专属字段,
        # Cocos 会直接判定版本过高、导入失败。所以这里固定写 5.5:
        #   animation 写成对象(名字 -> 数据)、skin 名固定 "default"、不写 6.0 的字段。
        if getattr(args, "cocos", False):
            try:
                import spine2dragonbones as S2D
                _db = S2D.convert_multi(
                    [{"spine": m["out"], "images": m["images"], "anim": m["action"]}
                     for m in _items],
                    os.path.join(root_dir, "cocos"), db_name,
                    version=getattr(args, "db_version", "5.5"),
                    merge_existing=False,          # 只放这次勾选的动作
                    plain=getattr(args, "db_plain", False),
                    rigid=getattr(args, "db_rigid", False),
                    rigid_slots=tuple(x.strip() for x in
                                      (getattr(args, "db_rigid_slots", "") or "").split(",")
                                      if x.strip()),
                    # 闭眼补丁别丢: 它靠"骨缩放 0.001"隐藏, 龙骨一样支持骨缩放,
                    # 倒地落地后照样能盖上去。丢了的话受击/倒地就没有闭眼。
                    drop_parts=())
                print(f"  Cocos 包 -> {os.path.dirname(_db['ske'])}  "
                      f"version={getattr(args, 'db_version', '5.5')}  "
                      f"动画 {_db['anim_names']}")

                # ---- 同时再出一份网格版, 便于和纯图片版对照 ----
                # DragonBones 的官方形态就是网格(官方样例 131 个显示全是网格,
                # 0 个带权重), 只是我们这份 Pro 5.6 还解析不对。留着它能随时对比
                # "是网格的问题还是坐标的问题"。字段按官方样例改写:
                # type 用字符串 "mesh", 且不写 width/height。
                if not getattr(args, "db_plain", False) or getattr(
                        args, "db_mesh_also", True):
                    try:
                        import db_mesh_fix
                        _mesh = S2D.convert_multi(
                            [{"spine": m["out"], "images": m["images"],
                              "anim": m["action"]} for m in _items],
                            os.path.join(root_dir, "cocos_mesh"), db_name + "_mesh",
                            version=getattr(args, "db_version", "5.5"),
                            merge_existing=False,
                            plain=False,
                            # 固定刚体四边形: 官方样例 131 个网格全是刚体、0 个带权重,
                            # 我们的加权网格在 Pro 里会被斜切
                            rigid=True,
                            rigid_slots=tuple(x.strip() for x in
                                              (getattr(args, "db_rigid_slots", "") or "").split(",")
                                              if x.strip()),
                            drop_parts=())
                        _n, _f = db_mesh_fix.to_official(_mesh["ske"])
                        try:
                            import db_mesh_from_plain
                            db_mesh_from_plain.align(root_dir)
                        except Exception as _e2:       # noqa: BLE001
                            print(f"  网格对齐跳过: {_e2}")
                        print(f"  网格版(对照用) -> {os.path.dirname(_mesh['ske'])}  "
                              f"网格显示 {_n} 个, 已按官方样例改写 {_f} 个")
                    except Exception as ex:       # noqa: BLE001
                        print(f"  网格版跳过: {ex}")
            except Exception as ex:               # noqa: BLE001
                print(f"  Cocos 包跳过: {ex}")
    except Exception as ex:                       # noqa: BLE001
        print(f"  Spine 包跳过: {ex}")


if __name__ == "__main__":
    main()
