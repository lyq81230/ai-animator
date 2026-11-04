#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spine2dragonbones.py — 把本项目生成的 Spine 3.8 JSON 转成【龙骨 DragonBones 5.5】数据。

为什么用骨骼动画而不是序列帧: 本项目的动画本来就是骨骼关键帧(rotate/translate/scale),
贴图是带权重的网格(mesh skinning), 所以转过去仍然是"骨骼驱动"的骨骼动画 —— 在龙骨里
能直接拖骨头、调曲线, 而不是一张张图切帧。

两种用法:
    单动作:   convert(result.json, images, out_dir)          -> 一个 armature, 一条动画
    多动作:   convert_multi([walk, hit, death], out_dir)     -> 一个 armature, 多条动画
              同一套拆图的不同动作共用同一副骨架, 所以只建一次骨架+插槽+蒙皮,
              把每个动作的骨骼关键帧作为独立的 animation 挂上去。龙骨里一个角色
              直接切换 待机/行走/受击/死亡, 不用导四份文件、不用换骨架。

输出:
    <name>_ske.json    骨架 + 插槽 + 网格蒙皮 + 骨骼动画(可多条)
    <name>_tex.json    图集描述(SubTexture 列表)
    <name>_tex.png     打包好的一张图集

坐标约定(两边都是 y 向下, 但本项目的中间坐标是 y 向上):
    x_db = x_spine,  y_db = -y_spine   (位置镜像)
    旋转角度【取反】: 视觉效果 = R(-θ)·(镜像后的偏移), 而龙骨算的是 R(φ)·(镜像偏移),
    所以 φ = -θ。已用顶点级比对(逐帧比世界坐标)验证过。
"""
import json
import math
import os

from PIL import Image

ATLAS_PAD = 2
DB_VERSION = "5.5"                 # 老版本 DragonBones Pro 可改成 "5.0" / "4.5"
LOOP_ANIMS = ("idle", "walk")      # 循环动作; 其余(受击/死亡)一次性播放
POW2 = (256, 512, 1024, 2048, 4096)


# ---------------------------------------------------------------- 图集打包
def pack_atlas(images, out_png, pad=ATLAS_PAD):
    """简单的货架算法: 按高度降序排, 逐行铺。

    images: {名字: (w, h, 图片绝对路径)}。多动作合并时不同动作的贴图目录可能
    不是同一个, 所以每个条目自带路径。
    返回 (w, h, {name: (x, y, w, h)})。
    """
    items = sorted(images.items(), key=lambda kv: (-kv[1][1], -kv[1][0]))
    total_area = sum(w * h for (_n, (w, h, _p)) in items)
    need = int(math.sqrt(total_area) * 1.35) + 32

    for W in [w for w in POW2 if w >= max(need, max(it[1][0] for it in items))]:
        x = y = row_h = 0
        rects = {}
        ok = True
        for name, (w, h, _p) in items:
            if w + 2 * pad > W:
                ok = False
                break
            if x + w + 2 * pad > W:            # 换行
                x = 0
                y += row_h + pad
                row_h = 0
            rects[name] = (x + pad, y + pad, w, h)
            x += w + 2 * pad
            row_h = max(row_h, h)
        H = y + row_h + 2 * pad
        if ok and H <= 4096:
            break
    else:                                       # 极端情况: 全塞进 4096
        raise RuntimeError("图集太多贴图, 超过 4096")

    H = max(64, int(math.ceil(H / 4.0) * 4))
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for name, (rx, ry, rw, rh) in rects.items():
        with Image.open(images[name][2]) as im:
            canvas.paste(im.convert("RGBA"), (rx, ry))
    canvas.save(out_png)
    return W, H, rects


# ---------------------------------------------------------------- 单动作(兼容旧接口)
def convert(spine_json, images_dir, out_dir, name=None, fps=None):
    """spine_json: result.json 路径; images_dir: 贴图目录; out_dir: 输出目录。"""
    return convert_multi([{"spine": spine_json, "images": images_dir}],
                         out_dir, name=name, fps=fps)


# ---------------------------------------------------------------- 多动作合并
def convert_multi(items, out_dir, name=None, fps=None, merge_existing=True,
                  version=DB_VERSION, plain=False, rigid=False, rigid_slots=(),
                  drop_parts=("eyes_closed",), reverse_slots=False,
                  no_zindex=False, uv_local=False):
    """把同一角色的多条动作合并成一个龙骨 armature。

    items: [{"spine": result.json 路径, "images": 贴图目录, "anim": 可选动画名}]
           anim 不给就用 spine 里自带的动画名(我们的生成器就是用动作名)。
    merge_existing: 目标文件已存在且骨架/蒙皮一致时, 把里面原有的动作也保留下来 ——
           先单独出行走、再单独出受击, 最后仍是同一个文件里的两条动画(同名以新为准)。
    reverse_slots: 把插槽数组倒过来(前->后)。有的编辑器把数组第一个当作最前面。
    no_zindex: 不写 zIndex(完全靠数组顺序)。
    drop_parts: 这些插槽不导出。默认丢掉"闭眼补丁" —— 它靠骨缩放 0.001 隐藏,
           有的编辑器(Pro 5.6)对网格不结算骨骼缩放, 补丁会整块盖在脸上。
    rigid: True 时导出"刚性网格"(4 顶点四边形, 无权重) —— 与官方 Pro 导出同一种形式
           (官方示例 131 个显示全是刚体网格, 0 个 weighted mesh), 兼容性最好且保留网格。
    plain: True 时不用带权重的网格, 每个部件导出成一张"贴在某根骨上的图"(image)。
           不会有机件的网格形变(膝盖/头发是整块跟着转), 但兼容性最好 ——
           编辑器里导网格有问题时用这个先跑起来。
    uv_local: uv 写成"相对该部件图"的 0~1(不换算成图集全局坐标)。
           标准 DragonBones 要全局 uv(默认); LoongBones(龙骨) 的网格按局部 uv 采样,
           给它导时加这个开关, 否则每个部件会采到图集上别的位置。
    返回 dict: ske/tex/png 路径, bones/slots/anims 数量, atlas 尺寸, skipped 被跳过的动作。
    """
    items = [it if isinstance(it, dict) else {"spine": it[0], "images": it[1]}
             for it in items]
    if not items:
        raise RuntimeError("没有要转换的动作")

    docs = []
    for it in items:
        p = it["spine"]
        if not os.path.isfile(p):
            raise RuntimeError(f"缺文件: {p}")
        docs.append({"doc": json.load(open(p, encoding="utf-8")), "path": p,
                     "images": it["images"], "anim": it.get("anim")})

    base = docs[0]["doc"]
    name = name or os.path.splitext(os.path.basename(docs[0]["path"]))[0]
    fps = fps or int(round(base.get("skeleton", {}).get("fps") or 24)) or 24
    cw = float(base["skeleton"].get("width") or 0)
    ch = float(base["skeleton"].get("height") or 0)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 骨骼(所有动作共用) ----
    bones = base["bones"]
    db_bones = []
    for b in bones:
        t = {"x": float(b.get("x", 0.0)), "y": -float(b.get("y", 0.0))}
        rot = -float(b.get("rotation", 0.0))
        if abs(rot) > 1e-6:
            # 5.x 的 setup 变换里旋转字段是 skX/skY(度), 不是 rotate。
            # 运行时两种都认(先看 rotate/skew), 但老的编辑器/导入器只读 skX/skY,
            # 写 rotate 会被当成 0 度 —— 整副骨架变成"所有骨都朝右"。
            t["skX"] = rot
            t["skY"] = rot
        db = {"name": b["name"], "length": 0, "transform": t}   # length 稍后算
        if b.get("parent"):
            db["parent"] = b["parent"]
        db_bones.append(db)

    # 坐标原点: spine 是 y 向上(0..ch), 龙骨是 y 向下且原点在左上角。
    # 只取反的话整个角色会落在 y<0 的区域 —— 编辑器里常常表现为"导入成功但舞台空白"。
    # 这里把根骨整体下移 ch, 让内容落在 0..ch 的可视范围内(官方导出也是这样)。
    for b in db_bones:
        if not b.get("parent"):
            b["transform"]["y"] = float(b["transform"].get("y", 0.0)) + ch

    # 绑定姿势下没有旋转, 世界坐标 = 沿链累加局部平移 (y 取反)
    bind_world = {}

    def bind_pos(bname):
        if bname in bind_world:
            return bind_world[bname]
        b = next(x for x in db_bones if x["name"] == bname)
        t = b["transform"]
        par = b.get("parent")
        px, py = bind_pos(par) if par else (0.0, 0.0)
        bind_world[bname] = (px + t["x"], py + t["y"])
        return bind_world[bname]

    for b in db_bones:
        bind_pos(b["name"])

    # 骨骼长度: 龙骨的骨是"一条有长度的线", length=0 在编辑器里根本看不见。
    # 取"到第一个子骨的距离", 没有子骨就用"父骨到自己"的距离, 至少 6。
    children = {}
    for b in db_bones:
        if b.get("parent"):
            children.setdefault(b["parent"], []).append(b["name"])
    for b in db_bones:
        bx, by = bind_world[b["name"]]
        kids = children.get(b["name"]) or []
        if kids:
            kx, ky = bind_world[kids[0]]
            ln = math.hypot(kx - bx, ky - by)
        else:
            par = b.get("parent")
            if par:
                px2, py2 = bind_world[par]
                ln = math.hypot(bx - px2, by - py2)
            else:
                ln = 0.0
        b["length"] = round(min(40.0, max(8.0, ln)), 2)

    # 骨缩放是【乘性】的: 最终缩放 = setup(scX/scY) × 动画 scale。
    # 靠"缩到 0"隐藏的部件(闭眼补丁)如果 setup=1, 编辑器停在绑定姿态时会露出来
    # (看起来就是"一导入角色就闭着眼")。所以这里把 setup 写成 0.001,
    # 同时把该骨所有动画的 scale 关键帧乘 1000 —— 隐藏时 0.001×1=0.001(看不见),
    # 需要露出时 0.001×1000=1。两边都不受影响。
    HIDE = 0.001
    hidden = set()
    for dd in docs:                      # 每个动作是独立的 spine 文档, 都要折算
        for an in (dd["doc"].get("animations") or {}).values():
            for bname, tl in (an.get("bones") or {}).items():
                fr = tl.get("scale") or []
                if fr and float(fr[0].get("x", 1.0)) <= 0.01:
                    hidden.add(bname)
                    for kf in fr:
                        kf["x"] = round(float(kf.get("x", 1.0)) / HIDE, 4)
                        kf["y"] = round(float(kf.get("y", 1.0)) / HIDE, 4)
    for bname in sorted(hidden):
        for b in db_bones:
            if b["name"] == bname:
                b["transform"]["scX"] = HIDE
                b["transform"]["scY"] = HIDE
        print(f"  隐藏部件骨 {bname}: setup 缩放 {HIDE}, 动画关键帧已折算")

    # ---- 插槽 / 网格(所有动作共用) ----
    slots = base["slots"]
    att = base["skins"][0]["attachments"]
    used_imgs = {}
    db_slots, db_skin_slots = [], []
    # mesh 的 uvs 必须先写成"相对自身部件图"的 0~1, 图集打包完再换算成
    # 【图集全局 uv】。DragonBones/LoongBones 都要求 uv 落在该 SubTexture
    # 在 atlas 里的矩形内, 否则会把整个 slot 判成贴图缺失(红框 Missing)。
    uv_jobs = []
    for z, s in enumerate(slots):
        sname = s["name"]
        a = att.get(sname)
        if not a:
            continue
        aname, ad = next(iter(a.items()))
        if sname in (drop_parts or ()):
            continue
        png = ad.get("path") or (aname + ".png")
        tex_key = os.path.splitext(png)[0]     # 图集里的键(不带 .png), 官方风格
        verts, uvs, tris = ad.get("vertices") or [], ad.get("uvs") or [], \
            ad.get("triangles") or []
        n_uv = len(uvs) // 2
        raw, weights, bpose_ids = _mesh_vertices(verts, db_bones, bind_world)
        slot_parent = "root"

        # 权重合计最大的那根骨 = 这块网格的"主骨": 决定 slot 父骨与 slotPose 基准。
        wsum = {}
        i2 = 0
        while i2 < len(verts):
            bc = int(verts[i2]); i2 += 1
            for _ in range(max(0, bc)):
                bi = int(verts[i2]); w = float(verts[i2 + 3]); i2 += 4
                wsum[bi] = wsum.get(bi, 0.0) + w
        main_bi = max(wsum, key=wsum.get) if wsum else 0
        main_bone = db_bones[main_bi]["name"]
        sbx, sby = bind_world[main_bone]

        if rigid or (sname in rigid_slots):
            # 刚性四边形网格: 顶点 = 部件外框在【骨局部空间】的四个角, uv = 对应 uv 外框。
            # 不带 weights/slotPose/bonePose —— 与官方 Pro 导出完全同构。
            bname = main_bone
            bx0, by0 = sbx, sby
            xs = raw[0::2]; ys = raw[1::2]
            x0, x1 = min(xs) - bx0, max(xs) - bx0
            y0, y1 = min(ys) - by0, max(ys) - by0
            ux = uvs[0::2]; uy = uvs[1::2]
            u0, u1 = min(ux), max(ux)
            v0, v1 = min(uy), max(uy)
            # width/height 必须写: DragonBones(Pro) 会把它当作网格外框去"适配",
            # 缺了它会按默认值缩放 -> 整块被缩小并挪位(实测头部一组缩到 ~0.6 倍浮在肩上)。
            disp = {"name": tex_key, "path": tex_key, "type": 2,
                    "width": round(x1 - x0, 2), "height": round(y1 - y0, 2),
                    "vertices": [round(x0, 3), round(y0, 3),
                                 round(x1, 3), round(y0, 3),
                                 round(x1, 3), round(y1, 3),
                                 round(x0, 3), round(y1, 3)],
                    "uvs": [round(u0, 5), round(v0, 5),
                            round(u1, 5), round(v0, 5),
                            round(u1, 5), round(v1, 5),
                            round(u0, 5), round(v1, 5)],
                    "triangles": [0, 1, 2, 0, 2, 3]}   # 官方网格不带 width/height
            slot_parent = bname
            # 注意: 这里的四边形只覆盖**内容包围盒**(min/max of raw), 不是整张部件图,
            # 所以 uv 必须保持"内容范围"的 u0..u1 —— 早先这里写成整张图四角(_uv_full),
            # 结果内容被拉伸到整张图上(头饰 97x45 的实际内容只有 50x25 -> 放大 2 倍),
            # 头/五官/头饰在 DragonBones 里就会糊成一团。
            disp["_uv_keep"] = True
        elif plain:
            # 简化模式: 每个部件一张图, 贴到"权重最大的那根骨"上, 偏移 = 部件中心 - 骨位置。
            # 没有任何网格/权重, 任何版本的编辑器都能画出来, 只是部件不会跟着权重弯。
            bname = main_bone
            bx0, by0 = sbx, sby
            # 用"部件整张图"的中心当锚点(图片显示画的是整张区域图, 包括四周透明边),
            # 所以要把内容包围盒按 uv 边距外扩回整张图, 再取中心 —— 否则头饰这种
            # "内容偏向一侧"的部件会被整体挪位。
            _xs = raw[0::2]; _ys = raw[1::2]
            _x0, _x1 = min(_xs), max(_xs)
            _y0, _y1 = min(_ys), max(_ys)
            _u = uvs[0::2]; _v = uvs[1::2]
            _u0, _u1 = min(_u), max(_u)
            _v0, _v1 = min(_v), max(_v)
            if (_u1 - _u0) > 1e-4 and (_v1 - _v0) > 1e-4:
                _rx0 = _x0 - _u0 / (_u1 - _u0) * (_x1 - _x0)
                _rx1 = _x1 + (1.0 - _u1) / (_u1 - _u0) * (_x1 - _x0)
                _ry0 = _y0 - _v0 / (_v1 - _v0) * (_y1 - _y0)
                _ry1 = _y1 + (1.0 - _v1) / (_v1 - _v0) * (_y1 - _y0)
                cx0 = (_rx0 + _rx1) / 2.0
                cy0 = (_ry0 + _ry1) / 2.0
            else:
                cx0 = (_x0 + _x1) / 2.0
                cy0 = (_y0 + _y1) / 2.0
            disp = {"name": tex_key, "path": tex_key, "type": 0,
                    "transform": {"x": round(cx0 - bx0, 2),
                                  "y": round(cy0 - by0, 2),
                                  "pivot": {"x": 0.5, "y": 0.5}}}
            slot_parent = bname
        else:
            # width/height = 网格实际外框(与 uvs 覆盖的内容范围一致), 不要写整张部件图 ——
            # 两者不一致时编辑器会按 width/height 去缩放宽高比。多数部件内容刚好铺满, 但
            # 腿这类左右留白的部件会差 1.3 倍。
            _xs = raw[0::2]; _ys = raw[1::2]
            disp = {"name": aname, "path": tex_key, "type": 2,
                    "width": round(max(_xs) - min(_xs), 2),
                    "height": round(max(_ys) - min(_ys), 2)}
            disp["uvs"] = [round(float(v), 5) for v in uvs]
            disp["triangles"] = [int(t) for t in tris]
            # 顶点必须写在【插槽主骨的局部空间】, slotPose 写这根骨的绑定矩阵 ——
            # 这是官方导出的写法。以前为了"少一个基准", 把顶点直接写成世界坐标 +
            # slotPose 单位矩阵: 合成式 W·bonePose⁻¹·slotPose·v 在数学上等价,
            # 但编辑器/导入器会拿 width/height 当网格外框去适配(顶点是世界坐标,
            # 外框比 width/height 大得多), 于是整块网格被缩小 + 挪位,
            # 表现就是"骨骼和贴图分家了"。
            disp["slotPose"] = [1, 0, 0, 1, round(sbx, 3), round(sby, 3)]
            disp["vertices"] = [round(float(v) - (sbx if k % 2 == 0 else sby), 3)
                                for k, v in enumerate(raw)]
            slot_parent = main_bone
            if weights:
                disp["weights"] = [int(w) if isinstance(w, int) else round(w, 4)
                                   for w in weights]
                bp = []
                for bi in bpose_ids:                  # 必须写"绑定世界坐标", 不是局部偏移
                    wx, wy = bind_world[db_bones[bi]["name"]]
                    bp += [bi, 1, 0, 0, 1, round(wx, 3), round(wy, 3)]
                disp["bonePose"] = bp
            if len(disp["vertices"]) // 2 != n_uv:
                raise RuntimeError(
                    f"{sname}: 顶点 {len(disp['vertices'])//2} 与 uv {n_uv} 数量不符")

        # uvs 此刻还是"相对部件图"的 0~1, 打包后再换算成图集全局坐标
        if disp.get("uvs"):
            uv_jobs.append((disp, png, [float(v) for v in disp["uvs"]]))

        # 不写 z: 官方导出没有这个字段, 层级按 slot 数组顺序(我们的顺序就是 z 序)
        db_slots.append({"name": sname, "parent": slot_parent, "displayIndex": 0,
                         "zIndex": 500 + len(db_slots)})
        db_skin_slots.append({"name": sname, "display": [disp]})
        # 贴图: 谁有这个文件就用谁的目录(各动作的 images 是同一套部件的拷贝)
        for it in docs:
            p = os.path.join(it["images"], png)
            if os.path.isfile(p):
                with Image.open(p) as im:
                    used_imgs[png] = (im.size[0], im.size[1], p)
                break
        else:
            raise RuntimeError(f"缺贴图: {png} (在 {docs[0]['images']} 等目录都没找到)")

    # ---- 图集 ----
    tex_name = name + "_tex"
    W, H, rects = pack_atlas(used_imgs, os.path.join(out_dir, tex_name + ".png"))

    # ---- uv 局部 → 图集全局 ----
    # 之前把 spine 的 0~1(相对部件图) 直接当图集 uv 写出去, 等于每个网格都采样
    # 图集左上角那块; 龙骨的编辑器看到 uv 落在自己 SubTexture 之外, 就整片报
    # "Missing"。这里按 SubTexture 的 x/y/w/h 做一次线性映射。
    out_of_range = []
    for disp, key, local in uv_jobs:
        rx, ry, rw, rh = rects[key]
        disp.pop("_uv_keep", False)      # 内容范围 uv: 两种模式都原样保留
        if uv_local:
            # 保留相对部件图的 0~1(刚性网格用四角); LoongBones 就吃这一套
            if disp.pop("_uv_full", False):
                disp["uvs"] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
            continue
        if disp.pop("_uv_full", False):
            # 整张图: 直接用 SubTexture 的四角, 不留 1px 缝
            disp["uvs"] = [round(rx / float(W), 6), round(ry / float(H), 6),
                           round((rx + rw) / float(W), 6), round(ry / float(H), 6),
                           round((rx + rw) / float(W), 6), round((ry + rh) / float(H), 6),
                           round(rx / float(W), 6), round((ry + rh) / float(H), 6)]
            continue
        out = []
        for i in range(0, len(local), 2):
            u = (rx + local[i] * rw) / float(W)
            v = (ry + local[i + 1] * rh) / float(H)
            out.append(round(min(1.0, max(0.0, u)), 6))
            out.append(round(min(1.0, max(0.0, v)), 6))
        disp["uvs"] = out
        if min(out[0::2]) < 0.0 or max(out[0::2]) > 1.0:
            out_of_range.append(key)
    if out_of_range:
        raise RuntimeError(f"uv 越界: {sorted(set(out_of_range))}")
    # 图集名必须和骨架同名(文件名仍叫 <name>_tex.json)。官方导出就是这样:
    # 骨架 name=mecha_1502b, 图集 name=mecha_1502b, 文件 mecha_1502b_tex.json。
    # 名字不一致时龙骨 Pro 会找不到对应骨架/图集。
    # frame 系列字段官方导出总会写(即使没裁剪): 老解析器靠它算贴图原始尺寸,
    # 缺了会被当成 -1, 尺寸算错。
    tex = {"name": name, "imagePath": tex_name + ".png",
           "width": W, "height": H, "SubTexture": [
               {"name": os.path.splitext(n)[0], "x": r[0], "y": r[1],
                "width": r[2], "height": r[3],
                "frameX": 0, "frameY": 0,
                "frameWidth": r[2], "frameHeight": r[3]}
               for n, r in rects.items()]}

    # ---- 动画(每个动作一条, 全部挂在同一个 armature 上) ----
    db_anims, skipped, seen = [], [], {}
    for it in docs:
        d = it["doc"]
        if not _same_skeleton(d, base):
            skipped.append(os.path.basename(it["path"]))
            continue
        for aname, a in (d.get("animations") or {}).items():
            nm = it.get("anim") or aname
            if nm in seen:                        # 重名 -> 加后缀, 龙骨里不能有同名动画
                seen[nm] += 1
                nm = f"{nm}_{seen[nm]}"
            else:
                seen[nm] = 1
            db_anims.append(_db_anim(nm, a, fps, loop=(nm in LOOP_ANIMS)))

    if reverse_slots:
        db_slots.reverse()
        db_skin_slots.reverse()
    if no_zindex:
        for s_ in db_slots:
            s_.pop("zIndex", None)

    # 官方 5.5 导出: skin 名是 "default"; animation 是【对象】(名字 -> 数据),
    # 数组写法是 6.0 才有的。LoongBones(6.0 内部分支) 对 5.5 文件按对象读,
    # 给数组会解析失败/导不进去, 所以这里按 5.5 规范写。
    def _anim_obj(anims):
        return {a["name"]: {k: v for k, v in a.items() if k != "name"} for a in anims}

    armature = {"type": "Armature", "frameRate": fps, "name": name,
                "aabb": {"x": 0, "y": 0, "width": cw, "height": ch},
                "bone": db_bones, "slot": db_slots,
                "skin": [{"name": "default", "slot": db_skin_slots}], "ik": [],
                "animation": _anim_obj(db_anims)}

    ske_p = os.path.join(out_dir, name + "_ske.json")

    # ---- 合并已有文件里的动作(同一个角色反复生成时累积到同一份数据) ----
    if merge_existing and os.path.isfile(ske_p) and db_anims:
        try:
            old = json.load(open(ske_p, encoding="utf-8"))
            oa = old["armature"][0]
            if _arm_sig(oa) == _arm_sig(armature):
                new_names = {a["name"] for a in db_anims}
                _old_an = oa.get("animation") or {}
                if isinstance(_old_an, list):
                    _old_items = _old_an
                else:
                    _old_items = [{"name": k, **v} for k, v in _old_an.items()]
                keep = [a for a in _old_items if a.get("name") not in new_names]
                if keep:
                    db_anims = keep + db_anims
                    armature["animation"] = _anim_obj(db_anims)
                    print(f"  合并已有动作: 保留 {[a['name'] for a in keep]}, "
                          f"共 {len(db_anims)} 条")
        except (OSError, ValueError, KeyError, TypeError):
            pass

    if db_anims:
        armature["defaultActions"] = [{"gotoAndPlay": db_anims[0]["name"]}]

    # 官方 5.5 导出的顶层字段是 name/version/frameRate/isGlobal/armature,
    # 没有 compatibleVersion(规范里是可选的), 这里跟着官方写, 少一处被挑刺的机会。
    ske = {"name": name, "version": version, "frameRate": fps,
           "isGlobal": 1, "armature": [armature]}
    tex_p = os.path.join(out_dir, tex_name + ".json")
    json.dump(ske, open(ske_p, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(tex, open(tex_p, "w", encoding="utf-8"), ensure_ascii=False)
    return {"ske": ske_p, "tex": tex_p, "png": os.path.join(out_dir, tex_name + ".png"),
            "bones": len(db_bones), "slots": len(db_slots), "anims": len(db_anims),
            "anim_names": [a["name"] for a in db_anims],
            "atlas": (W, H), "skipped": skipped}


# ---------------------------------------------------------------- 工具
def _arm_sig(arm):
    """骨架+蒙皮指纹: 骨/插槽/每个网格的顶点与三角形数量。用来判断能否合并动画。"""
    skin = []
    for sk in arm.get("skin", []):
        for s in sk.get("slot", []):
            for d in s.get("display", []):
                skin.append((s.get("name"), d.get("path"),
                             len(d.get("vertices") or []),
                             len(d.get("uvs") or []),
                             len(d.get("triangles") or [])))
    return ([(b.get("name"), b.get("parent")) for b in arm.get("bone", [])],
            [s.get("name") for s in arm.get("slot", [])], skin)


def _same_skeleton(d, base):
    """两条动作的骨架/插槽必须完全一致, 否则不能共用同一份蒙皮。"""
    def sig(x):
        return ([(b["name"], b.get("parent"), round(float(b.get("x", 0)), 4),
                  round(float(b.get("y", 0)), 4), round(float(b.get("rotation", 0)), 4))
                 for b in x["bones"]],
                [s["name"] for s in x["slots"]])
    return sig(d) == sig(base)


def _db_anim(name, a, fps, loop=True):
    """一条 spine 动画 -> 龙骨 animation 对象(骨骼关键帧)。"""
    frames = _anim_len(a, fps)
    out = {"name": name, "duration": frames, "playTimes": 0 if loop else 1}
    bframes = []
    for bname, tl in (a.get("bones") or {}).items():
        entry = {"name": bname}
        if tl.get("translate"):
            entry["translateFrame"] = _frames(tl["translate"], fps, frames,
                                              lambda k: {"x": round(k.get("x", 0), 3),
                                                         "y": round(-k.get("y", 0), 3)})
        if tl.get("rotate"):
            entry["rotateFrame"] = _frames(tl["rotate"], fps, frames,
                                           lambda k: {"rotate": round(-k.get("angle", 0), 3),
                                                      "skew": 0})
        if tl.get("scale"):
            entry["scaleFrame"] = _frames(tl["scale"], fps, frames,
                                          lambda k: {"x": round(k.get("x", 1), 4),
                                                     "y": round(k.get("y", 1), 4)})
        if len(entry) > 1:
            bframes.append(entry)
    if bframes:
        out["bone"] = bframes
    return out


def _mesh_vertices(verts, db_bones, bind_world):
    """Spine 加权网格 [n,(boneIdx,lx,ly,w)...] -> 龙骨 (raw顶点, weights, bonePose索引)。

    weights 的排布必须是【逐顶点交错】的:
        [该顶点骨数, (骨索引, 权重) × 骨数, 下一顶点骨数, ...]
    不是"所有骨数写在前面, 所有 (索引,权重) 对写在后面"。龙骨运行时是顺序读的
    (顶点数 -> 骨头数 -> 若干对 -> 下一个顶点), 写成块状会让每个顶点都领到
    别人的骨和权重, 表现是"贴图炸开 / 骨骼和网格对不上"。
    """
    raw, seq = [], []
    used = []
    i, n = 0, len(verts)
    while i < n:
        bc = int(verts[i]); i += 1
        cnt = max(0, bc)
        seq.append(cnt)
        vx = vy = 0.0
        for _ in range(cnt):
            bi, lx, ly, w = (int(verts[i]), float(verts[i + 1]),
                             float(verts[i + 2]), float(verts[i + 3]))
            i += 4
            bx, by = bind_world[db_bones[bi]["name"]]
            wx = bx + lx                 # 骨的绑定世界坐标 + 顶点局部偏移
            wy = by + (-ly)              # ly 是 spine(y上) 局部, 换到 y下
            vx += wx * w
            vy += wy * w
            seq += [bi, round(w, 4)]
            if bi not in used:
                used.append(bi)
        raw += [round(vx, 3), round(vy, 3)]
    return raw, (seq if used else []), sorted(used)


def _frames(keys, fps, total, mk):
    """Spine 关键帧 [{time,...}] -> 龙骨帧 [{duration, tweenEasing, ...}]。"""
    idx = []
    for k in keys:
        f = int(round(float(k.get("time", 0)) * fps))
        idx.append(max(0, min(total, f)))
    out = []
    for j, k in enumerate(keys):
        f = idx[j]
        nxt = idx[j + 1] if j + 1 < len(idx) else total
        dur = max(0, nxt - f)
        if j + 1 == len(keys):
            # 官方导出的最后一个关键帧是 duration 0 的"终点帧";
            # 只有它没落在动画末尾时才需要补足到 duration。
            dur = 0 if f >= total else max(1, total - f)
        fr = {"duration": dur, "tweenEasing": 0}
        fr.update(mk(k))
        out.append(fr)
    return out


def _anim_len(a, fps):
    mx = 0.0
    for tl in (a.get("bones") or {}).values():
        for ch in ("translate", "rotate", "scale"):
            for k in tl.get(ch) or []:
                mx = max(mx, float(k.get("time", 0)))
    return max(1, int(round(mx * fps)))


def write_probe(out_dir, name="probe"):
    """写一个最小的"探针"骨架(1 骨 + 1 图 + 1 动画), 用来判断导入问题出在哪。

    三件套和正式产物完全同格式: 能导入探针 = 格式没问题, 那问题就在网格/图集;
    连探针都导不进 = 编辑器版本或导入方式的问题。
    """
    os.makedirs(out_dir, exist_ok=True)
    W = H = 64
    ske = {
        "name": name, "version": DB_VERSION, "frameRate": 24, "isGlobal": 1,
        "armature": [{
            "type": "Armature", "frameRate": 24, "name": name,
            "aabb": {"x": 0, "y": 0, "width": W, "height": H},
            "defaultActions": [{"gotoAndPlay": "idle"}],
            "bone": [{"name": "root", "length": 32, "transform": {"x": 32, "y": 32}}],
            "slot": [{"name": "img", "parent": "root", "displayIndex": 0}],
            "skin": [{"name": "", "slot": [{"name": "img", "display": [{
                "name": "probe", "path": "probe", "type": 0,
                "transform": {"x": 0, "y": 0}}]}]}],
            "ik": [],
            "animation": [{
                "name": "idle", "duration": 48, "playTimes": 0,
                "bone": [{
                    "name": "root",
                    "rotateFrame": [
                        {"duration": 24, "tweenEasing": 0, "rotate": 0},
                        {"duration": 24, "tweenEasing": 0, "rotate": 30},
                        {"duration": 0, "tweenEasing": 0, "rotate": 0}],
                    "translateFrame": [
                        {"duration": 24, "tweenEasing": 0, "x": 0, "y": 0},
                        {"duration": 24, "tweenEasing": 0, "x": 0, "y": -20},
                        {"duration": 0, "tweenEasing": 0, "x": 0, "y": 0}],
                }],
            }],
        }],
    }
    tex = {"name": name, "imagePath": name + "_tex.png", "width": 64, "height": 64,
           "SubTexture": [{"name": "probe", "x": 0, "y": 0,
                           "width": W, "height": H,
                           "frameX": 0, "frameY": 0,
                           "frameWidth": W, "frameHeight": H}]}
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, W - 1, H - 1], fill=(90, 190, 120, 255),
                 outline=(20, 90, 50, 255))
    dr.line([4, H // 2, W - 5, H // 2], fill=(20, 40, 90, 255), width=3)
    img.save(os.path.join(out_dir, name + "_tex.png"))

    ske_p = os.path.join(out_dir, name + "_ske.json")
    tex_p = os.path.join(out_dir, name + "_tex.json")
    json.dump(ske, open(ske_p, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(tex, open(tex_p, "w", encoding="utf-8"), ensure_ascii=False)
    return {"ske": ske_p, "tex": tex_p,
            "png": os.path.join(out_dir, name + "_tex.png")}


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Spine JSON -> 龙骨骨骼动画; 可以一次传多个动作合并成一个 armature")
    ap.add_argument("--spine", action="append", default=[],
                    help="result.json, 可重复传多个(每个一条动画)")
    ap.add_argument("--images", default="", help="贴图目录; 多个动作时用 --images-of")
    ap.add_argument("--images-of", action="append", default=[],
                    help="与 --spine 一一对应的贴图目录, 可重复")
    ap.add_argument("--anim", action="append", default=[],
                    help="动画名, 与 --spine 一一对应; 不给就用 json 里的名字")
    ap.add_argument("--out", default="", help="输出目录, 默认第一个 json 同级的 dragonbones")
    ap.add_argument("--name", default="")
    ap.add_argument("--probe", default="",
                    help="只写一个最小探针骨架到该目录(1 骨 1 图 1 动画), 用来排查导入问题")
    ap.add_argument("--reverse-slots", action="store_true",
                    help="插槽数组倒序(前->后), 并去掉 zIndex")
    ap.add_argument("--rigid", action="store_true",
                    help="导出刚性网格(4 顶点四边形, 无权重): 与官方 Pro 导出同一种形式")
    ap.add_argument("--plain", action="store_true",
                    help="导出无网格版本: 每个部件一张图贴在骨上(兼容性最好, 没有网格形变)")
    ap.add_argument("--uv-local", action="store_true",
                    help="uv 写成相对部件图的 0~1(不换算图集全局); LoongBones 需要这个")
    ap.add_argument("--version", default=DB_VERSION,
                    help="数据版本(默认 5.5); 老版 Pro 可试 5.0 / 4.5")
    a = ap.parse_args()

    if a.probe:
        info = write_probe(a.probe, a.name or "probe_v2")
        print(f"探针: {info['ske']}")
        print(f"      {info['tex']} + {info['png']}")
        print("      先在龙骨里导这个: 能显示骨头和动画 = 格式没问题, "
              "问题在正式数据的网格/图集")
        return

    ps = a.spine
    imgs = a.images_of or ([a.images] if a.images else [])
    if not imgs:
        imgs = [os.path.join(os.path.dirname(os.path.abspath(p)), "images") for p in ps]
    if len(imgs) == 1 and len(ps) > 1:
        imgs = imgs * len(ps)
    items = []
    for i, p in enumerate(ps):
        items.append({"spine": p,
                      "images": imgs[i] if i < len(imgs) else imgs[-1],
                      "anim": a.anim[i] if i < len(a.anim) else None})
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(ps[0])), "dragonbones")
    info = convert_multi(items, out, a.name or None, version=a.version,
                         plain=a.plain, rigid=a.rigid,
                         reverse_slots=a.reverse_slots, no_zindex=a.reverse_slots,
                         uv_local=a.uv_local)
    print(f"龙骨: {info['ske']}")
    print(f"      骨骼 {info['bones']} 根, 插槽 {info['slots']} 个, "
          f"动画 {info['anims']} 条 {info['anim_names']}, "
          f"图集 {info['atlas'][0]}x{info['atlas'][1]}")
    if info["skipped"]:
        print(f"      跳过(骨架与第一个不一致): {info['skipped']}")


if __name__ == "__main__":
    main()
