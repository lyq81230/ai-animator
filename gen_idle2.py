#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_idle2.py — Idle 待机 (纯程序化, 零模型/零视频依赖, 可直接封装进 App)。

输入: coords.json (图层名+位置+网格) + parts PNG。无需 CoTracker/tracks/视频。

运动设计:
  * 身体/肢体: 仅呼吸 (胸口缩放 + 极轻浮动), 无左右摇摆。
  * 面部: 五官刚性跟随脸部 (相对位置锁定, 不错位); 眼睛只眨眼; 嘴不张。
  * 头发/飘带: 独立风场 (多频 + 阵风包络 + 按 x 相位的行波), 阻尼弹簧驱动
               绕顶端枢轴摆; 飘带更软。
  * 小鸟: 隐藏。

所有信号整周期 -> 无缝循环。确定性, 参数即 UI 滑块。
"""
import argparse
import json
import math
import os
import shutil

import numpy as np
from PIL import Image

from video2spine import build_grid_mesh, decimate, trim_deform, video_to_spine


# ---- 部件分类: 同时支持 see-through 英文语义名 与 中文拼音名 (关键词包含式) ----
# 风吹: 头发/飘带/裙摆/围巾/披风/缎带等
HAIR_KW = ("hair", "toufa", "fagu", "fasheng", "bang", "braid", "ponytail",
           "fringe", "liuhai")
RIBBON_KW = ("ribbon", "skirt", "scarf", "cape", "cloak", "tail", "sash",
             "drape", "shawl", "tassel", "qun", "ershi", "lianzi", "zhulian",
             "leisi", "shipin", "hua", "piaodai", "weiba", "doupeng")
EYE_KW = ("eyewhite", "irides", "iris", "pupil", "eyeball", "sclera",
          "yanbai", "tongkong", "yanjian", "yanpi", "yanzhu", "yanqiu",
          "shangyanpi", "xiayanpi", "eye")   # 注意排除 eyebrow/eyelash/biyan
EYE_EXCLUDE = ("eyebrow", "eyelash", "eyewear")
# 闭眼图层(直接 attachment 切换, 不挤压): biyan=闭眼。官方 youbiyan/zuobiyan 高频
BLINK_KW = ("biyan", "closedeye", "eyeclose", "closeeye", "binyan")
BROW_KW = ("eyebrow", "brow", "meimao", "_mei", "meimaoyou", "meimaozuo")
LASH_KW = ("eyelash", "jiemao", "lash")
# 手臂(官方词典): da/xiao/shang-bi=大/小/上-臂, xiuzi/xiukou=袖, shoubi=手臂
ARM_UPPER_KW = ("upperarm", "arm_upper", "dabi", "shangbi", "shoulder",
                "shoudabi")
ARM_FORE_KW = ("forearm", "arm_fore", "xiaobi", "xiukou", "xiuzi", "shoubi",
               "sleeve")
ARM_HAND_KW = ("hand", "shouzhi", "shouzhang", "shouwan", "shoutao",
               "jiezhi", "finger", "handwear", "glove",
               "shizhi", "zhongzhi", "wumingzhi", "xiaomuzhi", "damuzhi",
               "muzhi")
FACE_KW = ("face", "nose", "mouth", "lian", "bizi", "zui",
           "lip", "cheek", "head", "irides_extra")
EAR_KW = ("ears", "_er", "er2", "erduo")           # 耳朵(避免 topw-ear 误匹配)
BODY_KW = ("topwear", "body", "shenti", "yifu", "torso", "chest", "dress",
           "bottomwear", "pants", "leg", "neck", "bozi")
HEADWEAR_KW = ("headwear", "toushi", "maozi", "hat", "headgear")  # 头饰: 随头转+可飘


def _has(n, kws):
    return any(k in n for k in kws)


def classify(name, is_bird):
    """语义分类。优先级: bird > 闭眼 > 眼/眉/睫 > 头发/飘带 > 手臂 > 面部 > body。
    名称大小写不敏感, 兼容英文(see-through)与中文拼音(G92 官方命名)。"""
    n = name.lower()
    if is_bird:
        return "bird"
    # 闭眼图层 (biyan): 直接 attachment 切换播眨眼, 不挤压。须先于 eye(biyan 含 yan)
    if _has(n, BLINK_KW):
        return "blink_eye"
    # 眼部 (排除 eyebrow/eyelash/eyewear)
    if _has(n, EYE_KW) and not _has(n, EYE_EXCLUDE):
        return "eye"
    if _has(n, BROW_KW):
        return "brow"
    if _has(n, LASH_KW):
        return "lash"
    # 风吹: 头发 / 飘带裙摆 (handwear 不算飘带, 先于通用匹配排除)
    if _has(n, ARM_HAND_KW):
        return "arm_hand"
    if _has(n, HEADWEAR_KW):       # 头饰(发带/帽穗): 归 hair, 上部随头转下部飘
        return "hair"
    if _has(n, HAIR_KW):
        return "hair"
    if _has(n, RIBBON_KW):
        return "ribbon"
    if _has(n, ARM_FORE_KW):       # 先小臂/袖(xiaobi/xiuzi), 再大臂(_bi 易误匹配)
        return "arm_fore"
    if _has(n, ARM_UPPER_KW):
        return "arm_upper"
    if _has(n, BODY_KW):           # 衣服/躯干 显式归 body (先于 face, 避免 topw-ear)
        return "body"
    # 纯"手"命名(shou1/youshou/zuoshou) 归手部 —— 放 body 之后避免吞 shouxxx
    if _has(n, ("shou", "_shou")):
        return "arm_hand"
    if _has(n, EAR_KW) or _has(n, FACE_KW):
        return "face"
    return "body"


def _alpha_bbox(im, thresh=110):
    """不透明像素紧致 bbox (l,t,r,b); 全透明返回 None。
    see-through 输出常为满画布 PNG 且带低 alpha 噪声铺满全图,
    故用较高阈值取主体不透明像素, 还原部件真实位置/尺寸。"""
    a = np.asarray(im.convert("RGBA"))[:, :, 3]
    ys, xs = np.where(a > thresh)
    if len(xs) == 0:                       # 阈值过高无像素: 退回低阈值
        ys, xs = np.where(a > 8)
        if len(xs) == 0:
            return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def spring_follow(target, fps, k, c, loop=True):
    """阻尼弹簧: 滞后 + 过冲的二次运动。
    loop=True(默认): target 视为整周期信号, 把它重复多遍跑积分消除初始瞬态,
    取最后一周期 -> 输出首尾闭合(seamless loop), 否则单遍积分会因弹簧瞬态
    让循环首尾不连续而跳帧。"""
    dt = 1.0 / fps
    if loop and len(target) > 2:
        period = target[:-1]                 # 去重复末点(末=首)
        P = len(period)
        reps = 4                             # 跑 4 周期, 末遍已稳态
        x = v = 0.0
        out = np.empty(P)
        for r in range(reps):
            last = (r == reps - 1)
            for i in range(P):
                v += (k * (period[i] - x) - c * v) * dt
                x += v * dt
                if last:
                    out[i] = x
        return np.append(out, out[0])        # 闭合: 末帧=首帧
    x = v = 0.0
    out = np.empty(len(target))
    for i, tg in enumerate(target):
        v += (k * (tg - x) - c * v) * dt
        x += v * dt
        out[i] = x
    return out


def rot(pts, piv, ang):
    if abs(ang) < 1e-7:
        return pts
    c, s = math.cos(ang), math.sin(ang)
    dx = pts[:, 0] - piv[0]
    dy = pts[:, 1] - piv[1]
    out = pts.copy()
    out[:, 0] = piv[0] + c * dx - s * dy
    out[:, 1] = piv[1] + s * dx + c * dy
    return out


def rot_weighted(pts, piv, ang, wt):
    """绕 piv 旋转, 每顶点角度 = ang[i] (逐顶点) 或 ang*wt (标量)。
    根权重0=不动, 梢权重1=全摆。ang 可为标量或与 wt 同长的数组。"""
    if wt is None:
        return rot(pts, piv, ang)
    a = ang * wt if np.isscalar(ang) else np.asarray(ang)
    if np.max(np.abs(a)) < 1e-7:
        return pts
    c, s = np.cos(a), np.sin(a)
    dx = pts[:, 0] - piv[0]
    dy = pts[:, 1] - piv[1]
    out = pts.copy()
    out[:, 0] = piv[0] + c * dx - s * dy
    out[:, 1] = piv[1] + s * dx + c * dy
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--coords", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--tol", type=float, default=0.4)
    # 呼吸
    ap.add_argument("--breath", type=float, default=0.016, help="呼吸缩放幅度")
    ap.add_argument("--bob", type=float, default=2.5, help="呼吸上下浮动(px)")
    ap.add_argument("--breath-cycles", type=float, default=2.0, help="每循环呼吸次数")
    # 眨眼
    ap.add_argument("--blink-at", default="1.6,3.9")
    ap.add_argument("--blink-dur", type=float, default=0.13)
    ap.add_argument("--blink-depth", type=float, default=0.88)
    # 风场 (头发/飘带)
    ap.add_argument("--wind-deg", type=float, default=6.0, help="头发风吹基础摆角")
    ap.add_argument("--ribbon-deg", type=float, default=11.0, help="飘带风吹摆角")
    ap.add_argument("--wind-amp", type=float, default=0.5,
                    help="全局飘动幅度 0.1~1 (0.5=中度); 缩放所有风吹摆角")
    ap.add_argument("--wind-base", type=float, default=1.0, help="风主频(次/循环)")
    ap.add_argument("--gust", type=float, default=0.6, help="阵风强度0~1(缓慢起伏)")
    ap.add_argument("--wave", type=float, default=1.4, help="行波相位跨度(按x), 弧度")
    ap.add_argument("--wind-stagger", type=float, default=0.6,
                    help="各飘带间错相强度(0=同步,1=最大错开)")
    ap.add_argument("--soft", type=float, default=0.5,
                    help="飘动柔软度 0~1 (大=更软更滞后, 鞭梢延迟更明显)")
    ap.add_argument("--hair-k", type=float, default=85.0)
    ap.add_argument("--hair-c", type=float, default=10.0)
    ap.add_argument("--ribbon-k", type=float, default=55.0)
    ap.add_argument("--ribbon-c", type=float, default=7.0)
    # 顶点级权重: 发根/头顶不动, 发梢/带尾全飘 (沿吊点->远端线性, root-lock 段保持0)
    ap.add_argument("--root-lock", type=float, default=0.28,
                    help="自吊点起此比例长度内权重=0(头顶/发根固定)")
    ap.add_argument("--tip-gamma", type=float, default=1.5,
                    help="权重曲线指数(>1 让根更稳, 梢更飘)")
    # 飘带画笔: 白色区域(>阈值)才参与风吹, 黑色不动。video 像素对齐 PNG
    ap.add_argument("--wind-mask", default="", help="风吹遮罩PNG路径(白=飘动)")
    ap.add_argument("--mask-thresh", type=float, default=0.4)
    # 手部 FK 呼吸: 大臂/小臂/手 沿链旋转, 角度随呼吸起伏
    ap.add_argument("--sh-deg", type=float, default=2.2, help="肩(大臂)摆角")
    ap.add_argument("--el-deg", type=float, default=3.5, help="肘(小臂)摆角")
    ap.add_argument("--wr-deg", type=float, default=4.5, help="腕(手)摆角")
    ap.add_argument("--arm-sign", type=float, default=-1.0, help="摆动方向")
    ap.add_argument("--arm-cycles", type=float, default=1.0, help="手部呼吸次数/循环")
    ap.add_argument("--arm-bob", type=float, default=2.0, help="手部额外浮动(px)")
    ap.add_argument("--arm-phase", type=float, default=1.9,
                    help="手臂相对身体呼吸的相位偏移(弧度), 避免手身同步运动")
    # 眉毛眨眼联动 (随动比例: 眉下移 = 眼高 × 此比例 × 闭合度)
    ap.add_argument("--brow-follow", type=float, default=0.10,
                    help="眨眼时眉毛随动比例 (默认 10%)")
    ap.add_argument("--brow-drop", type=float, default=4.0,
                    help="[兼容旧版] 绝对像素下压, 现已改用 brow-follow")
    # 头部伪3D扭头 (网格 IK 头部约束): 绕颈枢轴小幅 yaw, 顶部固定根, 五官随头
    ap.add_argument("--head-yaw", type=float, default=6.0,
                    help="头部伪3D扭头最大角(度), 0=关闭")
    ap.add_argument("--head-cycles", type=float, default=1.0,
                    help="扭头次数/循环")
    ap.add_argument("--head-persp", type=float, default=0.18,
                    help="伪3D透视强度(0~0.4): 远侧压缩制造立体扭头感")
    # 闭眼贴图切换 (extract_eye.py 产出目录, 含 blink_meta.json)
    ap.add_argument("--blink-eye", default="",
                    help="闭眼贴图目录; 提供则眨眼用切图(替代纵向挤压)")
    ap.add_argument("--preview", action="store_true")
    args, _unknown = ap.parse_known_args()    # 容错: 忽略骨骼模式专用参数(如 head-turn)

    coords = json.load(open(args.coords, encoding="utf-8"))
    cv0 = coords.get("canvas", [1280, 720])
    cw, ch = (cv0["w"], cv0["h"]) if isinstance(cv0, dict) else (cv0[0], cv0[1])
    T, fps = args.duration, args.fps
    F = int(round(T * fps))
    blink_times = [float(x) for x in args.blink_at.split(",") if x.strip()]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    img_dir = os.path.join(out_dir, "images")
    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir, ignore_errors=True)
    os.makedirs(img_dir, exist_ok=True)

    # 风吹遮罩 (可选): 白=飘动区。video 像素空间, 与画布对齐
    wind_mask = None
    if args.wind_mask and os.path.isfile(args.wind_mask):
        mk = np.array(Image.open(args.wind_mask).convert("L"), float) / 255.0
        wind_mask = mk
        print(f"风吹遮罩 {args.wind_mask} {mk.shape}")

    def vert_weights(v, cls):
        """顶点风吹权重 0~1 (根锁梢飘的纵向梯度)。所有飘动部件统一用此自动梯度。"""
        piv_y = v[:, 1].min()
        span = max(1.0, v[:, 1].max() - piv_y)
        u = (v[:, 1] - piv_y) / span
        lock = args.root_lock
        return np.clip((u - lock) / max(1e-3, 1 - lock), 0, 1) ** args.tip_gamma

    # mask 灰度约定 (中性=0.5): 未涂/残留噪声 -> 视为默认满幅(倍率=1, 不改动);
    #   涂 >0.5 -> 增强(最高 ~2x); 涂 <0.5 -> 削弱。圈哪改哪, 没圈的永远默认满幅。
    MASK_PAINT_MIN = 0.06            # 低于此视为未涂(残留噪声), 倍率=1

    def mask_scale(v):
        """逐顶点幅度倍率。未被圈中(含残留噪声)-> 1.0(默认满幅, 绝不削减)。
        被明确涂抹处: 灰度 0.5=维持, >0.5 增强(到2x), <0.5 削弱。
        返回 (倍率数组, 是否有任何涂抹)。"""
        if wind_mask is None:
            return None, False
        H, W = wind_mask.shape
        xi = np.clip(v[:, 0].astype(int), 0, W - 1)
        yi = np.clip(v[:, 1].astype(int), 0, H - 1)
        m = wind_mask[yi, xi]                         # 0~1 灰度
        painted = m > MASK_PAINT_MIN                  # 真正涂抹的顶点
        scale = np.ones_like(m)                       # 未涂 = 默认满幅
        # 涂抹处: 灰度映射到倍率 [0.5->1.0 中性], 0->0, 1->2
        scale[painted] = m[painted] * 2.0
        return scale, bool(painted.any())

    parts = []
    body_pts = []
    for p in coords["parts"]:
        name = p["name"]
        png = p.get("image", name + ".png")
        sp = os.path.join(args.parts, png)
        im = Image.open(sp)
        iw, ih = im.size
        x, y = float(p["x"]), float(p["y"])
        w = float(p.get("width", iw))
        h = float(p.get("height", ih))
        gx, gy = p.get("grid", [6, 6])
        is_bird = bool(p.get("bird", False))
        # see-through 满画布 PNG: 标称 bbox=整图, 真实部件靠 alpha 定位。
        # 用 alpha 紧致 bbox 还原真实 x/y/w/h + 对应 UV, 否则所有运动基准错位。
        uv_box = None
        bb = _alpha_bbox(im)
        if bb is not None:
            bl, bt, br, bbm = bb
            tw, tht = br - bl, bbm - bt
            # 仅当标称框明显大于真实像素(满画布情形)才替换
            if tw < 0.92 * w or tht < 0.92 * h or (x == 0 and y == 0):
                x, y, w, h = float(bl), float(bt), float(tw), float(tht)
                uv_box = (bl / iw, bt / ih, br / iw, bbm / ih)
        verts, uvs, tris, hull, edges = build_grid_mesh(x, y, w, h, gx, gy)
        if uv_box is not None:
            u0, v0u, u1, v1u = uv_box
            for k2 in range(0, len(uvs), 2):
                uvs[k2] = u0 + uvs[k2] * (u1 - u0)
                uvs[k2 + 1] = v0u + uvs[k2 + 1] * (v1u - v0u)
        cls = p.get("cls") or classify(name, is_bird)
        if is_bird:
            cls = "bird"
        shutil.copy(sp, os.path.join(img_dir, png))
        v = np.array(verts, float)
        if cls in ("body", "face", "brow"):
            body_pts.append(v)
        # 风吹: hair/ribbon 默认满幅飘(不受 mask 削减); mask 仅补充——
        #   涂抹 hair/ribbon -> 增强/削弱该区; 涂抹衣摆等其他部件 -> 让它也飘。
        # 面部五官(eye/brow/lash/face)有专属动画(眨眼/转面), 绝不被 mask 变成风吹,
        # 否则残留 mask 碰到眼睛会让眨眼消失。没涂的区域永远默认满幅。
        NO_WIND_CLS = ("eye", "brow", "lash", "face", "blink_eye",
                       "arm_upper", "arm_fore", "arm_hand")
        wind_cls = cls in ("hair", "ribbon")
        region, painted = mask_scale(v)
        wt = None
        if wind_cls:
            wt = vert_weights(v, cls)                 # 默认满幅梯度
            # 有 mask 则按 region 逐顶点缩放(未涂处 region=1, 不变)
        elif painted and cls not in NO_WIND_CLS:
            wt = vert_weights(v, cls)                 # 非五官/手臂但被涂 -> 开始飘
            wind_cls = True
        else:
            region = None                             # 不飘的部件不需要 region
        parts.append(dict(
            name=name, attachment=os.path.splitext(png)[0],
            uvs=uvs, triangles=tris, hull=hull, edges=edges,
            verts0=v, cls=cls, bird=is_bird, wt=wt, wind=wind_cls,
            region=region,
            top_pivot=(float(v[:, 0].mean()), float(v[:, 1].min())),
            center=(float(v[:, 0].mean()), float(v[:, 1].mean())), iw=iw, ih=ih))

    if not body_pts:
        body_pts = [p["verts0"] for p in parts if not p["bird"]]
    if not body_pts:
        raise SystemExit("无有效部件 (parts 为空或全为 bird)")
    allb = np.vstack(body_pts)
    bx0, bx1 = allb[:, 0].min(), allb[:, 0].max()
    by0, by1 = allb[:, 1].min(), allb[:, 1].max()
    chest_y = by0 + 0.45 * (by1 - by0)
    body_center = ((bx0 + bx1) / 2, (by0 + by1) / 2)

    # ---- 底部锁定(脚踩地): 整张图最底缘视为脚, 呼吸/浮动的垂直位移自底向上压制 ----
    # 越靠图底 -> 位移权重越小, 最底缘=0(完全不动); 过渡带内 smoothstep 衔接。
    # 头/发远离底部 -> 权重≈1 不受影响; 仅角色下半身/脚被钉住, 不再整体上下滑。
    foot_bot = by1
    foot_lock_h = max(1.0, 0.42 * (by1 - by0))    # 自底向上 42% 为锁定过渡带

    def foot_lock_w(y):
        t = np.clip((foot_bot - y) / foot_lock_h, 0.0, 1.0)
        return t * t * (3 - 2 * t)               # 底=0(锁), 过渡带上沿=1(自由)

    # ---- 闭眼贴图: 作为覆盖 slot, 平时透明, 眨眼瞬间显现 (替代纵向挤压) ----
    blink_use = bool(args.blink_eye) and os.path.isfile(
        os.path.join(args.blink_eye, "blink_meta.json"))
    blink_slots = []
    if blink_use:
        bmeta = json.load(open(os.path.join(args.blink_eye, "blink_meta.json"),
                               encoding="utf-8"))
        for side, m in bmeta.items():
            png = m["image"]
            sp = os.path.join(args.blink_eye, png)
            iw, ih = Image.open(sp).size
            shutil.copy(sp, os.path.join(img_dir, png))
            x, y = float(m["x"]), float(m["y"])
            w, h = float(m["w"]), float(m["h"])
            verts, uvs, tris, hull, edges = build_grid_mesh(x, y, w, h,4, 4)
            v = np.array(verts, float)
            nm = f"blink_{side}"
            parts.append(dict(
                name=nm, attachment=os.path.splitext(png)[0],
                uvs=uvs, triangles=tris, hull=hull, edges=edges,
                verts0=v, cls="blink_eye", bird=False, wt=None,
                top_pivot=(float(v[:, 0].mean()), float(v[:, 1].min())),
                center=(float(v[:, 0].mean()), float(v[:, 1].mean())),
                iw=iw, ih=ih))
            blink_slots.append(nm)
        print(f"闭眼切图: {blink_slots}")

    name2p = {p["name"]: p for p in parts}
    # 眼区高度 (眉毛随动幅度参考)
    eye_pts = [p["verts0"] for p in parts if p["cls"] in ("eye", "lash")]
    if eye_pts:
        ev = np.vstack(eye_pts)
        eye_h = float(ev[:, 1].max() - ev[:, 1].min())
    else:
        eye_h = 20.0
    # ---- 手臂 FK 枢轴 (静止几何估计) ----
    arm_upper_parts = [p for p in parts if p["cls"] == "arm_upper"]
    fk_ok = len(arm_upper_parts) > 0
    if fk_ok:
        ud = arm_upper_parts[0]["verts0"]
        shoulder = ud[int(np.argmin(np.linalg.norm(ud - np.array(body_center), axis=1)))]
        elbow = ud[int(np.argmax(np.linalg.norm(ud - shoulder, axis=1)))]
        hv = [p["verts0"] for p in parts if p["cls"] == "arm_hand"]
        if hv:
            handv = np.vstack(hv)
            wrist = handv[int(np.argmin(np.linalg.norm(handv - elbow, axis=1)))]
        else:
            wrist = elbow
        print(f"FK shoulder={shoulder.round(0)} elbow={elbow.round(0)} wrist={wrist.round(0)}")

    # 脸部刚性基准: 用脸主体(lian)中心做五官跟随的参考(此处五官与脸同一全局变换, 天然锁定)
    face_pts = [p["verts0"] for p in parts if p["cls"] in ("face", "eye")]
    if not face_pts:
        # 无可识别面部: 用上半身 bbox 顶部区域兜底, 仅用于风场 x 归一化
        face_pts = [np.array([[bx0, by0], [bx1, by0 + 0.4 * (by1 - by0)]], float)]
    fb = np.vstack(face_pts)
    print(f"body y[{by0:.0f},{by1:.0f}] chest_y={chest_y:.0f} "
          f"face x[{fb[:,0].min():.0f},{fb[:,0].max():.0f}] F={F}")

    # ---- 头部 2.5D 转面: 柱面投影 + 逐部件深度视差 + 颈部约束 ----
    # 原理(Spine 2.5D): 脸裹在绕垂直轴的圆柱面上, 转头=转圆柱(非线性,远侧前缩);
    # 五官按凸出深度产生水平视差(鼻>眼眉>脸颊>耳反向); 颈接处随动衰减(约束)。
    HEAD_CLS = ("face", "eye", "brow", "lash", "blink_eye")
    head_pts = [p["verts0"] for p in parts if p["cls"] in HEAD_CLS]
    head_ok = args.head_yaw > 0.01 and len(head_pts) > 0
    if head_ok:
        hb = np.vstack(head_pts)
        head_cx = float((hb[:, 0].min() + hb[:, 0].max()) / 2)
        head_top = float(hb[:, 1].min())
        head_bot = float(hb[:, 1].max())
        head_h = max(1.0, head_bot - head_top)
        head_half = max(1.0, (hb[:, 0].max() - hb[:, 0].min()) / 2)
        # 柱面半径: 边缘顶点映射到 ~52° (sin=0.78), 留出转动余量
        head_R = head_half / 0.78
        print(f"head 2.5D cx={head_cx:.0f} half={head_half:.0f} R={head_R:.0f} "
              f"y[{head_top:.0f},{head_bot:.0f}] maxdeg={args.head_yaw}")

    def part_depth(name):
        """部件相对柱面的凸出深度 (×head_half = 视差位移). 鼻最凸, 耳在后。
        眼珠/眼白视差≈0: 眼睛随脸柱面一起转, 不在脸上自行左右滑动。"""
        n = name.lower()
        if _has(n, ("nose", "bizi")):
            return 0.22
        if _has(n, ("eyewhite", "irides", "iris", "pupil", "tongkong",
                    "yanbai", "eyeball")):
            return 0.0                          # 眼珠贴脸转, 不额外位移
        if _has(n, ("eye", "yan")) and not _has(n, ("brow", "lash", "meimao",
                                                    "jiemao")):
            return 0.04                         # 眼眶极轻视差
        if _has(n, ("brow", "meimao")):
            return 0.10
        if _has(n, ("lash", "jiemao")):
            return 0.06
        if _has(n, ("mouth", "zui", "lip")):
            return 0.06
        if _has(n, ("ears", "erduo", "_er", "er2")):
            return -0.42                       # 耳在侧后, 反向视差
        if _has(n, ("front_hair", "qianfa", "liuhai", "bang", "fringe")):
            return 0.04                         # 刘海贴脸前
        if _has(n, HAIR_KW):
            return -0.15                        # 其余头发在头后, 随头转但偏后
        return 0.0                             # 脸/头/帽: 贴柱面基底

    # ---- 时间曲线 (三循环取稳态, 弹簧无缝) ----
    Mn = 3 * F
    tt = np.arange(Mn + 1) / fps
    w = 2 * math.pi / T
    wb = w * args.breath_cycles
    breath = args.breath * np.sin(wb * tt)
    bob = args.bob * np.sin(wb * tt + 0.4)
    wa = w * args.arm_cycles
    aph = args.arm_phase
    arm_bob = args.arm_bob * np.sin(wa * tt + aph)      # 手部浮动(与身体呼吸错相)
    # FK 呼吸: 三关节角度随起伏, 肩->肘->腕依次滞后(连锁运动, 非整体同步)
    a_sh = args.arm_sign * math.radians(args.sh_deg) * np.sin(wa * tt + aph)
    a_el = args.arm_sign * math.radians(args.el_deg) * np.sin(wa * tt + aph + 0.45)
    a_wr = args.arm_sign * math.radians(args.wr_deg) * np.sin(wa * tt + aph + 0.9)

    def fk(pts, gi, seg):
        if not fk_ok:
            return pts
        if seg == "arm_hand":
            pts = rot(pts, wrist, a_wr[gi])
            pts = rot(pts, elbow, a_el[gi])
            pts = rot(pts, shoulder, a_sh[gi])
        elif seg == "arm_fore":
            pts = rot(pts, elbow, a_el[gi])
            pts = rot(pts, shoulder, a_sh[gi])
        elif seg == "arm_upper":
            pts = rot(pts, shoulder, a_sh[gi])
        return pts

    # 风场: 主频 + 二次谐波 + 缓慢阵风包络(整周期)
    wind_main = np.sin(args.wind_base * w * tt) + 0.4 * np.sin(2 * args.wind_base * w * tt + 0.7)
    wind_main /= 1.4
    gust = 1.0 + args.gust * (0.5 - 0.5 * np.cos(w * tt))   # 0~1 阵风, 整周期
    s0 = 2 * F

    # 头部转面角度 (整周期正弦, 无缝). 左右往复看
    wh = w * args.head_cycles
    head_yaw = math.radians(args.head_yaw) * np.sin(wh * tt + 0.3) if head_ok else None

    def head_turn(pts, gi, depth):
        """2.5D 柱面转面 (整张脸统一转, 颈接处自底向上收敛, 中线不整体横移):
        1) 脸/五官: 顶点 x 当作柱面角位置 a=asin((x-cx)/R), 转 θ 后重投影,
           并减去中心整体平移分量 R·sinθ -> 脸中线钉住、只有两侧前缩/展开,
           避免整脸相对脖子横移产生脱节。
        2) 深度视差: 凸出部件(鼻/眉)额外水平位移 depth·head_half·sin(θ); 眼珠≈0。
        3) 颈接约束: 上部 neck_w≈1 整脸转, 下部(贴脖)自 u>0.7 起平滑衰减到 0。
        4) 头发(depth<0): 不做柱面重投影(发比脸宽会被 clip), 改整体水平平移,
           发根随头移、发梢衰减回归风吹。
        """
        th = head_yaw[gi]
        if abs(th) < 1e-6 and abs(depth) < 1e-9:
            return pts
        out = pts.copy()
        u = (pts[:, 1] - head_top) / head_h
        if depth <= -0.05:
            # 头发: 发根(u 小)随头移, 发梢(u 大)衰减回归风吹; 纯平移不重投影
            t01 = np.clip((u - 0.25) / 0.7, 0.0, 1.0)
            hair_w = 1.0 - t01 * t01 * (3 - 2 * t01)
            shift = (head_R * math.sin(th) * 0.35 + depth * head_half * math.sin(th))
            out[:, 0] = pts[:, 0] + shift * hair_w
            return out
        # 颈接约束: u<0.7 整脸满转, u 0.7->1 平滑收敛到 0 (脸底贴脖处不横移)
        t01 = np.clip((u - 0.7) / 0.3, 0.0, 1.0)
        neck_w = 1.0 - (t01 * t01 * (3 - 2 * t01))
        neck_w = np.clip(neck_w, 0.0, 1.0)
        s = np.clip((pts[:, 0] - head_cx) / head_R, -0.999, 0.999)
        a = np.arcsin(s)
        # 减去中心整体平移 R·sinθ: 只保留转面非线性形变, 脸中线不漂移
        xn = head_cx + head_R * (np.sin(a + th * neck_w) - math.sin(th) * neck_w)
        xn = xn + depth * head_half * math.sin(th) * neck_w
        out[:, 0] = xn
        return out

    def blink_factor(t):
        k = 1.0
        for bt in blink_times:
            d = abs(t - bt)
            if d < args.blink_dur:
                bb = 0.5 * (1 + math.cos(math.pi * d / args.blink_dur))
                k = min(k, 1 - args.blink_depth * bb)
        return k

    # 每个发缕/飘带按 x 位置算行波相位, 各自弹簧驱动
    # x 范围用全部风吹部件的中心跨度(而非脸宽), 保证不同飘带相位拉得开
    wind_parts = [p for p in parts if p.get("wind", False)]
    if wind_parts:
        wcx = [wp["center"][0] for wp in wind_parts]
        xw_lo, xw_hi = min(wcx), max(wcx)
    else:
        xw_lo, xw_hi = fb[:, 0].min(), fb[:, 0].max()
    # 给每个风吹部件分配一个稳定的相位偏移(按名字排序的序号), 即使 x 相近也错开
    wind_order = {wp["name"]: i for i, wp in enumerate(
        sorted(wind_parts, key=lambda q: q["center"][0]))}
    n_wind = max(1, len(wind_parts))

    def wind_signal(part, amp_deg):
        cx = part["center"][0]
        u = (cx - xw_lo) / max(1.0, xw_hi - xw_lo)
        # 行波相位(按 x) + 部件序号相位(错开同 x 的飘带)
        idx = wind_order.get(part["name"], 0)
        phase = (u - 0.5) * args.wave + (idx / n_wind) * 2 * math.pi * args.wind_stagger
        target = math.radians(amp_deg) * wind_main * gust
        shift = int(round(phase / w * fps)) if w > 0 else 0
        tg = np.roll(target, shift)
        return tg

    for p in parts:
        v0 = p["verts0"]
        nv = len(v0)
        fr = np.zeros((F + 1, nv, 2))
        cls = p["cls"]
        p["depth"] = part_depth(p["name"]) if head_ok else 0.0
        is_wind = p.get("wind", False)
        if is_wind:
            base = args.ribbon_deg if cls == "ribbon" else args.wind_deg
            amp = base * args.wind_amp                  # 全局幅度缩放
            kk = args.ribbon_k if cls == "ribbon" else args.hair_k
            cc = args.ribbon_c if cls == "ribbon" else args.hair_c
            # 柔软度: soft 越大 -> 弹簧更软(k小)、阻尼更低(c小) -> 更滞后飘逸
            soft = np.clip(args.soft, 0.0, 1.0)
            kk *= (1.0 - 0.6 * soft)
            cc *= (1.0 - 0.45 * soft)
            # 三循环取稳态(s0 偏移), 故此处单遍积分; loop=False 用旧逻辑。
            ang = spring_follow(wind_signal(p, amp), fps, kk, cc, loop=False)
            # 有效顶点权重 = 根锁梢飘梯度 × 区域倍率(未涂=1满幅, 涂抹处增强/削弱)
            eff_wt = p["wt"]
            if p.get("region") is not None:
                eff_wt = eff_wt * p["region"]
            p["_eff_wt"] = eff_wt
            # 鞭梢延迟(柔软度): 发梢比发根时间滞后, 越软滞后越大 -> 鞭子甩动感
            max_lag = soft * 0.18 * fps                 # 最大滞后帧(梢)
            lag = (eff_wt * max_lag).astype(int)        # 逐顶点滞后帧
            p["_lag"] = np.clip(lag, 0, s0 - 1)
            piv = p["top_pivot"]
        for fi in range(F + 1):
            gi = s0 + fi
            t = fi / fps
            pts = v0.copy()
            # 呼吸 + 浮动 (全局, 无摇摆); 底部(脚)位移按 foot_lock_w 自底向上压制, 脚踩地不滑
            flw = foot_lock_w(v0[:, 1])
            ny = chest_y + (v0[:, 1] - chest_y) * (1 + breath[gi]) + bob[gi]
            pts[:, 1] = v0[:, 1] + (ny - v0[:, 1]) * flw
            if is_wind:
                # 顶点级风吹 + 鞭梢延迟: 每顶点取 ang[gi - lag_i], 梢滞后更多
                avert = ang[gi - p["_lag"]] * p["_eff_wt"]
                pts = rot_weighted(pts, piv, avert, p["_eff_wt"])
            elif cls in ("arm_upper", "arm_fore", "arm_hand"):
                pts = fk(pts, gi, cls)                  # FK 链旋转呼吸
                pts[:, 1] += arm_bob[gi]               # 叠加轻浮动
            elif cls == "eye":
                if not blink_use:
                    kf = blink_factor(t)
                    ecy = p["center"][1]
                    pts[:, 1] = ecy + (pts[:, 1] - ecy) * kf
            elif cls == "lash":
                # 睫毛随眨眼挤压 (同眼)
                if not blink_use:
                    kf = blink_factor(t)
                    lcy = p["center"][1]
                    pts[:, 1] = lcy + (pts[:, 1] - lcy) * kf
            elif cls == "brow":
                # 眉随眨眼 10% 随动: 闭眼时眉下移 眼高×brow-follow×闭合度
                kf = blink_factor(t)
                closed = (1.0 - kf) / max(1e-3, args.blink_depth)   # 0~1
                pts[:, 1] += eye_h * args.brow_follow * closed
            # 头/五官/头发 叠加 2.5D 转面 (头发上部随头转, 发梢已被 neck_w 衰减→只风吹)
            if head_ok and (cls in HEAD_CLS or cls == "hair"):
                pts = head_turn(pts, gi, p["depth"])
            fr[fi] = pts
        p["frames"] = fr

    # ---- Spine json ----
    spine = {"skeleton": {"spine": "3.8.86", "x": 0, "y": 0, "width": cw, "height": ch,
                          "images": "./images/", "audio": ""},
             "bones": [{"name": "root"}], "slots": [],
             "skins": [{"name": "default", "attachments": {}}], "animations": {}}
    att = spine["skins"][0]["attachments"]
    for p in parts:
        slot = {"name": p["name"], "bone": "root", "attachment": p["attachment"]}
        if p["bird"]:
            slot["color"] = "ffffff00"          # idle 隐藏小鸟
        if p["cls"] == "blink_eye":
            slot["color"] = "ffffff00"          # 闭眼图初始透明, 眨眼时显现
        spine["slots"].append(slot)
        vs = []
        for (vx, vy) in p["verts0"]:
            sx, sy = video_to_spine(vx, vy, ch)
            vs += [round(sx, 2), round(sy, 2)]
        att[p["name"]] = {p["attachment"]: {"type": "mesh", "uvs": p["uvs"],
                          "triangles": p["triangles"], "vertices": vs, "hull": p["hull"],
                          "edges": p["edges"], "width": p["iw"], "height": p["ih"]}}
    deform = {}
    total = 0
    for p in parts:
        if p["bird"]:
            continue
        fr = p["frames"]
        v0 = p["verts0"]
        nv = len(v0)
        offs = fr - v0[None]
        offs[:, :, 1] *= -1.0
        flat = offs.reshape(F + 1, nv * 2)
        keep = decimate(flat, args.tol)
        kfs = []
        for fi in keep:
            o, tr = trim_deform(flat[fi].tolist())
            kf = {"time": round(fi / fps, 5)}
            if tr is not None:
                kf["offset"] = o
                kf["vertices"] = [round(x, 3) for x in tr]
            kfs.append(kf)
        total += len(kfs)
        deform.setdefault(p["name"], {})[p["attachment"]] = kfs
    spine["animations"]["idle"] = {"deform": {"default": deform}}

    # ---- 眨眼切图: slot color 时间线 (闭眼图淡入, 睁眼眼部淡出) ----
    # cvals 闭合度曲线: 无论是否有视频闭眼贴图都计算, 供预览显示眨眼。
    def closed_at(t):
        kf = blink_factor(t)
        return (1.0 - kf) / max(1e-3, args.blink_depth)   # 0~1
    cvals = np.array([closed_at(fi / fps) for fi in range(F + 1)])

    if blink_use:
        # 有视频闭眼贴图: 用 slot color 时间线做闭眼图淡入 / 睁眼眼部淡出
        eye_names = [p["name"] for p in parts if p["cls"] == "eye"]
        slots_tl = {}
        keepc = decimate(cvals.reshape(-1, 1), 0.03)
        for nm in blink_slots:
            slots_tl.setdefault(nm, {})["color"] = [
                {"time": round(fi / fps, 5),
                 "color": "ffffff%02x" % int(round(255 * cvals[fi]))}
                for fi in keepc]
        for nm in eye_names:
            slots_tl.setdefault(nm, {})["color"] = [
                {"time": round(fi / fps, 5),
                 "color": "ffffff%02x" % int(round(255 * (1 - cvals[fi])))}
                for fi in keepc]
        spine["animations"]["idle"]["slots"] = slots_tl

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(spine, f, ensure_ascii=False, separators=(",", ":"))
    print(f"完成 -> {args.out} 部件 {len(parts)} 关键帧 {total} 循环 {T}s")

    if args.preview:
        render_preview(parts, F, cw, ch, args.parts, out_dir, cvals, blink_use)


def render_preview(parts, F, cw, ch, parts_dir, out_dir, cvals=None,
                   blink_use=False):
    import cv2

    def load(p):
        # 闭眼图在 out_dir/images, 其余在 parts_dir
        cand = os.path.join(parts_dir, p["attachment"] + ".png")
        if not os.path.isfile(cand):
            cand = os.path.join(out_dir, "images", p["attachment"] + ".png")
        return np.array(Image.open(cand).convert("RGBA"))

    imgs = {p["name"]: load(p) for p in parts}

    def warp(img, uvs, tris, dst, canvas, ga=1.0):
        if ga <= 0.001:
            return
        iw, ih = img.shape[1], img.shape[0]
        nn = len(uvs) // 2
        src = np.array([[uvs[2 * k] * iw, uvs[2 * k + 1] * ih] for k in range(nn)], np.float32)
        for t in range(0, len(tris), 3):
            ix = [tris[t], tris[t + 1], tris[t + 2]]
            s = src[ix].astype(np.float32)
            d = dst[ix].astype(np.float32)
            x, y, ww, hh = cv2.boundingRect(d)
            if ww <= 0 or hh <= 0:
                continue
            d2 = (d - np.array([x, y], np.float32)).astype(np.float32)
            Maff = cv2.getAffineTransform(s, d2)
            patch = cv2.warpAffine(img, Maff, (ww, hh), flags=cv2.INTER_LINEAR)
            mk = np.zeros((hh, ww), np.uint8)
            cv2.fillConvexPoly(mk, d2.astype(np.int32), 255, cv2.LINE_AA)
            al = (patch[..., 3].astype(float) / 255.0) * (mk / 255.0) * ga
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(cw, x + ww), min(ch, y + hh)
            if x1 <= x0 or y1 <= y0:
                continue
            cx0, cy0 = x0 - x, y0 - y
            a3 = al[cy0:cy0 + (y1 - y0), cx0:cx0 + (x1 - x0)][..., None]
            col = patch[cy0:cy0 + (y1 - y0), cx0:cx0 + (x1 - x0), :3].astype(float)
            canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - a3) + col * a3

    gif = []
    # 抽稀: 含眨眼最闭合帧 (cvals 峰值) 便于核对
    base = [min(F - 1, int(round(F * r))) for r in (0, .14, .42, .56, .70, .98)]
    if cvals is not None:
        cv_arr = np.asarray(cvals[:F]).copy()
        for _ in range(2):
            bi = int(np.argmax(cv_arr))
            base.append(bi)
            cv_arr[max(0, bi - 4):bi + 4] = -1
    strip_idx = sorted(set(base))
    strip = {}
    for fi in range(F):
        canvas = np.full((ch, cw, 3), 60, float)
        cv = float(cvals[fi]) if cvals is not None else 0.0
        for p in parts:
            if p["bird"]:
                continue
            ga = 1.0
            if p["cls"] == "blink_eye":
                ga = cv                       # 闭眼图随闭合度淡入
            elif p["cls"] == "eye" and blink_use and cvals is not None:
                ga = 1.0 - cv                 # 有闭眼贴图时: 睁眼眼部淡出(被闭眼图替代)
            # 无闭眼贴图(纯Idle): 眼睛不淡出, 靠 frames 纵向挤压表现眨眼
            warp(imgs[p["name"]], p["uvs"], p["triangles"], p["frames"][fi], canvas, ga)
        full = canvas.astype(np.uint8)
        # 预览 GIF: 完整画布 (不裁切), 等比缩放到长边 ~400
        sc = 400.0 / max(cw, ch)
        gw, gh = max(1, int(cw * sc)), max(1, int(ch * sc))
        gif.append(Image.fromarray(full).resize((gw, gh)))
        if fi in strip_idx:
            strip[fi] = full
    gp = os.path.join(out_dir, "_idle_loop.gif")
    gif[0].save(gp, save_all=True, append_images=gif[1:], duration=1000 // 30, loop=0, disposal=2)
    sh = 200
    sw = max(1, int(cw * sh / ch))
    sp = np.hstack([cv2.resize(strip[k], (sw, sh)) for k in strip_idx if k in strip])
    Image.fromarray(sp).save(os.path.join(out_dir, "_idle_strip.png"))
    print(f"预览 -> {gp}  +  _idle_strip.png")

    # 眨眼脸部特写: 睁眼帧 vs 闭眼帧, 放大看拼缝
    if cvals is not None:
        cv_arr = np.asarray(cvals[:F])
        bi = int(np.argmax(cv_arr))               # 最闭合
        oi = int(np.argmin(cv_arr))               # 睁开
        # 自适应脸区: 用 eye/face/brow 部件 bbox (兜底整画布)
        fpts = [p["verts0"] for p in parts
                if p["cls"] in ("eye", "face", "brow", "lash", "blink_eye")]
        if fpts:
            fb = np.vstack(fpts)
            mx = 0.25 * (fb[:, 0].max() - fb[:, 0].min() + 1)
            my = 0.40 * (fb[:, 1].max() - fb[:, 1].min() + 1)
            fx0 = max(0, int(fb[:, 0].min() - mx))
            fy0 = max(0, int(fb[:, 1].min() - my))
            fx1 = min(cw, int(fb[:, 0].max() + mx))
            fy1 = min(ch, int(fb[:, 1].max() + my))
        else:
            fx0, fy0, fx1, fy1 = 0, 0, cw, ch
        shots = []
        for fi, tag in ((oi, "open"), (bi, "blink")):
            canvas = np.full((ch, cw, 3), 60, float)
            cv = float(cvals[fi])
            for p in parts:
                if p["bird"]:
                    continue
                ga = 1.0
                if p["cls"] == "blink_eye":
                    ga = cv
                elif p["cls"] == "eye" and blink_use:
                    ga = 1.0 - cv          # 仅有闭眼贴图时淡出; 纯Idle靠frames挤压
                warp(imgs[p["name"]], p["uvs"], p["triangles"], p["frames"][fi], canvas, ga)
            fc = canvas.astype(np.uint8)[fy0:fy1, fx0:fx1]
            fc = cv2.resize(fc, (fc.shape[1] * 2, fc.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
            cv2.putText(fc, tag, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            shots.append(fc)
        Image.fromarray(np.hstack(shots)).save(os.path.join(out_dir, "_blink_face.png"))
        print(f"眨眼特写 -> _blink_face.png (open f{oi} / blink f{bi})")


if __name__ == "__main__":
    main()
