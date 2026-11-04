"""校验: 两套独立实现分别求值 Spine / DragonBones 数据, 比对网格顶点的世界坐标。

Spine 侧: y 向上 -> 世界矩阵 = 父 * T*R*S, 最后翻转 y 得屏幕坐标。
DragonBones 侧: y 向下, 标准 2D 仿射; 顶点按 weights 合成:
    world = Σ w * ( W_bone · bonePose⁻¹ · slotPose · v )
两边同帧同顶点应重合(误差 < 0.5px), 否则转换有误。
"""
import json
import math
import os
import sys


def lerp(a, b, t):
    return a + (b - a) * t


def sample(keys, t, rec):
    if not keys:
        return dict(rec)
    if t <= keys[0]["time"]:
        return {**rec, **{k: v for k, v in keys[0].items() if k != "time"}}
    if t >= keys[-1]["time"]:
        return {**rec, **{k: v for k, v in keys[-1].items() if k != "time"}}
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        if a["time"] <= t <= b["time"]:
            u = (t - a["time"]) / max(1e-9, b["time"] - a["time"])
            out = dict(rec)
            for k in set(a) | set(b):
                if k == "time":
                    continue
                out[k] = lerp(a.get(k, rec.get(k, 0.0)), b.get(k, rec.get(k, 0.0)), u)
            return out
    return dict(rec)


def mmul(A, B):
    a1, b1, c1, d1, e1, f1 = A
    a2, b2, c2, d2, e2, f2 = B
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)


def trs(x, y, rot, sx, sy):
    r = math.radians(rot)
    c, s = math.cos(r), math.sin(r)
    return (c * sx, s * sx, -s * sy, c * sy, x, y)


def app(M, p):
    return (M[0] * p[0] + M[2] * p[1] + M[4], M[1] * p[0] + M[3] * p[1] + M[5])


def spine_worlds(d, t, anim=None):
    bones = {b["name"]: b for b in d["bones"]}
    an = d["animations"]
    anim = an[anim] if anim in an else an[list(an)[0]]
    tl = anim.get("bones", {})
    out = {}

    def w(name):
        if name in out:
            return out[name]
        b = bones[name]
        x, y = float(b.get("x", 0)), float(b.get("y", 0))
        rot, sx, sy = float(b.get("rotation", 0)), 1.0, 1.0
        tr = tl.get(name, {})
        if tr.get("translate"):
            v = sample(tr["translate"], t, {"x": 0.0, "y": 0.0}); x += v["x"]; y += v["y"]
        if tr.get("rotate"):
            rot += sample(tr["rotate"], t, {"angle": 0.0})["angle"]
        if tr.get("scale"):
            v = sample(tr["scale"], t, {"x": 1.0, "y": 1.0}); sx *= v["x"]; sy *= v["y"]
        M = trs(x, y, rot, sx, sy)
        p = b.get("parent")
        out[name] = mmul(w(p), M) if p else M
        return out[name]

    for n in bones:
        w(n)
    return out


def db_worlds(ske, t, fps, anim=None):
    arm = ske["armature"][0]
    bones = {b["name"]: b for b in arm["bone"]}
    anim, anim_data = _anim_entry(arm, anim)
    tl = {e["name"]: e for e in anim_data.get("bone", [])}
    out = {}

    def keys(e, conv):
        ks, acc = [], 0
        for fr in e:
            v = {"time": acc / fps}
            v.update(conv(fr))
            ks.append(v)
            acc += fr.get("duration", 1)
        return ks

    def w(name):
        if name in out:
            return out[name]
        b = bones[name]
        tr = b.get("transform", {})
        x, y = float(tr.get("x", 0)), float(tr.get("y", 0))
        # 5.x 骨头 setup 的旋转字段是 skX/skY; 也兼容我们自己以前写的 rotate
        rot = float(tr.get("skY", tr.get("rotate", 0)))
        # 绑定姿态的缩放(scX/scY)会与动画 scale 相乘, 必须一起算
        sx = float(tr.get("scX", tr.get("scaleX", 1.0)))
        sy = float(tr.get("scY", tr.get("scaleY", 1.0)))
        e = tl.get(name)
        if e:
            if e.get("translateFrame"):
                v = sample(keys(e["translateFrame"], lambda k: {"x": k.get("x", 0.0), "y": k.get("y", 0.0)}),
                           t, {"x": 0.0, "y": 0.0}); x += v["x"]; y += v["y"]
            if e.get("rotateFrame"):
                v = sample(keys(e["rotateFrame"], lambda k: {"angle": k.get("rotate", 0.0)}),
                           t, {"angle": 0.0}); rot += v["angle"]
            if e.get("scaleFrame"):
                v = sample(keys(e["scaleFrame"], lambda k: {"x": k.get("x", 1.0), "y": k.get("y", 1.0)}),
                           t, {"x": 1.0, "y": 1.0}); sx *= v["x"]; sy *= v["y"]
        M = trs(x, y, rot, sx, sy)
        p = b.get("parent")
        out[name] = mmul(w(p), M) if p else M
        return out[name]

    for n in bones:
        w(n)
    return out


def _anim_entry(arm, anim):
    """取一条动画的数据。5.5: 对象按名字取; 6.0: 数组找 name。"""
    an = arm.get("animation") or {}
    if isinstance(an, dict):
        if anim and anim in an:
            return anim, an[anim]
        k = next(iter(an))
        return k, an[k]
    if anim:
        for x in an:
            if x.get("name") == anim:
                return anim, x
    return (an[0]["name"], an[0]) if an else (anim, {})


def main(spine_p, ske_p, frames=None, anim=None):
    d = json.load(open(spine_p, encoding="utf-8"))
    ske = json.load(open(ske_p, encoding="utf-8"))
    fps = float(ske["frameRate"])
    arm0 = ske["armature"][0]
    anim, adata = _anim_entry(arm0, anim)
    total = int(adata.get("duration") or 0)
    if not frames:
        frames = [int(round(total * k / 8.0)) for k in range(9)]
    arm = ske["armature"][0]
    disp = {}
    for sk in arm["skin"]:
        for s in sk["slot"]:
            disp[s["name"]] = s["display"][0]
    satt = d["skins"][0]["attachments"]

    worst, worst_where = 0.0, ""
    for fr in frames:
        t = fr / fps
        Ws = spine_worlds(d, t, anim)
        Wd = db_worlds(ske, t, fps, anim)
        for sname, dd in disp.items():
            a = satt.get(sname)
            if not a:
                continue
            att = next(iter(a.values()))
            sv = att.get("vertices") or []
            # spine: [n,(bi,lx,ly,w)...]
            sp_pts, i = [], 0
            while i < len(sv):
                n = int(sv[i]); i += 1
                px = py = 0.0
                for _ in range(n):
                    bi, lx, ly, wt = int(sv[i]), sv[i + 1], sv[i + 2], sv[i + 3]
                    i += 4
                    M = Ws[d["bones"][bi]["name"]]
                    q = app(M, (lx, ly))
                    px += q[0] * wt; py += q[1] * wt
                # 翻 y 并整体下移 ch: 龙骨数据把根骨下移了 ch(原点左上), 这里对齐
                sp_pts.append((px, float(ske["armature"][0]["aabb"]["height"]) - py))
            # db: vertices + weights
            dv = dd.get("vertices") or []
            w = dd.get("weights") or []
            bp = dd.get("bonePose") or []
            bpos = {}
            for k in range(len(bp) // 7):
                bpos[bp[k * 7]] = (bp[k * 7 + 5], bp[k * 7 + 6])
            vc = len(dv) // 2
            k = 0
            # 龙骨的合成顺序: world = Σ w · ( W_bone · bonePose⁻¹ · slotPose · v )
            sp_pose = dd.get("slotPose") or [1, 0, 0, 1, 0, 0]
            if not w:
                # 刚性网格(4 角四边形): 顶点是"挂在 slot 父骨上的局部坐标", world = W_bone · v
                # 顶点数与原网格不同, 比值没意义 -> 只比"包围盒", 仍是有效的形变校验
                _par = {q2["name"]: q2.get("parent") for q2 in arm["slot"]}.get(sname)
                M0 = Wd.get(_par) if _par else None
                qs = []
                for vi in range(vc):
                    qs.append(app(M0, (dv[vi * 2], dv[vi * 2 + 1])) if M0
                              else (dv[vi * 2], dv[vi * 2 + 1]))
                if qs and sp_pts:
                    bx0 = min(q[0] for q in qs); bx1 = max(q[0] for q in qs)
                    by0 = min(q[1] for q in qs); by1 = max(q[1] for q in qs)
                    sx0 = min(q[0] for q in sp_pts); sx1 = max(q[0] for q in sp_pts)
                    sy0 = min(q[1] for q in sp_pts); sy1 = max(q[1] for q in sp_pts)
                    e = max(abs(bx0 - sx0), abs(bx1 - sx1), abs(by0 - sy0), abs(by1 - sy1))
                    if e > worst:
                        worst, worst_where = e, f"frame={fr} slot={sname} 包围盒"
                continue
            for vi in range(vc):
                wx, wy = app(sp_pose, (dv[vi * 2], dv[vi * 2 + 1]))
                px = py = 0.0
                # weights 是逐顶点交错的: 骨数, (骨索引, 权重) × 骨数, ...
                cnt_i = int(w[k]); k += 1
                for _ in range(cnt_i):
                    bi, wt = w[k], w[k + 1]; k += 2
                    bx, by = bpos[bi]
                    local = (wx - bx, wy - by)
                    M = Wd[arm["bone"][bi]["name"]]
                    q = app(M, local)
                    px += q[0] * wt; py += q[1] * wt
                if vi < len(sp_pts):
                    e = math.hypot(px - sp_pts[vi][0], py - sp_pts[vi][1])
                    if e > worst:
                        worst, worst_where = e, f"frame={fr} slot={sname} v={vi}"
    print(f"[{anim}] 最大顶点偏差: {worst:.3f}px  ({worst_where})")
    return worst


def _anim_names(arm):
    """取动画名。5.5 是对象(名字->数据), 6.0 是数组(每项带 name)。
    Cocos 只吃 5.5, 但校验脚本两种都认, 免得换了写法就跑不起来。"""
    an = arm.get("animation") or {}
    if isinstance(an, dict):
        return list(an.keys())
    return [x.get("name") for x in an]


def _verify_work(work):
    """--work 模式: 一个角色的全部动作 + 合并后的龙骨文件, 逐条校验。"""
    import glob
    db = os.path.join(work, "cocos")
    if not os.path.isdir(db):
        db = os.path.join(work, "dragonbones")
    skes = sorted(glob.glob(os.path.join(db, "*_ske.json")))
    if not skes:
        print(f"没找到 {db}/*_ske.json")
        return
    ske_p = skes[0]
    ske = json.load(open(ske_p, encoding="utf-8"))
    names = _anim_names(ske["armature"][0])
    print(f"龙骨: {ske_p}  版本 {ske.get('version')}  动画 {names}")
    worst, n = 0.0, 0
    for act in names:
        sp = os.path.join(work, act, "result.json")
        if not os.path.isfile(sp):
            print(f"  [跳过] 没有 {sp}")
            continue
        worst = max(worst, main(sp, ske_p, anim=act))
        n += 1
    if n:
        print(f"共 {n} 条动作, 最大顶点偏差 {worst:.3f}px "
              f"({'OK' if worst < 0.5 else '有偏差, 检查转换'})")


if __name__ == "__main__":
    # python verify_dragonbones.py <result.json> <ske.json> [动画名]
    # python verify_dragonbones.py --work output/<素材名>     (全部动作)
    if len(sys.argv) >= 3 and sys.argv[1] == "--work":
        _verify_work(sys.argv[2])
    elif len(sys.argv) >= 4:
        main(sys.argv[1], sys.argv[2], anim=sys.argv[3])
    elif len(sys.argv) == 3:
        d = json.load(open(sys.argv[1], encoding="utf-8"))
        ske = json.load(open(sys.argv[2], encoding="utf-8"))
        names = _anim_names(ske["armature"][0])
        shared = [nm for nm in names if nm in d["animations"]]
        if not shared:
            main(sys.argv[1], sys.argv[2])
        else:
            worst = 0.0
            for nm in shared:
                worst = max(worst, main(sys.argv[1], sys.argv[2], anim=nm))
            print(f"{len(shared)} 条共有动画, 最大偏差 {worst:.3f}px "
                  f"(该 result.json 里没有的动作跳过)")
    else:
        print(__doc__)
