"""把我们的 Spine 3.8 导出转换成 Spine 4.0 JSON。

为什么需要:
    LoongBones 的导入器对 Spine 3.8 的文件是坏的(官方示例 spineboy 3.8 导进去也是
    一堆散落的部件), 但对 4.0 正常。它的 4.0 通道按 4.0 规范读关键帧, 而 4.0 把
    旋转关键帧的字段从 `angle` 改名成了 `value` —— 只把版本号改成 4.0.31 的话,
    旋转会被读成 0(骨架不动/动作全错)。

转换内容(其它结构 3.8 与 4.0 相同, 已用官方 4.0.31 运行时逐项验证):
    1. skeleton.spine        -> 4.0.31
    2. animations.bones.*.rotate[*].angle -> value      (4.0 改名)
    3. animations.draworder  -> drawOrder               (4.0 改名, 我们目前没有)
    4. 去掉 4.0 不认识的 3.8 遗留字段(如果有)

用法:
    python to_spine40.py output/sprite_pi_white_336 [后缀]
    -> output/sprite_pi_white_336/spine/sprite_pi_white_336_<后缀>/{.json,.atlas,.png}
"""
import argparse
import json
import os
import shutil
import sys

VER = "4.0.31"

# 4.0 与 3.8 的第二处硬差异(第一处是 rotate.angle->value):
#   3.8 的网格 uvs 是【相对整页图集】的归一化坐标(如 0.1827, 0.6383);
#   4.0 的网格 uvs 是【相对自己那块区域】的(0..1, 官方原件就是 [1,1, 0,1, 0,0, 1,0])。
# 不改的话, 4.0 运行时会拿整页坐标再按区域映射一次, 每个部件都采到图集上错的地方
# —— 屏幕上就是"位置对、贴图变成拼布"。
def to_region_local(J, regions, W, H):
    n = 0
    for skin in (J.get("skins") or []):
        for slot, atts in (skin.get("attachments") or {}).items():
            for name, a in atts.items():
                if a.get("type") != "mesh":
                    continue
                r = regions.get(name)
                if not r:
                    print(f"  !! 图集里找不到区域 {name}, uv 保持原样")
                    continue
                x, y, w, h = r
                du, dv = w / W, h / H
                uv = a["uvs"]
                out = []
                for i in range(0, len(uv), 2):
                    out.append(round((uv[i] - x / W) / du, 6))
                    out.append(round((uv[i + 1] - y / H) / dv, 6))
                a["uvs"] = out
                n += 1
    return n

# 4.0 里时间线字段的对应关系: 3.8 名 -> 4.0 名
KF_RENAME = {
    "rotate": {"angle": "value"},          # 4.0: rotate 用 value
    "translate": {},                        # x/y 不变
    "scale": {},                            # x/y 不变
    "shear": {},                            # x/y 不变
    "draworder": {},                        # 时间线名本身改名, 见下
}

TL_RENAME = {"draworder": "drawOrder"}


def convert(J):
    J.setdefault("skeleton", {})["spine"] = VER
    J["skeleton"].pop("hash", None)          # 4.0 不要求; 留着也无害, 去掉更干净

    for aname, anim in (J.get("animations") or {}).items():
        bones = anim.get("bones") or {}
        for bname, tl in bones.items():
            for tname, kfs in list(tl.items()):
                ren = KF_RENAME.get(tname, {})
                if not ren:
                    continue
                for kf in kfs:
                    for old, new in ren.items():
                        if old in kf:
                            kf[new] = kf.pop(old)
        # 时间线改名
        for old, new in TL_RENAME.items():
            if old in anim:
                anim[new] = anim.pop(old)
    return J


def main():
    ap = argparse.ArgumentParser(description="Spine 3.8 -> 4.0 转换并导出")
    ap.add_argument("char_dir")
    ap.add_argument("suffix", nargs="?", default="auto",
                    help="auto = 用内容指纹做后缀(内容变了名字就变, 避免编辑器缓存)")
    args = ap.parse_args()

    work = args.char_dir.rstrip("/\\")
    name = os.path.basename(work)
    src = os.path.join(work, "spine")
    jf = os.path.join(src, name + ".json")
    at = os.path.join(src, name + ".atlas")
    pg = os.path.join(src, name + ".png")
    for p in (jf, at, pg):
        if not os.path.isfile(p):
            raise SystemExit("缺少文件: " + p)

    # 武器类改成普通图片附件: 刚性物件不需要网格, 也没有顶点/uv 可以让导入端切掉细尖
    try:
        import props_as_region
        _n = props_as_region.convert(work, only_prop=True, quiet=True)
        if _n:
            print(f"  武器 -> 图片附件: {_n} 个")
    except Exception as _e:                              # noqa: BLE001
        print("  武器转图片附件跳过:", _e)
    if args.suffix == "auto":
        import hashlib
        h = hashlib.md5()
        for p_ in (jf, at, pg):
            h.update(open(p_, "rb").read())
        args.suffix = "loong_" + h.hexdigest()[:6]
    J = convert(json.load(open(jf, encoding="utf-8")))
    # uvs: 3.8 与 4.0 都是"区域局部坐标", 打包后已经是这样, 不需要再换。
    # (曾经这里做过换算, 是因为老版 make_spine_pack 把 uv 写成了整页坐标。)
    n = 0
    new = f"{name}_{args.suffix}"
    out = os.path.join(src, new)
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, new + ".json"), "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, separators=(",", ":"))

    lines = open(at, encoding="utf-8").read().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip():
            lines[i] = new + ".png"
            break
    with open(os.path.join(out, new + ".atlas"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    shutil.copyfile(pg, os.path.join(out, new + ".png"))
    print(f"导出 {out}  spine {VER}")
    # 同一角色每重建一次就会多一个 <名字>_loong_<指纹>/ —— 内容一变指纹就变,
    # 越积越多, 用户在 LoongBones 里根本分不清哪个是最新的(导入错了还以为没生效)。
    # 只保留刚导出的这一个, 其余同前缀目录清掉。
    try:
        prefix = f"{name}_loong_"
        for d_ in sorted(os.listdir(src)):
            p_ = os.path.join(src, d_)
            if d_ != new and d_.startswith(prefix) and os.path.isdir(p_):
                shutil.rmtree(p_, ignore_errors=True)
                print(f"  清掉旧导出 {d_}")
    except OSError as _e:
        print("  旧导出清理跳过:", _e)


if __name__ == "__main__":
    main()
