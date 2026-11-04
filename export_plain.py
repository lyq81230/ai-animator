# -*- coding: utf-8 -*-
"""把角色的 DragonBones 导出统一成"纯图片显示版"(type 0, 无网格)。

背景: DragonBones Pro 5.6 导入带网格/权重的导出后, 头部会缩小浮在肩上
      (位置对不上)。纯图片版导入后位置正确, 所以 cocos/ 走这条路。

同时可以再出一份网格版(cocos_mesh/)做对照 —— DragonBones 官方形态本来就是网格
(官方样例 131 个显示全是网格), 只是我们这份 Pro 还解析不对。字段按官方样例改写:
type 用字符串 "mesh", 且不写 width/height。

用法:
    python export_plain.py output/sprite_pi_white_336 [更多目录...]
    python export_plain.py --mesh output/sprite_pi_white_336   # 只出网格版

做的事:
    1. 现有 cocos/ 若还存在 -> 备份成 cocos_mesh/(已存在则跳过)
    2. 用 walk/hit/death 三个 result.json 重新导出 --plain 到 cocos/
    3. 修 tex.json 的 name(必须等于骨架名)
    4. 渲染 _预览_纯图片版.png
    5. --mesh: 额外导一份网格版到 cocos_mesh/, 并按官方样例改写字段
"""
import json
import os
import shutil
import sys

from spine2dragonbones import convert_multi

ANIMS = ["walk", "hit", "death"]
MESH_ONLY = "--mesh" in sys.argv
if MESH_ONLY:
    sys.argv.remove("--mesh")


def export_mesh(d, name=None):
    """额外导一份网格版到 cocos_mesh/, 字段按官方样例改写。"""
    d = os.path.abspath(d)
    name = name or os.path.basename(d)
    import shutil as _sh
    _sh.rmtree(os.path.join(d, "cocos_mesh"), ignore_errors=True)
    items, missing = [], []
    for a in ANIMS:
        p = os.path.join(d, a, "result.json")
        if not os.path.isfile(p):
            missing.append(a)
            continue
        items.append({"spine": p, "images": os.path.join(d, a, "images"),
                      "anim": a})
    if not items:
        return None
    # rigid=True: 每个部件导成一个 4 顶点四边形, 不写 weights/slotPose/bonePose。
    # 这是 DragonBones 官方样例里唯一出现过的网格形态(131/131 全是刚体, 0 个带权重),
    # 加权网格我们这份和 Pro 的 slotPose/bonePose 语义对不上, 整块会被斜切。
    # 骨架名/文件名加 _mesh 后缀: cocos/ 和 cocos_mesh/ 如果同名同文件名,
    # 导进同一个 DragonBones 项目会互相覆盖贴图 —— 一部分部件采到空白区,
    # 在编辑器里就是一块块灰方框和拉扯的竖条。
    info = convert_multi(items, os.path.join(d, "cocos_mesh"), name=name + "_mesh",
                         plain=False, rigid=True, drop_parts=())
    tex_p = os.path.join(d, "cocos_mesh", name + "_mesh_tex.json")
    t = json.load(open(tex_p, encoding="utf-8"))
    # 图集内部名必须和骨架名/文件名一致, 并且**不能和 cocos/ 那份重名**:
    # DragonBones 用 tex.name 当图集资源键, 两套同名时后导入的会复用旧的贴图
    # (旧排版) → 部件采到错误区域: 头巾缺左半、动作里贴图全乱。
    if t.get("name") != name + "_mesh":
        t["name"] = name + "_mesh"
        json.dump(t, open(tex_p, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    import db_mesh_fix
    n_mesh, n_fix = db_mesh_fix.to_official(info["ske"])
    # 再对齐成"和图片版同一套定位": 四边形铺满整张贴图格 + 抄图片版的 transform。
    # 用户实测: 顶点写成骨局部 / 画布绝对都不对, 只有"居中 + transform"能对上;
    # 而按网格内容框算的四边形比贴图格小 1~2px, 头巾这类细部件会被压出偏差。
    try:
        import db_mesh_from_plain
        db_mesh_from_plain.align(d)
    except Exception as ex:            # noqa: BLE001
        print(f"  网格对齐跳过: {ex}")
    print(f"  网格版: 网格显示 {n_mesh} 个, 按官方样例改写 {n_fix} 个"
          + (f"  缺动作 {missing}" if missing else ""))
    return info


def export_one(d, name=None):
    d = os.path.abspath(d)
    name = name or os.path.basename(d)
    cocos = os.path.join(d, "cocos")
    mesh = os.path.join(d, "cocos_mesh")
    if MESH_ONLY:
        return export_mesh(d, name)
    _export_plain(d, name, cocos, mesh)
    export_mesh(d, name)


def _export_plain(d, name, cocos, mesh):
    items, missing = [], []
    for a in ANIMS:
        p = os.path.join(d, a, "result.json")
        img = os.path.join(d, a, "images")
        if not os.path.isfile(p):
            missing.append(a)
            continue
        items.append({"spine": p, "images": img, "anim": a})
    if not items:
        raise RuntimeError(f"{name}: 没有任何动作数据")

    # 1) 备份网格版
    if os.path.isdir(cocos) and not os.path.isdir(mesh):
        os.rename(cocos, mesh)
        print(f"  备份: cocos -> cocos_mesh")
    os.makedirs(cocos, exist_ok=True)

    # 2) 纯图片导出
    #    drop_parts=() 保留"闭眼补丁": 它挂在缩放 0.001 的骨上, 编辑器结算骨缩放
    #    就看不见(董卓已验证 Pro 里正常)。想要绝对保险可以改回默认值。
    info = convert_multi(items, cocos, name=name, plain=True, drop_parts=())

    # 3) 图集名/骨架名对齐
    tex_p = os.path.join(cocos, name + "_tex.json")
    t = json.load(open(tex_p, encoding="utf-8"))
    if t.get("name") != name:
        old = t.get("name")
        t["name"] = name
        json.dump(t, open(tex_p, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  图集名 {old} -> {name}")

    print(f"  导出: 骨 {info['bones']} 插槽 {info['slots']} "
          f"动作 {info['anim_names']} 图集 {info['atlas'][0]}x{info['atlas'][1]}"
          + (f"  缺动作 {missing}" if missing else ""))
    return info


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        return
    for d in dirs:
        print(f"=== {d}")
        export_one(d)
    # 渲染预览
    try:
        import db_player
        for d in dirs:
            out = os.path.join(d, "_预览_纯图片版.png")
            db_player.draw(os.path.join(d, "cocos"), out, draw_bones=False)
            print(f"  预览: {out}")
    except Exception as e:  # 渲染失败不影响导出
        print(f"  预览渲染跳过: {e}")


if __name__ == "__main__":
    main()
