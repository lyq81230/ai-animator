# -*- coding: utf-8 -*-
"""把全部角色重新走一遍: 重新拆图 -> 重新生成 行走/受击/死亡 三合一 -> 导出两版。

用法:
    python rebuild_all.py                 # 全部 8 个角色
    python rebuild_all.py sprite_pi_white_336   # 只做指定角色

每一步:
    1. build_from_seethrough.py  --src <AI分层> --out <work>/parts --head-wear <原模式>
       (头饰模式取自现有 coords.json 的 head_wear 字段, 保证和之前一致)
    2. 把 parts/coords.json 复制成 <work>/coords.json
    3. gen_idle_bones.py --action walk,hit,death
       -> <work>/{walk,hit,death}/ + spine/(json+atlas+png)
       不再导出 DragonBones/cocos 格式(网格版的 pivot/归一化问题已放弃)

绘制顺序(遮挡)的三种写法, 拆图会重写 coords, 所以都放在角色目录下持久化:
    <work>/z_order.json     {"z": {"arm_upper_L": -1}}       # 小于 0 = 画在躯干之前
    <work>/drop_extra.json  {"drop": ["ears"]}               # 垃圾层直接丢掉
    本文件的 DRAW_BEHIND / DRAW_Z                             # 再兜一层, 保证重建后一致
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
LAYERS = os.path.join(HERE, "seethrough_repo", "workspace", "layerdiff_output")
IMGS = {
    "sprite_boss_dongzhuo_white_336": "E:/pic/sprite_boss_dongzhuo_white_336.png",
    "sprite_boss_huaxiong_white_336": "E:/pic/sprite_boss_huaxiong_white_336.png",
    "sprite_boss_moshen_white_336": "E:/pic/sprite_boss_moshen_white_336.png",
    "sprite_pi_white_336": "E:/pic/sprite_pi_white_336.png",
    "sprite_quan_white_336": "E:/pic/sprite_quan_white_336.png",
    "sprite_suo_white_336": "E:/pic/sprite_suo_white_336.png",
    "fa": "E:/pic/fa.png",
    "sprite_boss_lvbu_white_336": "E:/pic/sprite_boss_lvbu_white_336.png",
    # 僵尸扇子男的重命名版(用户把 zombie_fan_hero_338 改名成 fashu_hero_338 重出图):
    # app 里能拖进来拆, 这里不登记的话命令行 `rebuild_all.py fashu_hero_338` 会
    # 直接"跳过: 缺源图或 AI 分层", 两边就对不上了。
    "fashu_hero_338": "E:/pic/fashu_hero_338.png",
    # 魔神的用户改图版(只改了右手武器尖端 x[284,306] y[272,295]), 用来重拆验证。
    # 只登记源图、不进 ORDER: 批量重建不碰它, 单名重建/继承母本配置可用。
    "sprite_boss_moshen_white_336_tipfix_preview":
        "E:/pic/sprite_boss_moshen_white_336_tipfix_preview.png",
}
ORDER = ["sprite_boss_dongzhuo_white_336", "sprite_boss_huaxiong_white_336",
         "sprite_boss_moshen_white_336", "sprite_pi_white_336",
         "sprite_quan_white_336", "sprite_suo_white_336", "fa",
         "sprite_boss_lvbu_white_336", "fashu_hero_338"]
DUR = {"walk": 2.0, "hit": 0.9, "death": 2.0}
# 需要"从身体层挖头"的角色: 它们的头本来就埋在 topwear 里。
# 其余角色开了这步会把两肩/胸甲挖进 head(董卓会变成 144x144)。
CARVE_HEAD = {"sprite_boss_moshen_white_336", "sprite_boss_lvbu_white_336"}
# 头部层已经带完整五官, 再叠睫毛/眉毛会重影 -> 拆图后直接丢掉
DROP_PARTS = "eyelash,eyebrow"
# 某个部件要画在躯干下面(被躯干遮住)的角色。数值写在 coords 的 z 字段里,
# 拆图会重写 coords, 所以每次重建都要重新应用一次。
DRAW_BEHIND = {"sprite_pi_white_336": ["arm_hand_L"],
               "fa": ["arm_upper_L"],
               "sprite_suo_white_336": ["arm_upper_L"]}
# 需要直接指定 z 的角色: 部件名 -> z。0 = 默认层, 小于 0 更靠后, 大于 0 更靠前。
#   dongzhuo 的靴子原来在最底层, 被裤腿盖掉靴筒 -> 提到腿上(0.5)
DRAW_Z = {"sprite_boss_dongzhuo_white_336": {"footwear_L": 0.5, "footwear_R": 0.5},
          # 魔神右手剑: 刀的护手/刀身要压住握把和护手甲。
          # 刀的杆和护手甲拆在两块里(arm_fore_R / arm_upper_R), 它们跟的是大臂/小臂,
          # 刀跟的是自己的道具骨 —— 走路时两者会错开一点, 握把处就"像断了一截"。
          # 把刀放到最上层, 交叉处由刀身盖住, 缝隙永远看不见。
          "sprite_boss_moshen_white_336": {"objects_R": 1.0}}
# 拆图后必须再跑一遍的修补脚本(顺序执行)。拆图会把人工修过的东西冲掉, 所以放这儿兜底。
#   lvbu: AI 给的头/五官层是错的(头只有半个脸, 五官是错位的椭圆块), 头要按多边形重建
# 所有角色拆完图都要跑的后处理: (脚本, 参数...)。app 也读这份, 避免两边不一致。
COMMON_POST = [# 先跟磁盘上的 PNG 对齐一次: 上一步(角色专属修补)常常重裁了部件,
               # 没同步的话下面的 bleed 会拿旧尺寸切新图, 直接 broadcast 报错中断。
               ("sync_coords.py", []),
               ("clear_foreign.py", ["600"]),     # 清掉跑进躯干/腿里的手和武器
               ("bleed_parts.py", ["3"]),         # 部件出血, 免得一动就露底
               # 清碎屑 + 画布裁到内容: 原画里 1~3px 的杂点会把画布撑大一截
               # (魔神右手剑 49x52 的内容被 2px 碎屑撑成 49x90), 编辑器里
               # 刀身就只剩上半截, 看着像"刀尖断了"。
               ("trim_parts.py", ["12"]),
               # 手臂沿主轴切三段时, 交界处会切出 1x13 / 1x4 这种一条线, 或者
               # 8x7 的小疙瘩。这些"碎片段"会各自变成槽位和骨骼, 还会抢走"真正的
               # 大臂"的身份(魔神左大臂就是这么被 8x7 碎块顶掉、整块绑到躯干上)。
               # 按"离谁最近"并回相邻部件, 像素一个不丢; 空片直接删。
               ("merge_tiny_parts.py", ["120"]),
               # 肩甲的大尖角被切进上臂时, 走路会跟着手臂甩出去(镜像的那一侧在躯干上),
               # 两边不对称。只在 rig.json 打开 trim_arm_dup 的角色上生效。
               ("trim_arm_dup.py", []),
               # 武器轮廓那圈半透明像素会把背后的手臂透出来 -> 刀边一帧一个样,
               # 看着像"轻微折断"。把轮廓压成实心(内部光晕不动)。
               ("harden_prop_rim.py", [])]

POST_FIX = {"sprite_boss_lvbu_white_336": ["fix_lvbu_head.py"],
            # 僵尸扇子男: AI 把"头巾+两肩+整个上半身"糊成一个 314x202 的 head,
            # ears 也是一条横带。拆图后必须拿原图重新分: 头巾归 headwear_overlay,
            # 扇子边上的紫烟雾归 objects。脸由 POST_FIX_LATE 里的 fix_zombie_head 重建。
            "zombie_fan_hero_338": ["repair_scarf_v2.py output/zombie_fan_hero_338/parts",
                                    "merge_fan_smoke.py output/zombie_fan_hero_338/parts"],
            # 同一张立绘的新命名文件(用户重命名), 走完全一样的补丁链
            "fashu_hero_338": ["repair_scarf_v2.py output/fashu_hero_338/parts",
                               "merge_fan_smoke.py output/fashu_hero_338/parts"],
            # 董卓: AI 给的 objects_L/R 是"拳头+披风+裙甲"大杂烩, 而 topwear 里又留着护腕
            # 上半截。披风并回 topwear, objects 删掉, 手从 topwear 里抹掉。
            "sprite_boss_dongzhuo_white_336": ["fix_dongzhuo_parts.py "
                                               "output/sprite_boss_dongzhuo_white_336"],
            # pi: 盾牌会被拆进右臂三块里(弯肘就裂开) -> 抽成独立的 objects_R;
            #     顺带用 fill_holes 把拆图剩下的 1px 缝和漏掉的零碎补回来
            "sprite_pi_white_336": ["extract_shield.py output/sprite_pi_white_336",
                                    "fill_holes.py output/sprite_pi_white_336 --alpha 200"],
            # 魔神: 两把剑被并进左右手/腿, 腿又是横跨整个下半身的一大块 —— 按区域重切
            "sprite_boss_moshen_white_336": [
                "recut_regions.py output/sprite_boss_moshen_white_336 config/moshen_regions.json",
                # 矩形切法剩下的两块孤岛(左刀柄/右披风边)按种子点整块搬家
                "move_island.py output/sprite_boss_moshen_white_336 --to objects_L --at 85,190",
                "move_island.py output/sprite_boss_moshen_white_336 --to wings --at 237,240",
                "move_island.py output/sprite_boss_moshen_white_336 --to leg_L_up --at 105,241",
                "fill_holes.py output/sprite_boss_moshen_white_336 --alpha 200"]}


# 必须在 COMMON_POST【之后】跑的修补: 出血/修边/轮廓硬化会往部件外面糊 3px
# 邻居颜色, 刚切出来的东西会被它们重新糊回去(董卓的佩剑就被 bleed 糊回了半圈
# 裙摆和甲片, harden_prop_rim 又把它压成不透明)。所以这类"要从原图上精确抠形"的
# 处理放在最后。
POST_FIX_LATE = {"sprite_boss_dongzhuo_white_336": [
    # AI 把头部下半段(脸/须)整块 1494px 复制进了 topwear, head 画在 topwear
    # 之上 —— 头一动就露出一张"贴在胸口的脸"。抠掉 head 覆盖的像素, 留 2px
    # 垫边防缝(与 fashu 的下巴同一类问题)。
    "cut_head_from_body.py output/sprite_boss_dongzhuo_white_336 --erode=2",
    "fix_dongzhuo_legs.py output/sprite_boss_dongzhuo_white_336"],
    # 僵尸扇子男: AI 给的 head 只有半个脸, 眼睛挖成两个大洞; 之前几轮都从坏掉的
    # head.png 出发反复裁, 越裁越小, 和头巾中间那道白缝就是"拆图错误"的来源。
    # 改成用源图 + 其余部件占位反推整张脸(见脚本注释), 必须在 bleed 之后跑。
    # fill_gaps 收尾: bleed 是 0.55 半透明的, 缝里那几像素会透出白底 -> 淡白虚线,
    # 用源图把没盖实的地方按原 alpha 补一遍。
    "zombie_fan_hero_338": ["fix_zombie_head.py output/zombie_fan_hero_338",
                            "fill_gaps.py output/zombie_fan_hero_338 --grow 2"],
    "fashu_hero_338": ["fix_zombie_head.py output/fashu_hero_338",
                       "fill_gaps.py output/fashu_hero_338 --grow 2",
                       "clean_frags.py output/fashu_hero_338",
                       # 下巴被 AI 同时分给了 head 和 topwear, 身体部件里带着半张脸。
                       # head 在 topwear 之上, 把 head 覆盖的像素从 topwear 扣掉即可。
                       "cut_head_from_body.py output/fashu_hero_338 --erode=2"]}


# 这些脚本会用源图(原画)重新抠形。命令行按 output/<名字> 去 E:/pic 找图就够了,
# 但 app 里拖进来的图可能在任意目录 —— 所以 app 会显式补 --src, 这里列出可补的脚本。
SRC_OK = {"fix_zombie_head.py", "repair_scarf_v2.py", "merge_fan_smoke.py",
          "fill_gaps.py"}


def cfg_key(table, name):
    """角色配置键解析: 精确匹配优先, 否则最长的"母本前缀"命中。

    用户把源图改名另存(如 sprite_boss_moshen_white_336_tipfix_preview.png,
    改了武器尖想重拆验证)后, 目录名跟着文件名走 —— 写死角色全名的专属配置
    (武器重切/挖头/绘制顺序)一个都不会命中, 拆出来的就是坏形态: 武器糊在
    手臂里、腿横跨整个下半身、靴尖成了游离碎块挂在 leg_R_low。
    所以改名字副本继承母本配置; 命中不到返回 None。
    """
    if name in table:
        return name
    hits = [k for k in table if name.startswith(k)]
    return max(hits, key=len) if hits else None


def prepare_args(script, work=None, src=""):
    """把配置里的一行补丁串展开成参数列表, CLI 和 app 共用同一套规则。

    两件事:
    1. 路径里的 output/<角色名> 换成**实际工作目录** —— app 的工作目录是
       OUT_ROOT/<名字>, 不一定等于仓库里的 output/<名字>(换目录/改名都会错位)。
    2. 带源图的脚本补 --src: 否则它们按名字去 E:/pic 找图, app 拖入的素材
       不在那里时直接 "找不到源图" 报错(界面上就是"拆图失败")。
    """
    bits = script.split()
    if not bits:
        return []
    head, rest = bits[0], []
    for b in bits[1:]:
        norm = b.replace("\\", "/").rstrip("/")
        m = re.match(r"^output/([^/]+)(/parts)?$", norm)
        rest.append(os.path.join(work, "parts") if (m and m.group(2) and work)
                    else work if (m and work) else b)
    if (src and head in SRC_OK
            and not any(x == "--src" or x.startswith("--src=") for x in rest)):
        rest += ["--src", src]
    return [head] + rest


def run(cmd, label):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    ok = p.returncode == 0
    print(f"  [{label}] {'OK' if ok else '失败'}  {time.time() - t0:.1f}s")
    tail = [x for x in (p.stdout or "").splitlines() if x.strip()][-6:]
    for x in tail:
        print("     ", x)
    if not ok:
        for x in (p.stderr or "").splitlines()[-8:]:
            print("   ! ", x)
    return ok


def mesh_report(work):
    """网格体检: 附件必须是 mesh、uv 落在自己图集格内、权重/三角索引合法。"""
    try:
        import check_spine_mesh as _ck
        _bad, _tot = _ck.problems(os.path.join(work, "spine"))
        print(f"  网格体检: {_tot - _bad}/{_tot} 个网格正常"
              + ("" if not _bad else f"  !! {_bad} 个有问题"))
    except Exception as _e:                      # noqa: BLE001
        print(f"  网格体检跳过: {_e}")


def one(name, no_split=False):
    work = os.path.join(OUT, name)
    src = IMGS.get(name)
    layers = os.path.join(LAYERS, name)
    if not (src and os.path.isdir(layers)):
        print(f"跳过 {name}: 缺源图或 AI 分层")
        return False

    mode = "scarf"
    cp = os.path.join(work, "coords.json")
    if os.path.isfile(cp):
        try:
            mode = json.load(open(cp, encoding="utf-8")).get("head_wear", "scarf")
        except Exception:                     # noqa: BLE001
            pass

    # 角色配置(z_order.json / drop_extra.json)平时放在 output/<名字>/ 下, 但那是
    # 生成目录 —— 一旦被删/清空就跟着没了(实测 output 被清过两次, 皮甲的手臂 z、
    # 吕布的 drop_extra 全丢了)。config/<名字>/ 是母本, 缺了就补回来。
    cfg = os.path.join(HERE, "config", name)
    if os.path.isdir(cfg):
        for fn in os.listdir(cfg):
            src_cfg = os.path.join(cfg, fn)
            if not os.path.isfile(src_cfg):
                # 子目录是"母本资料"(比如 shield/ 里的盾模板), 不是要铺到
                # 工作目录的配置 —— 直接 copy2 会抛 IsADirectoryError。
                continue
            dst = os.path.join(work, fn)
            # rig.json 是"人手写的骨骼/刚性配置", 永远以 config 下的母本为准 ——
            # 以前只在缺失时补, 改了 config 却不生效(work 里旧的那份还在)。
            if fn == "rig.json" or not os.path.isfile(dst):
                os.makedirs(work, exist_ok=True)
                shutil.copy2(os.path.join(cfg, fn), dst)
                print(f"  {'同步' if fn == 'rig.json' else '恢复'}配置 {fn}")

    print(f"=== {name}  (头饰模式 {mode})")
    if no_split:
        print("  跳过重新切片, 直接用现有 parts/coords 重新生成动作")
        ok = run([sys.executable, os.path.join(HERE, "gen_idle_bones.py"),
                  "--parts", os.path.join(work, "parts"), "--coords", cp,
                  "--out-dir", work, "--action", "walk,hit,death",
                  "--duration-map", ",".join(f"{k}={v:g}" for k, v in DUR.items()),
                  "--fps", "24",
                  # app 里播的动图就是这一步产出的 _<动作>_loop.gif; 不带就会一直播旧图
                  "--preview"],
                 "生成+导出")
        if ok:
            mesh_report(work)
            export_loong(work)
        return ok
    split_cmd = [sys.executable, os.path.join(HERE, "build_from_seethrough.py"),
                 "--src", layers, "--out", os.path.join(work, "parts"),
                 "--orig-image", src, "--head-wear", mode,
                 "--drop-parts", DROP_PARTS]
    _ck = cfg_key(CARVE_HEAD, name)
    if _ck:
        split_cmd.append("--carve-head")
    ok = run(split_cmd, "拆图")
    if not ok:
        return False

    shutil.copy2(os.path.join(work, "parts", "coords.json"), cp)

    _kb = cfg_key(DRAW_BEHIND, name)
    if _kb:
        run([sys.executable, os.path.join(HERE, "set_draw_order.py"), work,
             "--behind", *DRAW_BEHIND[_kb]], "绘制顺序")
    _kz = cfg_key(DRAW_Z, name)
    if _kz:
        run([sys.executable, os.path.join(HERE, "set_draw_order.py"), work,
             "--z", *[f"{k}={v:g}" for k, v in DRAW_Z[_kz].items()]], "绘制顺序")
    _kf = cfg_key(POST_FIX, name)
    for script in POST_FIX.get(_kf, []):
        bits = prepare_args(script, work, src)
        run([sys.executable, os.path.join(HERE, bits[0])] + bits[1:], script)

    # 通用后处理(和 app 共用同一份 COMMON_POST)
    for _sc, _args in COMMON_POST:
        run([sys.executable, os.path.join(HERE, _sc), work] + _args,
            {"clear": "清残片", "bleed": "出血", "trim": "修边"}.get(
                _sc.split("_")[0], "合并碎片"))

    _kl = cfg_key(POST_FIX_LATE, name)
    for script in POST_FIX_LATE.get(_kl, []):
        bits = prepare_args(script, work, src)
        run([sys.executable, os.path.join(HERE, bits[0])] + bits[1:], script)

    ok = run([sys.executable, os.path.join(HERE, "gen_idle_bones.py"),
              "--parts", os.path.join(work, "parts"), "--coords", cp,
              "--out-dir", work,
              "--action", "walk,hit,death",
              "--duration-map", ",".join(f"{k}={v:g}" for k, v in DUR.items()),
              "--fps", "24",
              # 必须带 --preview: app 里看到的动图就是这一步产出的 _<动作>_loop.gif,
              # 不带就只更新 spine, app 会继续播放上一次的旧动图(用户以为没生效)
              "--preview"],
             "生成+导出")
    if ok:
        mesh_report(work)
        export_loong(work)
    return ok


def export_loong(work):
    """另外导出一套给 LoongBones 的包(<名字>_loong): Spine 4.0 + 区域局部 uv。

    为什么: LoongBones 的 Spine 3.8 导入通道是坏的(连官方 spineboy 3.8 导进去也是
    一堆散落部件), 4.0 通道正常; 4.0 又把旋转关键帧 angle 改名成 value。所以走
    to_spine40 做字段转换。名字带 _loong 也顺便避开贴图缓存。
    """
    # 后缀用内容指纹(auto): 内容一变名字就变, 编辑器不会命中旧贴图缓存
    return run([sys.executable, os.path.join(HERE, "to_spine40.py"), work, "auto"],
               "导出 LoongBones 包")


def main():
    args = [a for a in sys.argv[1:] if a != "--no-split"]
    no_split = "--no-split" in sys.argv
    names = args or ORDER
    bad = []
    t0 = time.time()
    for n in names:
        if not one(n, no_split):
            bad.append(n)
    print(f"\n完成 {len(names) - len(bad)}/{len(names)}  用时 {time.time() - t0:.0f}s")
    if bad:
        print("失败:", ", ".join(bad))


if __name__ == "__main__":
    main()
