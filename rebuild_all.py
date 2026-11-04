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
"""
import json
import os
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
}
ORDER = ["sprite_boss_dongzhuo_white_336", "sprite_boss_huaxiong_white_336",
         "sprite_boss_moshen_white_336", "sprite_pi_white_336",
         "sprite_quan_white_336", "sprite_suo_white_336", "fa",
         "sprite_boss_lvbu_white_336"]
DUR = {"walk": 2.0, "hit": 0.9, "death": 2.0}
# 需要"从身体层挖头"的角色: 它们的头本来就埋在 topwear 里。
# 其余角色开了这步会把两肩/胸甲挖进 head(董卓会变成 144x144)。
CARVE_HEAD = {"sprite_boss_moshen_white_336", "sprite_boss_lvbu_white_336"}


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

    print(f"=== {name}  (头饰模式 {mode})")
    if no_split:
        print("  跳过重新切片, 直接用现有 parts/coords 重新生成动作")
        return run([sys.executable, os.path.join(HERE, "gen_idle_bones.py"),
                    "--parts", os.path.join(work, "parts"), "--coords", cp,
                    "--out-dir", work, "--action", "walk,hit,death",
                    "--duration-map", ",".join(f"{k}={v:g}" for k, v in DUR.items()),
                    "--fps", "24"],
                   "生成+导出")
    split_cmd = [sys.executable, os.path.join(HERE, "build_from_seethrough.py"),
                 "--src", layers, "--out", os.path.join(work, "parts"),
                 "--orig-image", src, "--head-wear", mode]
    if name in CARVE_HEAD:
        split_cmd.append("--carve-head")
    ok = run(split_cmd, "拆图")
    if not ok:
        return False

    shutil.copy2(os.path.join(work, "parts", "coords.json"), cp)

    ok = run([sys.executable, os.path.join(HERE, "gen_idle_bones.py"),
              "--parts", os.path.join(work, "parts"), "--coords", cp,
              "--out-dir", work,
              "--action", "walk,hit,death",
              "--duration-map", ",".join(f"{k}={v:g}" for k, v in DUR.items()),
              "--fps", "24"],
             "生成+导出")
    return ok


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
