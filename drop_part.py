# -*- coding: utf-8 -*-
"""丢部件 + 持久化到 <角色>/drop_extra.json, 重新拆图也会继续丢。

用法:
    python drop_part.py output/sprite_boss_lvbu_white_336 ears
    python drop_part.py output/sprite_boss_lvbu_white_336 --list   # 看已丢弃的

被丢的 png 移到 <角色>/_dropped/ , 不直接删, 免得误判后找不回来。
"""
import json
import os
import shutil
import sys


def main():
    char = sys.argv[1]
    names = sys.argv[2:]
    cp = os.path.join(char, "coords.json")
    if not os.path.isfile(cp):
        print("没有", cp)
        return
    spec_path = os.path.join(char, "drop_extra.json")

    if names == ["--list"]:
        print(json.load(open(spec_path, encoding="utf-8"))
              if os.path.isfile(spec_path) else "（没有丢弃记录）")
        return

    co = json.load(open(cp, encoding="utf-8"))
    have = {p["name"] for p in co["parts"]}
    gone = [n for n in names if n in have]
    miss = [n for n in names if n not in have]
    if miss:
        print("警告: 没有这些部件", ", ".join(miss))
    if not gone:
        return

    co["parts"] = [p for p in co["parts"] if p["name"] not in gone]
    for path in (cp, os.path.join(char, "parts", "coords.json")):
        if os.path.isfile(path):
            json.dump(co, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    ddir = os.path.join(char, "_dropped")
    os.makedirs(ddir, exist_ok=True)
    pdir = os.path.join(char, "parts")
    for n in gone:
        f = os.path.join(pdir, n + ".png")
        if os.path.isfile(f):
            shutil.move(f, os.path.join(ddir, n + ".png"))

    spec = {"drop": []}
    if os.path.isfile(spec_path):
        try:
            spec = json.load(open(spec_path, encoding="utf-8"))
        except Exception:                    # noqa: BLE001
            pass
    spec["drop"] = sorted(set(spec.get("drop") or []) | set(gone))
    json.dump(spec, open(spec_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("已丢弃:", ", ".join(gone), "| 记录在", spec_path)


if __name__ == "__main__":
    main()
