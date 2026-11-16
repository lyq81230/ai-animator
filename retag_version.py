"""把角色 spine 包复制成"改名 + 改版本号"的一套, 给第三方工具导入用。

为什么需要:
    spine-ts 3.8 官方运行时里有一条硬校验 —— 骨架里 `skeleton.spine` 恰好等于
    "3.8.75" 时直接抛 "Unsupported skeleton data, please export with a newer
    version of Spine."。我们之前一直写 3.8.75(LoongBones 文档说支持这个版本),
    结果任何内置了这段校验的导入器都会整份拒绝, 然后退化成"每个部件一张默认四边形"
    —— 这正是用户截图里的样子。

    改名是另一件事: 贴图页文件名固定时, 编辑器可能按文件名缓存旧纹理; 换名字
    就不会命中缓存。

用法:
    python retag_version.py output/sprite_suo_white_336 3.8.55 r2
    -> output/sprite_suo_white_336/spine/sprite_suo_white_336_r2/{.json,.atlas,.png}
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser(description="改名 + 改版本号重导一套 spine 包")
    ap.add_argument("char_dir", help="角色目录, 如 output/sprite_pi_white_336")
    ap.add_argument("version", nargs="?", default="3.8.55", help="要写的 skeleton.spine")
    ap.add_argument("suffix", nargs="?", default="r2", help="新包后缀")
    args = ap.parse_args()

    work = args.char_dir.rstrip("/\\")
    name = os.path.basename(work)
    src = os.path.join(work, "spine")
    jf = os.path.join(src, name + ".json")
    at = os.path.join(src, name + ".atlas")
    pg = os.path.join(src, name + ".png")
    for p in (jf, at, pg):
        if not os.path.isfile(p):
            raise SystemExit(f"缺少文件: {p}")

    new = f"{name}_{args.suffix}"
    out = os.path.join(src, new)
    os.makedirs(out, exist_ok=True)

    J = json.load(open(jf, encoding="utf-8"))
    J.setdefault("skeleton", {})["spine"] = args.version
    dst_json = os.path.join(out, new + ".json")
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, separators=(",", ":"))

    # atlas 第一行是贴图页文件名, 跟着改名
    lines = open(at, encoding="utf-8").read().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip():
            lines[i] = new + ".png"
            break
    # 区域名不动(附件名要和 region 名一致)
    with open(os.path.join(out, new + ".atlas"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    shutil.copyfile(pg, os.path.join(out, new + ".png"))
    print(f"导出 {out}  版本 {args.version}")


if __name__ == "__main__":
    main()
