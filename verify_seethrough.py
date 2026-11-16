# -*- coding: utf-8 -*-
"""验证 AI 分层这条链路能不能跑通：跑一张图，打印档位/耗时/产物。

用法:
    python verify_seethrough.py [图片]           # 量化版(默认)
    python verify_seethrough.py [图片] --full    # 全参数版
"""
import os
import subprocess
import sys
import time

import seethrough

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
FULL = "--full" in sys.argv
IMG = _args[0] if _args else r"E:/pic/fa.png"
cmd, env, cwd, out_dir = seethrough.infer_cmd(IMG, full=FULL)
print("IMG   :", IMG)
print("档位  :", seethrough.model_text(FULL))
print("OUT   :", out_dir)
print("CMD   :", " ".join(str(c) for c in cmd))
print("quant :", cmd[cmd.index("--quant_mode") + 1] if "--quant_mode" in cmd else "?")
print("gpu   :", end=" ", flush=True)
subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader"], check=False)

before = {}
if os.path.isdir(out_dir):
    for f in os.listdir(out_dir):
        before[f] = os.path.getmtime(os.path.join(out_dir, f))

t0 = time.time()
p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
dt = time.time() - t0
print(f"\n=== 退出码 {p.returncode}  用时 {dt:.1f}s ===")
tail = (p.stdout or "").strip().splitlines()[-25:]
print("\n".join(tail))
if p.returncode != 0:
    print("--- STDERR 尾部 ---")
    print("\n".join((p.stderr or "").strip().splitlines()[-25:]))
    sys.exit(1)

print("\n--- 产物 ---")
for f in sorted(os.listdir(out_dir)):
    fp = os.path.join(out_dir, f)
    fresh = before.get(f) != os.path.getmtime(fp)
    print(f"  {'新' if fresh else '旧'} {f:24s} {os.path.getsize(fp)/1024:8.0f} KB")
