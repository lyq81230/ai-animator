#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seethrough.py — AI 自动拆图（see-through / LayerDiff3D）的环境探测、按需安装与调用。

包默认不带模型与推理依赖（合计约 12GB），首次使用时下载安装到：
    <包>\python           推理依赖 (torch cu128 + diffusers ...)
    <包>\models           3 个模型仓库 (~6.7GB)

可用 st_env.txt 覆盖（每行 KEY=VALUE），用于复用机器上已有的环境：
    PY=D:\some\.venv\Scripts\python.exe
    REPO=D:\see-through-main
    MODELS=D:\see-through-main\models
"""
import os
import subprocess
import sys

PKG = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(PKG, "st_env.txt")
DEF_PY = os.path.join(PKG, "python", "python.exe")
DEF_REPO = os.path.join(PKG, "seethrough_repo")
DEF_MODELS = os.path.join(PKG, "models")

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_PKGS = ["torch==2.8.0+cu128", "torchvision==0.23.0+cu128"]
ST_PKGS = [
    "transformers==5.0.0", "diffusers==0.37.0", "huggingface-hub>=1.7",
    "accelerate>=1.13", "safetensors>=0.7", "tokenizers>=0.22",
    "bitsandbytes==0.49.2", "einops>=0.8", "kornia>=0.8", "omegaconf==2.3.0",
    "scipy>=1.15", "scikit-image>=0.25", "timm", "simple-parsing==0.1.8",
]
MODEL_JOBS = [
    ("24yearsold/seethroughv0.0.2_layerdiff3d_nf4", "layerdiff3d_nf4", None),
    ("24yearsold/seethroughv0.0.1_marigold_nf4", "marigold_nf4", None),
    # 完整 depth 管线: 每个组件只要一份 safetensors, 不取 .bin/.fp16 重复权重
    ("prs-eth/marigold-depth-v1-1", "marigold-base",
     ["model_index.json", "README.md", "scheduler/*",
      "text_encoder/config.json", "text_encoder/model.safetensors",
      "tokenizer/*", "unet/config.json",
      "unet/diffusion_pytorch_model.safetensors",
      "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]),
    # 只有几百字节的调度器配置, 但管线默认要联网去 SDXL 仓库取,
    # 镜像限流时会报 "does not appear to have a file named scheduler_config.json"
    ("frankjoshua/juggernautXL_version6Rundiffusion", "sdxl_scheduler",
     ["scheduler/*"]),
]

# 可选模型: 不进"就绪"判定(量化版三个目录齐了就算就绪), 但**会被用到** ——
# 界面上勾了「全参数模型」时 infer_cmd 会把 layerdiff 换成本目录里的 bf16 权重。
# 没下载时 full_ready() 返回 False, 自动回退量化版, 不会报错。
# 用 `python st_download.py --full` 下载。
OPTIONAL_JOBS = [
    # 全参数(未量化) layerdiff3d, 对应 --quant_mode none。UNet 单文件 8.1GB,
    # 需要 12GB 以上显存(或靠分组 offload 硬挤)。
    ("layerdifforg/seethroughv0.0.2_layerdiff3d", "layerdiff3d_bf16", None),
]

# 每个仓库必须存在的关键文件 {相对路径: 最小字节数}。
# 只判断"目录非空"会把半成品当成已就绪 —— 必须落到权重文件本身。
REQUIRED = {
    "layerdiff3d_nf4": {
        "unet/diffusion_pytorch_model.safetensors": 2_000_000_000,
        "vae/diffusion_pytorch_model.safetensors": 100_000_000,
        "trans_vae/diffusion_pytorch_model.safetensors": 100_000_000,
        "text_encoder/model.safetensors": 50_000_000,
        "text_encoder_2/model.safetensors": 400_000_000,
    },
    "marigold_nf4": {
        "unet/diffusion_pytorch_model.safetensors": 1_000_000_000,
        "vae/diffusion_pytorch_model.safetensors": 100_000_000,
        "text_encoder/model.safetensors": 200_000_000,
    },
    "marigold-base": {  # 完整 depth 管线(每个组件一份权重)
        "model_index.json": 1,
        "text_encoder/model.safetensors": 1_000_000_000,
        "tokenizer/vocab.json": 500_000,
        "unet/diffusion_pytorch_model.safetensors": 3_000_000_000,
        "vae/diffusion_pytorch_model.safetensors": 300_000_000,
    },
    "layerdiff3d_bf16": {  # 全参数(未量化) layerdiff3d
        "unet/diffusion_pytorch_model.safetensors": 7_000_000_000,
        "vae/diffusion_pytorch_model.safetensors": 100_000_000,
        "trans_vae/diffusion_pytorch_model.safetensors": 100_000_000,
        "text_encoder/model.safetensors": 100_000_000,
        "text_encoder_2/model.safetensors": 800_000_000,
    },
    "sdxl_scheduler": {  # 调度器只有 config, 没有权重
        "scheduler/scheduler_config.json": 50,
    },
}


def cfg():
    d = {}
    if os.path.isfile(CFG):
        for line in open(CFG, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                d[k.strip().upper()] = v.strip()
    return d


def paths():
    c = cfg()
    py = c.get("PY") or (DEF_PY if os.path.isfile(DEF_PY) else sys.executable)
    repo = c.get("REPO") or DEF_REPO
    models = c.get("MODELS") or DEF_MODELS
    return py, repo, models


def deps_ok(py=None):
    py = py or paths()[0]
    try:
        r = subprocess.run(
            [py, "-c", "import torch,diffusers,transformers,bitsandbytes"],
            capture_output=True, timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def models_ok(models=None):
    models = models or paths()[2]
    for _repo, sub, _pat in MODEL_JOBS:
        d = os.path.join(models, sub)
        if not os.path.isdir(d):
            return False
        for rel, minsize in REQUIRED.get(sub, {}).items():
            p = os.path.join(d, rel.replace("/", os.sep))
            if not os.path.isfile(p) or os.path.getsize(p) < minsize:
                return False
    return True


def ready():
    py, repo, models = paths()
    script = os.path.join(repo, "inference", "scripts",
                          "inference_psd_quantized.py")
    return os.path.isfile(script) and deps_ok(py) and models_ok(models)


def dir_ok(models, sub):
    """某个模型目录是否真下全了(按 REQUIRED 的文件+最小体积判断)。"""
    d = os.path.join(models, sub)
    if not os.path.isdir(d):
        return False
    for rel, minsize in REQUIRED.get(sub, {}).items():
        p = os.path.join(d, rel.replace("/", os.sep))
        if not os.path.isfile(p) or os.path.getsize(p) < minsize:
            return False
    return True


def full_ready(models=None):
    """全参数模型(layerdiff3d_bf16)是否已下载完整。没下全就只能走量化版。"""
    return dir_ok(models or paths()[2], "layerdiff3d_bf16")


def model_text(full=False):
    """日志/界面用的一行说明, 同时告诉用户"选了全参数但没权重"的情况。"""
    if not full:
        return "量化版 layerdiff3d_nf4 (快、约 8GB 显存)"
    if not full_ready():
        return ("全参数权重缺失 (models/layerdiff3d_bf16), 已自动回退量化版; "
                "要全参数先跑 python st_download.py --full")
    return "全参数 bf16 layerdiff3d_bf16 (质量最好、需 ~14GB 显存、慢数倍)"


def gpu_free_gb():
    """当前空闲显存(GB); 查不到返回 None。只用来给提示, 不参与任何判定。"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        vals = [float(x) for x in (r.stdout or "").replace("\r", "").split()
                if x.replace(".", "", 1).isdigit()]
        return max(vals) / 1024.0 if vals else None
    except Exception:      # noqa: BLE001
        return None


FULL_NEED_GB = 11.5        # 全参数实测: 8.14GB UNet + VAE/TE(bf16) + 推理中间量


def status_text():
    py, repo, models = paths()
    if not os.path.isfile(os.path.join(repo, "inference", "scripts",
                                       "inference_psd_quantized.py")):
        return "拆图代码缺失: " + repo
    if not deps_ok(py):
        return "未安装 AI 拆图依赖 (torch/diffusers, 约 5GB)"
    if not models_ok(models):
        return "未下载拆图模型 (约 6.7GB)"
    return "AI 拆图就绪"


def _env(extra=None):
    e = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    # 国内直连 huggingface.co 会被 RST (WinError 10054); 若外部没指定就走镜像
    e.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if extra:
        e.update(extra)
    return e


# ------------------------------------------------------------------ 安装步骤
def install_steps():
    """返回 [(label, cmd, env), ...]：装依赖 + 下模型。"""
    py, _repo, models = paths()
    steps = [
        ("安装 PyTorch (CUDA 12.8, 约 3GB)",
         [py, "-m", "pip", "install", "--no-warn-script-location",
          "--index-url", TORCH_INDEX] + TORCH_PKGS, _env()),
        ("安装推理依赖 (约 2GB)",
         [py, "-m", "pip", "install", "--no-warn-script-location"] + ST_PKGS,
         _env()),
        ("下载模型 (约 6.7GB)",
         [py, os.path.join(PKG, "st_download.py")],
         _env({"ST_MODELS": models})),
    ]
    return steps


# ------------------------------------------------------------------ 推理调用
def infer_cmd(image, full=False):
    """平面图 -> see-through 分层。返回 (cmd, env, cwd, out_dir)。

    full=True  全参数(bf16, 未量化) layerdiff3d_bf16 + marigold-base：
               质量最好, 但要 ~14GB 显存、单张图慢好几倍。
    full=False 量化(nf4) layerdiff3d_nf4 + marigold_nf4：快、省显存, 默认。
    由界面上的「全参数模型」复选框决定; 环境变量 SEETHROUGH_QUANT 仍可强制覆盖
    (设成 none/full 就是全参数, nf4 就是量化)。
    """
    py, repo, models = paths()
    name = os.path.splitext(os.path.basename(image))[0]
    script = os.path.join(repo, "inference", "scripts",
                          "inference_psd_quantized.py")
    force = (os.environ.get("SEETHROUGH_QUANT") or "").strip().lower()
    if force in ("none", "full", "bf16"):
        full = True
    elif force == "nf4":
        full = False
    # 选了全参数但权重没下全: 硬跑会在 from_pretrained 时报"找不到文件",
    # 还不如直接退回量化版把图拆出来(界面日志里 model_text 会写明回退了)。
    if full and not full_ready(models):
        full = False
    if full:
        # layerdiff 用全参数; depth 只能用 marigold_nf4 —— 调用方喂给 marigold 的是
        # 12 通道的 framed latent, 而 marigold-base 是 8 通道的普通 UNet,
        # 直接喂会 "expected 8 channels, got 12"。depth 只决定图层前后顺序,
        # 量化与否对最终分层几乎没有影响。
        qmode, ld, dp = "none", "layerdiff3d_bf16", "marigold_nf4"
    else:
        qmode, ld, dp = "nf4", "layerdiff3d_nf4", "marigold_nf4"
    cmd = [py, script, "--srcp", image, "--save_to_psd", "--tblr_split",
           "--resolution", "768", "--quant_mode", qmode,
           "--repo_id_layerdiff", os.path.join(models, ld),
           "--repo_id_depth", os.path.join(models, dp)]
    if qmode == "nf4":
        # 量化版够小, 关掉分组 offload 换速度; 全参数版保留默认的 group_offload,
        # 否则 8.14GB 的 UNet 会在小显存卡上直接 OOM
        cmd.append("--no_group_offload")
    sched = os.path.join(models, "sdxl_scheduler")
    extra = {
        "MARIGOLD_BASE": os.path.join(models, "marigold-base"),
        "ST_MODELS": models,
        "PYTHONPATH": repo + os.pathsep + os.path.join(repo, "common"),
    }
    # 有本地副本就走本地, 避免管线为了几百字节的 config 去联网
    if os.path.isdir(sched):
        extra["SDXL_SCHEDULER_REPO"] = sched
    env = _env(extra)
    out_dir = os.path.join(repo, "workspace", "layerdiff_output", name)
    return cmd, env, repo, out_dir
