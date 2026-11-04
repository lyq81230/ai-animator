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

# 可选模型: **只下载归档, 不参与推理**。
# 不进"就绪"判定, 也不会被 infer_cmd 引用(infer_cmd 固定走 nf4 那两份)。
# 仅在明确要跑未量化基线时手动用 `--quant_mode none` 指定。
# 用 `python st_download.py --full` 下载。
OPTIONAL_JOBS = [
    # 全参数(未量化) layerdiff3d, 对应 --quant_mode none。UNet 单文件 8.1GB,
    # 6GB 显卡放不下, 需要 12GB 以上显存或 CPU offload。
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
def infer_cmd(image):
    """平面图 -> see-through 分层。返回 (cmd, env, cwd, out_dir)。"""
    py, repo, models = paths()
    name = os.path.splitext(os.path.basename(image))[0]
    script = os.path.join(repo, "inference", "scripts",
                          "inference_psd_quantized.py")
    # quant_mode 显式钉成 nf4: 不依赖脚本里 argparse 的默认值。
    # layerdiff3d_bf16(全参数/未量化) 只下载归档, 不参与推理 ——
    # 它的 UNet 单文件 8.14GB, 6GB 显卡根本载入不了。
    cmd = [py, script, "--srcp", image, "--save_to_psd", "--tblr_split",
           "--no_group_offload", "--resolution", "768", "--quant_mode", "nf4",
           "--repo_id_layerdiff", os.path.join(models, "layerdiff3d_nf4"),
           "--repo_id_depth", os.path.join(models, "marigold_nf4")]
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
