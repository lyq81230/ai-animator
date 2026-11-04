#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parts_check.py — 判断一份拆图结果"够不够做动作"。

以前各处用的是**部件总数**门槛(`len(parts) < 20`), 这条件会误杀:
模型每一版给的东西不一样 —— 华雄把头发/五官并进 head 之后只剩 18 个部件,
但 髋/膝/踝×4 + 脚×2 + 大臂×2 + 小臂×2 + 手×2 + 躯干 + 头 + 眼 + 闭眼补丁 + 道具
该有的一件不缺, 照样能出四条动作。结果界面直接判"不可用", 生成按钮点不动。

所以改成**按功能角色查**, 而不是数个数:

    必需: 腿 / 大臂 / 小臂 / 躯干 / 头(或 face)
    可选: 脚、手   —— 很多模型的靴子并进 leg_low、手并进 arm_fore,
                     缺这两个照样能出动作, 只是少一段独立骨

`check_parts()` 返回缺失的**必需**角色(空 = 可用), `missing_optional()` 返回可选项。
"""

# (role, 显示名, 是否必需, 判定)
_ROLE_RULES = (
    ("leg", "腿", True,
     lambda n, c: c.startswith("leg") or n.startswith("leg_")),
    ("arm_upper", "大臂", True,
     lambda n, c: c == "arm_upper" or n.startswith("arm_upper")),
    ("arm_fore", "小臂", True,
     lambda n, c: c == "arm_fore" or n.startswith("arm_fore")),
    ("torso", "躯干/衣服", True,
     lambda n, c: n in ("topwear", "torso", "body") or c in ("topwear", "torso")),
    # 注意别把 headwear(头饰) 当脑袋: 只有真正的 head / face 才算
    ("head", "头/脸", True,
     lambda n, c: n in ("head", "face") or n.startswith("head_") or c in ("head", "face")),
    ("foot", "脚", False,
     lambda n, c: c in ("foot", "footwear") or n.startswith("foot")),
    ("arm_hand", "手", False,
     lambda n, c: c == "arm_hand" or n.startswith("arm_hand")),
)


def _roles_of(parts):
    have = set()
    for p in parts or []:
        n = str(p.get("name") or "").lower()
        c = str(p.get("cls") or "").lower()
        for key, _label, _req, test in _ROLE_RULES:
            try:
                if test(n, c):
                    have.add(key)
            except Exception:                        # noqa: BLE001
                continue
    return have


def check_parts(parts):
    """必需角色里缺哪些(人话列表, 空 = 可以做动作)。"""
    have = _roles_of(parts)
    return [label for key, label, req, _t in _ROLE_RULES if req and key not in have]


def missing_optional(parts):
    have = _roles_of(parts)
    return [label for key, label, req, _t in _ROLE_RULES if not req and key not in have]


def parts_ready(parts):
    return not check_parts(parts)


def note_of(parts, coords=None):
    """给界面看的一行说明: 部件怎么组织的(不是错, 但能解释为什么数目少)。"""
    parts = parts or []
    names = {str(p.get("name") or "").lower() for p in parts}
    merged = []
    if "head" in names and not ({"front_hair", "back_hair", "headwear"} & names):
        merged.append("头发/五官并进 head")
    if "eyes_closed" in names:
        merged.append("死亡闭眼补丁")
    if not ({"footwear_l", "footwear_r", "foot_l", "foot_r"} & names):
        merged.append("靴子并进腿")
    tail = f"（{'、'.join(merged)}）" if merged else ""
    return f"{len(parts)} 个部件{tail}"
