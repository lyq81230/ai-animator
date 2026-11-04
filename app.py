#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Spine 动作生成器 (PySide6)。

两步走, 拆一次图 -> 反复出动作:
  ① 拆图: 分层 PNG 文件夹 / 分层 PSD / 一张平面图(AI 拆图)
          -> output/<名字>/parts/ + coords.json + _split_preview.png(拆图体检单)
  ② 动作: 待机 idle / 行走 walk / 受击 hit / 死亡 death
          -> output/<名字>/<动作>/result.json  (Spine 3.8, 自动骨骼 + 蒙皮 + 动作)
          + _<动作>_loop.gif / _<动作>_once.gif 预览

拆图正确后四种动作可以随便反复生成, 不再重复拆图(平面图的 AI 拆图一张要 2~4 分钟)。
"""
import json
import os
import shutil
import subprocess
import sys

from PySide6.QtCore import QSettings, QSize, Qt, QThread, Signal
from PySide6.QtGui import QMovie, QPixmap
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QSlider, QVBoxLayout, QWidget)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PARTS_SCHEMA = 3        # 与 build_from_seethrough.PARTS_SCHEMA 一致
OUT_ROOT = os.path.join(HERE, "output")
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PREVIEW_PNG = "_split_preview.png"

import seethrough as ST  # noqa: E402


def kind_of(path):
    if os.path.isdir(path):
        return "分层文件夹"
    ext = os.path.splitext(path)[1].lower()
    if ext == ".psd":
        return "分层 PSD"
    if ext in IMG_EXT:
        return "平面图 (需 AI 拆图)"
    return "不支持的文件"


# 动作元信息: 标签 / 默认时长 / 是否循环
ACTION_LABEL = {"idle": "待机", "walk": "行走", "hit": "受击", "death": "死亡"}
# 与 gen_idle_bones._ACTION_DUR 保持一致: 这里填的是"以前的时长",
# 生成时统一除以 SPEED_BASE(0.75) 拉长 -> 播放速度就是以前的 75%。
# 死亡比原来的倒地(1.6)长: 倒下 + 沉降 + 定住, 三段都要时间。
ACTION_DUR = {"idle": 5.0, "walk": 2.0, "hit": 0.9, "death": 2.0}
ACTION_LOOP = {"idle": True, "walk": True, "hit": False, "death": False}

# 生成器参数默认值 (gen_idle_bones.py)
PARAMS = {
    "breath": 0.016, "bob": 2.5, "breath-cycles": 2.0,
    "wind-deg": 7.0, "ribbon-deg": 16.0, "wind-base": 1.0,
    "gust": 0.6, "wave": 1.4, "root-lock": 0.32, "tip-gamma": 1.7,
    "wind-stagger": 0.6, "wind-amp": 0.5, "soft": 0.5,
    # 待机摆臂: 三段必须沿链条【递减】(肩 > 肘 > 腕)。Spine 的子骨是相对父骨的,
    # 世界角 = 肩+肘+腕; 以前反过来写(2.2/3.5/5.0) 世界角就成了 2.2/5.7/10.7,
    # 肩甲几乎不动、手却甩 5 倍 —— 看着就是"肩膀和小臂各走各的"。
    "sh-deg": 3.4, "el-deg": 1.5, "wr-deg": 0.7, "arm-bob": 2.0,
    "arm-cycles": 1.0, "arm-phase": 1.9,
    "brow-follow": 0.10, "blink-dur": 0.13, "blink-depth": 0.88,
    "blink-at": "1.6,3.9",
    "head-yaw": 5.0, "head-cycles": 1.0, "head-turn": 0.5,
    # 行走 / 受击 / 死亡 的主参数
    "step-bob": 3.5, "sway": 4.0, "lean": 2.0, "twist": 2.5,
    "cadence": 2.0, "arm-swing": 9.0, "hair-lag": 6.0,
    "impact": 12.0, "impact-deg": 14.0,
    # 死亡: 86° 是一块木板横拍下去; 74° + 后倒压缩才是"人塌下去",
    # 和 gen_idle_bones 的 --death-deg 默认值保持一致(否则界面生成的和预览的不是一个东西)
    "death-deg": 74.0, "death-vanish": 0.20,
}

# ---------------------------------------------------------------- 分层文件夹导入
_ZORDER = (
    ("back_hair", "houfa", "houtoufa", 0),
    ("headwear", "toushi", 5),
    ("body", "shenti", 10),
    ("topwear", "yifu", "dress", "shang", 12),
    ("neck", "bozi", 14),
    ("arm", "bi", "shou", "hand", 16),
    ("ears", "erduo", 18),
    ("head", "tou", "lian", "face", 20),
    ("nose", "bizi", 24),
    ("mouth", "zui", "lip", 25),
    ("eyewhite", "yanbai", "sclera", 26),
    ("irides", "iris", "pupil", "tongkong", 27),
    ("eye", "yan", 28),
    ("eyelash", "jiemao", "lash", 30),
    ("eyebrow", "meimao", "brow", 32),
    ("front_hair", "qianfa", "liuhai", "bang", "fringe", 40),
)


def _zkey(name):
    n = name.lower()
    for entry in _ZORDER:
        for kw in entry[:-1]:
            if kw in n:
                return entry[-1]
    return 15


def _layout_layers(parts_dir, pngs):
    """无坐标时按语义放置: 估画布, 每层居中/顶对齐, z-order 排序。"""
    from PIL import Image
    sizes = {f: Image.open(os.path.join(parts_dir, f)).size for f in pngs}
    cw = max(s[0] for s in sizes.values())
    ch = max(s[1] for s in sizes.values())
    items = []
    for f in pngs:
        w, h = sizes[f]
        n = os.path.splitext(f)[0]
        z = _zkey(n)
        nl = n.lower()
        x = (cw - w) // 2
        if "neck" in nl or "bozi" in nl:
            y = int(ch * 0.16)
        elif z >= 18 or "hair" in nl or "fa" in nl or "head" in nl or "wear" in nl:
            y = 0
        else:
            y = max(0, ch - h)
        gx = max(2, min(10, round(w / 55)))
        gy = max(2, min(10, round(h / 55)))
        items.append((z, {"name": n, "image": f, "x": int(x), "y": int(y),
                          "width": w, "height": h, "grid": [gx, gy]}))
    items.sort(key=lambda t: t[0])
    return {"fps": 24.0, "canvas": {"w": cw, "h": ch},
            "transform": {"s": 1.0, "tx": 0.0, "ty": 0.0},
            "parts": [it[1] for it in items]}


def import_layer_dir(src_dir, work, log=print):
    """拷 PNG 到 work/parts, 有 coords.json 就用, 没有按文件名语义排层。"""
    parts = os.path.join(work, "parts")
    os.makedirs(parts, exist_ok=True)
    pngs = [f for f in sorted(os.listdir(src_dir))
            if f.lower().endswith(".png") and not f.startswith("_")]
    if not pngs:
        raise RuntimeError("文件夹里没有 PNG 图层")
    for f in pngs:
        shutil.copy(os.path.join(src_dir, f), os.path.join(parts, f))
    src_coords = os.path.join(src_dir, "coords.json")
    if os.path.isfile(src_coords):
        coords = json.load(open(src_coords, encoding="utf-8"))
        log(f"沿用文件夹里的 coords.json ({len(coords['parts'])} 部件)")
    else:
        coords = _layout_layers(parts, pngs)
        log(f"按文件名语义自动排层: {len(pngs)} 张 PNG")
    dst = os.path.join(work, "coords.json")
    json.dump(coords, open(dst, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    shutil.copy(dst, os.path.join(parts, "coords.json"))


# ---------------------------------------------------------------- 拖拽区
class DropZone(QFrame):
    dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(90)
        self.setStyleSheet(
            "QFrame{border:2px dashed #888;border-radius:8px;background:#2b2b2b;}"
            "QLabel{color:#ddd;border:none;}")
        lay = QVBoxLayout(self)
        self.title = QLabel("角色素材")
        self.title.setAlignment(Qt.AlignCenter)
        self.info = QLabel("把【分层 PNG 文件夹】/【分层 PSD】/【一张平面图】拖到这里\n"
                           "(点击=选 PSD/图片, Shift+点击=选文件夹)")
        self.info.setAlignment(Qt.AlignCenter)
        self.info.setWordWrap(True)
        lay.addWidget(self.title)
        lay.addWidget(self.info)
        self.path = ""

    def mousePressEvent(self, e):
        if e.modifiers() & Qt.ShiftModifier:
            d = QFileDialog.getExistingDirectory(self, "选择分层 PNG 文件夹", "")
            if d:
                self._set(d)
            return
        f, _ = QFileDialog.getOpenFileName(
            self, "选择 PSD 或平面图", "",
            "素材 (*.psd *.png *.jpg *.jpeg *.webp *.bmp)")
        if f:
            self._set(f)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            f = u.toLocalFile()
            ext = os.path.splitext(f)[1].lower()
            if os.path.isdir(f) or ext == ".psd" or ext in IMG_EXT:
                self._set(f)
                return

    def _set(self, f):
        self.path = f
        self.title.setText(os.path.basename(f.rstrip("\\/")))
        self.info.setText(kind_of(f))
        self.dropped.emit(f)


# ---------------------------------------------------------------- 预览面板
class PreviewPane(QLabel):
    """图片按面板大小自适应; 也能播 GIF; 双击用系统看图打开。"""

    def __init__(self, text):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(360)
        self.setWordWrap(True)
        self.setStyleSheet("background:#222;color:#888;border:1px solid #333;")
        self.setToolTip("双击用系统看图打开")
        self._img = ""
        self._movie = None

    def show_image(self, path):
        if self._movie:
            self._movie.stop()
            self.setMovie(None)
            self._movie = None
        self._img = path
        self.setText("")
        self._render()

    def show_movie(self, gif):
        self._img = ""
        self.setText("")
        mv = QMovie(gif)
        self.setMovie(mv)
        mv.start()
        self._movie = mv

    def _render(self):
        if not self._img or not os.path.isfile(self._img):
            return
        pm = QPixmap(self._img)
        if pm.isNull():
            return
        box = QSize(max(1, self.width() - 10), max(1, self.height() - 10))
        self.setPixmap(pm.scaled(box, Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()

    def mouseDoubleClickEvent(self, e):
        if self._img and os.path.isfile(self._img):
            os.startfile(self._img)  # noqa: S606


# ---------------------------------------------------------------- 工作线程
class Worker(QThread):
    log = Signal(str)
    step = Signal(str)
    done = Signal(bool, str)

    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def run(self):
        for st in self.steps:
            self.step.emit(st["label"])
            self.log.emit(f"\n===== {st['label']} =====")
            if st["type"] == "py":
                try:
                    st["fn"](self.log.emit)
                except Exception as ex:  # noqa: BLE001
                    self.done.emit(False, f"{st['label']} 失败: {ex}")
                    return
                continue
            cmd = st["cmd"]
            self.log.emit("$ " + subprocess.list2cmdline(cmd))
            env = st.get("env") or dict(os.environ, PYTHONIOENCODING="utf-8",
                                        PYTHONUNBUFFERED="1")
            p = subprocess.Popen(
                cmd, cwd=st.get("cwd") or HERE, env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=1, text=True,
                encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in p.stdout:
                self.log.emit(line.rstrip("\n"))
            p.wait()
            if p.returncode != 0:
                self.done.emit(False, f"{st['label']} 失败 (退出码 {p.returncode})")
                return
        self.done.emit(True, "完成")


# ---------------------------------------------------------------- 主窗
HINT = {
    "idle": "呼吸起伏 + 2.5D 转头 + 风吹发丝 + 微摆臂；可无缝循环",
    "walk": "重心上下起伏 + 左右移重心 + 躯干前倾反扭 + 摆臂 + 发丝滞后；原地循环",
    "hit":  "瞬间后仰击退 + 衰减回弹；一次性，末帧归位可直接接回待机",
    "death": "卸力屈膝 → 塌倒 → 触地压扁 → 四肢瘫软 → 闭眼 → 化掉消失；一次性。\n"
             "结尾 20% 把整个人缩到看不见（最后几帧画面全空），播完直接销毁节点即可",
}


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spine 动作生成器 — 拆图 → 待机 / 行走 / 受击 / 死亡")
        # 窗口不要写死大小: 1080p 笔记本(算上任务栏/125% 缩放)可用高度常常只有
        # 700 出头, 1180x800 会把底部那排按钮顶到屏幕外。按可用桌面自适应,
        # 上下各留 40px 余量; 左栏本来就带滚动条, 再小也能看全。
        _scr = QApplication.primaryScreen()
        _av = _scr.availableGeometry() if _scr else None
        _w, _h = 1260, 900
        if _av is not None:
            _w = min(_w, max(900, _av.width() - 80))
            _h = min(_h, max(640, _av.height() - 80))
        self.resize(_w, _h)
        self.setMinimumSize(880, 620)
        self.worker = None
        self.mode = ""          # "split" / "action" / "install"
        self.base = ""          # 素材名
        self.work = ""          # output/<素材名>
        self.parts = None       # 已拆图的 coords dict, None = 还没拆
        self.action = "idle"
        self.busy = False
        self._last = None
        self._dur_memo = {}

        root = QWidget()
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        left = QVBoxLayout()
        right = QVBoxLayout()
        # 左栏放进滚动容器: 小窗口下也不会把底部按钮/状态挤掉
        leftw = QWidget()
        leftw.setLayout(left)
        left.setContentsMargins(2, 2, 6, 2)
        left.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(leftw)
        lay.addWidget(scroll, 3)
        lay.addLayout(right, 4)

        self.drop = DropZone()
        self.drop.dropped.connect(self._on_drop)
        left.addWidget(self.drop)

        # ---------------- ① 拆图 ----------------
        sp = QGroupBox("① 拆图（先拆，确认正确再做动作）")
        sg = QGridLayout(sp)
        sg.setSpacing(6)
        self.btn_split = QPushButton("① 开始拆图")
        self.btn_split.setMinimumHeight(38)
        self.btn_split.clicked.connect(self.run_split)
        sg.addWidget(self.btn_split, 0, 0, 1, 2)
        self.btn_check = QPushButton("检查拆图")
        self.btn_check.setToolTip("在右边看拆图体检单：合成图 + 每个部件的名字/角色")
        self.btn_check.clicked.connect(lambda: self._show_split_preview(True))
        sg.addWidget(self.btn_check, 1, 0)
        self.split_state = QLabel("未拆图")
        self.split_state.setWordWrap(True)
        sg.addWidget(self.split_state, 1, 1)
        self.settings = QSettings("ai_spine", "SpineGen")
        wear_row = QHBoxLayout()
        wear_row.addWidget(QLabel("头部装饰"))
        self.head_wear = QComboBox()
        self.head_wear.addItem("有头发（会飘）", "hair")
        self.head_wear.addItem("头巾（可飘）", "scarf")
        self.head_wear.addItem("头盔（刚性）", "helmet")
        self.head_wear.setToolTip(
            "前发/后发/头饰这几个槽位到底是什么:\n"
            "· 有头发: 刘海/马尾, 正常风吹飘动(拆成前发后发)\n"
            "· 头巾: 布做的, 包在头上的那圈钉死, 垂下来的下摆会飘; 整块合并不再有接缝\n"
            "· 头盔: 硬壳, 整块刚性跟随头部, 一点都不飘")
        mode0 = str(self.settings.value("head_wear", "") or "")
        if mode0 not in ("hair", "scarf", "helmet"):
            mode0 = "helmet" if bool(self.settings.value("no_hair", False)) else "hair"
        self.head_wear.setCurrentIndex(max(0, self.head_wear.findData(mode0)))
        self.head_wear.currentIndexChanged.connect(self._on_head_wear_changed)
        wear_row.addWidget(self.head_wear, 1)
        sg.addLayout(wear_row, 2, 0, 1, 2)
        left.addWidget(sp)

        # ---------------- ② 动作 ----------------
        box = QGroupBox("② 动作（拆图后可用，可反复生成）")
        g = QGridLayout(box)
        g.setSpacing(6)
        self.act_btns = {}
        self._act_touched = False       # 用户是否手动动过勾选(首次点击=单选, 之后=开关)
        self.act_group = QButtonGroup(self)
        self.act_group.setExclusive(False)      # 可多选: 勾几个就一次生成几个动作
        for i, (key, label) in enumerate(ACTION_LABEL.items()):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setMinimumHeight(38)
            # 默认勾「行走 / 受击 / 死亡」, 待机不勾: 这三条是每个角色都要的。
            # 高亮的才是会生成的, 没有任何"首次点击自动增删"的暗规则。
            b.setChecked(key != "idle")
            b.setToolTip("点一下 = 勾上，再点一下 = 取消；勾几个就一次生成几条，"
                         "合并成同一个骨架\n"
                         "默认勾的是 行走/受击/死亡（待机要手动勾上）\n"
                         "想全要就按右边的【全选动作】")
            # 多选必须看得出来: 勾上的按钮给个明确的高亮底色
            b.setStyleSheet(
                "QPushButton:checked{background:#2f6ecb;color:#fff;"
                "border:1px solid #1e4f9a;font-weight:600;}")
            b.clicked.connect(lambda _=False, k=key: self._on_act_clicked(k))
            self.act_group.addButton(b)
            g.addWidget(b, i // 2, i % 2)
            self.act_btns[key] = b
        act_row = QHBoxLayout()
        self.btn_all_act = QPushButton("全选动作")
        self.btn_all_act.setToolTip("待机 + 行走 + 受击 + 死亡 一次全出，合并成一个 Spine 骨架")
        self.btn_all_act.clicked.connect(self._check_all_actions)
        act_row.addWidget(self.btn_all_act)
        self.cb_keep_act = QCheckBox("保留已有动作")
        self.cb_keep_act.setToolTip(
            "默认：Spine 包里只放这次勾选的动作（没勾待机就不会有待机）。\n"
            "勾上：把之前已经生成过的动作也一并保留在同一个文件里。")
        # 每次启动都从"不合并"开始。曾经把这个状态存进注册表, 结果只要勾过一次,
        # 之后每次生成都会悄悄把上次的旧动作并回来 —— 表现就是"没勾待机却出现待机"。
        # 这个开关影响的是"文件里到底有几条动画", 不该被上一次的选择暗中决定。
        self.cb_keep_act.setChecked(False)
        act_row.addWidget(self.cb_keep_act)
        act_hint = QLabel("可多选，合成一个骨架")
        act_hint.setStyleSheet("color:#8a8f98;")
        act_row.addWidget(act_hint, 1)
        g.addLayout(act_row, 2, 0, 1, 2)
        left.addWidget(box)

        self.hint = QLabel(HINT[self.action])
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#9aa;")
        left.addWidget(self.hint)

        # ---------------- 时长 / 帧率 ----------------
        row = QHBoxLayout()
        row.addWidget(QLabel("时长(秒)"))
        self.dur = QDoubleSpinBox()
        self.dur.setRange(0.3, 20.0)
        self.dur.setSingleStep(0.1)
        self.dur.setDecimals(1)
        self.dur.setValue(ACTION_DUR[self.action])
        row.addWidget(self.dur)
        row.addWidget(QLabel("帧率"))
        self.fps = QComboBox()
        self.fps.addItems(["24", "30", "60"])
        self.fps.setCurrentText("24")
        row.addWidget(self.fps)
        self.cb_blink = QCheckBox("眨眼")
        self.cb_blink.setChecked(True)
        row.addWidget(self.cb_blink)
        row.addStretch()
        left.addLayout(row)

        # ---------------- 幅度 / 速度 ----------------
        # 七个滑块以前是一列排下来(每个占一整行), 光这一块就 300px —— 在 1080p
        # 的笔记本上会被挤出窗口底部。改成两列网格, 高度省一半。
        amp = QGroupBox("幅度 / 速度")
        al = QGridLayout(amp)
        al.setHorizontalSpacing(12)
        al.setVerticalSpacing(4)
        self.s_act = self._slider_cell(al, 0, 0, "动作幅度", 100, 0, 200)
        self.s_wind = self._slider_cell(al, 0, 1, "风吹/发丝", 100)
        self.s_breath = self._slider_cell(al, 1, 0, "呼吸幅度", 100)
        self.s_head = self._slider_cell(al, 1, 1, "转头幅度", 100)
        self.s_arm_l = self._slider_cell(al, 2, 0, "左臂摆动", 100)
        self.s_arm_l.setToolTip("画面左侧那条手臂的摆幅（正面角色 = 角色自己的右手）")
        self.s_arm_r = self._slider_cell(al, 2, 1, "右臂摆动", 100)
        self.s_arm_r.setToolTip("画面右侧那条手臂的摆幅（正面角色 = 角色自己的左手）")
        self.s_speed = self._slider_cell(al, 3, 0, "播放速度", 100, 20, 300)
        self.s_speed.setToolTip("100% = 以前播放速度的 75%")
        al.addWidget(QLabel("左右按画面方向：画面左 = 角色的右手"),
                     3, 1)
        left.addWidget(amp)

        left.addStretch()          # 弹性留白放在上方, 主按钮锚到底部

        self.out_hint = QLabel()
        self.out_hint.setWordWrap(True)
        self.out_hint.setStyleSheet("color:#7a9;")
        left.addWidget(self.out_hint)

        self.btn = QPushButton(f"② 生成 {ACTION_LABEL[self.action]}")
        self.btn.setMinimumHeight(44)
        self.btn.clicked.connect(self.run_action)
        left.addWidget(self.btn)

        self.btn_open = QPushButton("打开输出目录")
        self.btn_open.clicked.connect(self.open_out)
        self.btn_open.setEnabled(False)
        left.addWidget(self.btn_open)

        self.btn_install = QPushButton("安装 AI 拆图组件 (约 12GB)")
        self.btn_install.setToolTip(
            "平面图自动拆层需要 PyTorch + LayerDiff3D 模型, 需 NVIDIA 显卡")
        self.btn_install.clicked.connect(self.install_st)
        left.addWidget(self.btn_install)

        self.status = QLabel("等待素材")
        self.status.setWordWrap(True)
        left.addWidget(self.status)

        self.preview = PreviewPane(
            "拆完图这里显示拆图体检单；生成动作后显示动画预览")
        right.addWidget(self.preview, 3)
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setStyleSheet(
            "background:#1b1b1b;color:#ccc;font-family:Consolas;")
        right.addWidget(self.logbox, 2)

        # 当前动作 = 默认勾上的第一条(行走): 时长框/提示/眨眼开关都跟着它
        self.action = ""
        self.set_action("walk")
        self._refresh_controls()
        self._refresh_out()

    # ------------------------------------------------------------ 控件
    def _slider_cell(self, grid, r, c, label, val, lo=0, hi=250):
        """两列网格里的一个小滑块: 左边标题(定宽) + 右边滑条。

        标题里带数值(100%), 所以"动作幅度 100%"不会再多占一行。
        """
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        cap = QLabel(f"{label} {val}%")
        cap.setMinimumWidth(88)
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.valueChanged.connect(lambda v, cc=cap, t=label: cc.setText(f"{t} {v}%"))
        h.addWidget(cap)
        h.addWidget(s, 1)
        grid.addWidget(w, r, c)
        return s

    # ------------------------------------------------------------ 素材 / 拆图状态
    def _on_drop(self, f):
        self.status.setText("已选: " + kind_of(f))
        self.parts = None
        self.base = os.path.splitext(os.path.basename(f.rstrip("\\/")))[0]
        self.work = os.path.join(OUT_ROOT, self.base)
        self._load_split()
        adopted = False
        if not self.parts and self._adopt_legacy_split():
            self._load_split()
            adopted = True
            self._post_adopt_fix()
        if self.parts:
            mode = self.parts.get("head_wear")
            if not mode:                       # 老结果只有 hair 布尔
                mode = "hair" if self.parts.get("hair", True) else "helmet"
            if self.head_wear.currentData() != mode:
                # 已有拆图自带模式: 让下拉跟着它走, 不去改人家的结果
                self.head_wear.blockSignals(True)
                self.head_wear.setCurrentIndex(max(0, self.head_wear.findData(mode)))
                self.head_wear.blockSignals(False)
            elif "head_wear" not in self.parts:
                self._apply_head_wear()        # 老结果没记录模式, 按当前选择补上
        self._refresh_controls()
        self._refresh_out()
        if self.parts:
            self.logbox.clear()
            self.logbox.appendPlainText(
                f"{'沿用上次拆图' if adopted else '发现已有拆图'}: "
                f"{self.work}\\parts  ({len(self.parts['parts'])} 部件)\n"
                "可以直接生成动作；想重做拆图点【① 重新拆图】。")
            self._show_split_preview(False)

    def _post_adopt_fix(self):
        """沿用旧拆图时补做的两步修整(正常情况下 build 里已经做过了)。

        旧结果是用老代码拆的: 头饰可能带着五官/额头皮肤, 手臂可能带着躯干。
        手臂要改就得重拆(拆图时才做), 头饰可以就地修 —— 只是重新抠一遍 png。
        """
        if not self.parts or not self.work or not self.drop.path:
            return
        mode = self.parts.get("head_wear") or "hair"
        if mode not in ("scarf", "helmet") or not os.path.isfile(self.drop.path):
            return
        try:
            import fix_headwear
            fix_headwear.main(self.work, self.drop.path)
        except Exception as ex:                       # noqa: BLE001
            self.logbox.appendPlainText(f"头饰修整跳过: {ex}")
            return
        # fix_headwear 只写 parts/coords.json, 界面读的是 <work>/coords.json
        try:
            shutil.copyfile(os.path.join(self.work, "parts", "coords.json"),
                            os.path.join(self.work, "coords.json"))
        except OSError:
            pass
        self._load_split()
        self.logbox.appendPlainText(
            "头饰已按新规则修整（摘掉混进来的五官/额头皮肤）。\n"
            "手臂若仍带躯干，点【① 重新拆图】会按新判据重切（有 AI 分层缓存，很快）。")

    def _adopt_legacy_split(self):
        """旧版输出是 output/<名字>_<动作>/parts: 自动搬到 output/<名字>/, 免重拆。

        两个硬性前提, 缺一不可:
        - 工作目录里**还没有** coords.json(绝不覆盖已经拆好的结果);
        - 旧结果的结构版本与当前一致(否则搬进来的就是"没腿/头巾是碎片"的老数据,
          生成出来必然缺件 —— 这正是之前反复出现的问题)。
        """
        if not self.base:
            return False
        if os.path.isfile(os.path.join(self.work, "coords.json")):
            return False
        for act in ACTION_LABEL:
            old = os.path.join(OUT_ROOT, f"{self.base}_{act}")
            if not (os.path.isdir(os.path.join(old, "parts"))
                    and os.path.isfile(os.path.join(old, "coords.json"))):
                continue
            try:
                lj = json.load(open(os.path.join(old, "coords.json"), encoding="utf-8"))
            except (OSError, ValueError):
                continue
            from parts_check import check_parts
            _miss = check_parts(lj.get("parts") or [])
            if lj.get("schema") != PARTS_SCHEMA or _miss:
                self.logbox.appendPlainText(
                    f"忽略旧拆图 {os.path.basename(old)}（结构版本 {lj.get('schema')}，"
                    f"缺 {'、'.join(_miss) or '—'}）—— 请点【① 重新拆图】。")
                continue
            try:
                os.makedirs(self.work, exist_ok=True)
                shutil.copytree(os.path.join(old, "parts"),
                                os.path.join(self.work, "parts"),
                                dirs_exist_ok=True)
                shutil.copyfile(os.path.join(old, "coords.json"),
                                os.path.join(self.work, "coords.json"))
            except OSError:
                continue
            return True
        return False

    def _load_split(self):
        """读 output/<名字>/coords.json 并校验图片齐全 —— 齐全才算「拆图正确可用」。"""
        self.parts = None
        if not self.work:
            return
        cpath = os.path.join(self.work, "coords.json")
        if not os.path.isfile(cpath):
            return
        try:
            coords = json.load(open(cpath, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        ps = coords.get("parts") or []
        if not ps:
            return
        parts_dir = os.path.join(self.work, "parts")
        for p in ps:
            if not os.path.isfile(os.path.join(parts_dir, p.get("image", ""))):
                self.logbox.appendPlainText(
                    f"拆图不完整: 缺 {p.get('image')}，请重新拆图")
                return
        # 拆图体检: 以前这里卡的是**部件总数**(<20 就判不可用), 但模型每版给的
        # 东西不一样 —— 华雄把头发/五官并进 head 之后只剩 18 件, 该有的一样不缺,
        # 却被拦在门外, 界面就变成"拆了也用不了、按钮点不动"。
        # 改成按功能角色查(腿/脚/大臂/小臂/手/躯干/头), 少什么说什么。
        from parts_check import check_parts
        _miss = check_parts(ps)
        if _miss:
            self.logbox.appendPlainText(
                "这份拆图结果不可用（缺 " + "、".join(_miss) +
                "）：请点【① 重新拆图】重新拆一遍再生成。")
            return
        self.parts = coords

    def _stale(self):
        """素材改过但拆图没更新 -> 提示重新拆图(仍允许用)。"""
        if not self.parts or not self.drop.path or not os.path.isfile(self.drop.path):
            return False
        try:
            return (os.path.getmtime(self.drop.path)
                    > os.path.getmtime(os.path.join(self.work, "coords.json")) + 2)
        except OSError:
            return False

    def _checked_actions(self):
        """按固定顺序返回已勾选的动作(Spine 里的动画顺序也就是这个)。"""
        return [k for k in ACTION_LABEL if self.act_btns[k].isChecked()]

    def _on_act_clicked(self, key):
        """点按钮 = 勾选/取消, 支持多选。点几下就有几个, 不做任何"自动帮你加/减"。

        以前这里有一条"第一次点击先取消启动时默认勾上的待机"，本意是修
        "没选待机却冒出待机"。结果反着坑人: 用户依次点 行走→受击→死亡, 待机被
        悄悄取消掉, 导出来只有三条, 表现就是"未生成需要的动作"。
        现在启动时**一个都不勾**, 谁勾谁算, 界面上高亮的就是会被生成的那些。
        """
        btn = self.act_btns[key]
        if not self._checked_actions():            # 不允许一个都不选
            btn.setChecked(True)
        if btn.isChecked():
            self.set_action(key)
        else:
            self.set_action(self._checked_actions()[0])
        self._refresh_controls()
        self._refresh_out()
        n = len(self._checked_actions())
        self.status.setText(
            f"已勾选 {n} 个动作: " + " / ".join(
                ACTION_LABEL[k] for k in self._checked_actions()))

    def _check_all_actions(self):
        self._act_touched = True
        for k in ACTION_LABEL:
            self.act_btns[k].setChecked(True)
        self._refresh_controls()
        self._refresh_out()
        self.status.setText("已全选 4 个动作：生成后是一个 Spine 骨架 + 4 条动画")

    def _refresh_controls(self):
        ready = bool(self.parts)
        keys = self._checked_actions()
        self.btn_split.setEnabled(bool(self.drop.path) and not self.busy)
        self.btn_split.setText("① 重新拆图" if ready else "① 开始拆图")
        self.btn_check.setEnabled(ready and not self.busy)
        for b in self.act_btns.values():
            b.setEnabled(ready and not self.busy)
        self.btn_all_act.setEnabled(ready and not self.busy)
        self.cb_keep_act.setEnabled(ready and not self.busy)
        # 一个动作都没勾就点不动生成 —— 比"悄悄替你补一个"清楚得多
        self.btn.setEnabled(ready and bool(keys) and not self.busy)
        if keys:
            self.btn.setText("② 生成 " + " / ".join(ACTION_LABEL[k] for k in keys)
                             + (f"（{len(keys)} 个动作）" if len(keys) > 1 else ""))
            self.btn.setToolTip("" if ready else "先完成 ① 拆图")
        else:
            self.btn.setText("② 生成（先勾一个动作）")
            self.btn.setToolTip("上面【② 动作】里至少勾一个要生成的动作")
        self.btn_open.setEnabled(ready and not self.busy)
        self.btn_install.setEnabled(not self.busy)
        self.head_wear.setEnabled(not self.busy)

        if not self.drop.path:
            self.split_state.setText("未选素材")
            self.split_state.setStyleSheet("color:#8a8f98;")
        elif not ready:
            self.split_state.setText("未拆图 · 先点上面的拆图")
            self.split_state.setStyleSheet("color:#e0a33a;")
        else:
            from parts_check import note_of
            note = note_of(self.parts.get("parts") or [], self.parts)
            if self._stale():
                self.split_state.setText(
                    f"已拆图 · {note}（素材比拆图新，建议重拆）")
                self.split_state.setStyleSheet("color:#e0a33a;")
            else:
                self.split_state.setText(f"已拆图 · {note} · 可做动作")
                self.split_state.setStyleSheet("color:#7dbb8a;")
        self.hint.setStyleSheet("color:#9aa;" if ready else "color:#666;")

    def _refresh_out(self):
        if not self.drop.path:
            self.out_hint.setText("")
            return
        keys = self._checked_actions()
        if not keys:
            self.out_hint.setText("还没勾动作 —— 上面【② 动作】里点一个（可多选）")
            return
        names = "、".join(ACTION_LABEL[k] for k in keys)
        if len(keys) > 1:
            self.out_hint.setText(
                f"本次只生成: {names}（{len(keys)} 条）\n"
                f"拆图: output/{self.base}/parts/\n"
                f"导出: output/{self.base}/spine/  ← png+atlas+json, 合并成 1 个骨架")
        else:
            self.out_hint.setText(
                f"本次只生成: {names}\n"
                f"拆图: output/{self.base}/parts/\n"
                f"动作: output/{self.base}/{keys[0]}/result.json\n"
                f"导出: output/{self.base}/spine/ (png + atlas + json)")

    def set_action(self, key):
        """切"当前动作"：只决定提示/时长/预览，不改其它动作的勾选。"""
        if not self.act_btns[key].isChecked():
            self.act_btns[key].setChecked(True)
        if key != self.action:
            if self.action:              # 记住上一个动作的时长, 切回来不丢用户调整
                self._dur_memo[self.action] = self.dur.value()
            self.action = key
            self.dur.setValue(self._dur_memo.get(key, ACTION_DUR[key]))
        self.cb_blink.setEnabled(key in ("idle", "walk"))
        self.hint.setText(HINT[key])
        self.status.setText(f"当前动作: {ACTION_LABEL[key]}")
        self._refresh_controls()
        self._refresh_out()

    def _set_busy(self, b):
        self.busy = b
        self._refresh_controls()

    # ------------------------------------------------------------ 有头发 / 无头发
    def _on_head_wear_changed(self, _idx):
        mode = self.head_wear.currentData()
        self.settings.setValue("head_wear", mode)
        if self.parts:
            self._apply_head_wear()

    def _apply_head_wear(self):
        """把头部装饰模式落到已拆好的 coords 上 —— 改分类不用重拆。"""
        if not self.parts or not self.work:
            return
        from build_from_seethrough import set_head_wear, _HAIR_NAME_KW
        mode = self.head_wear.currentData()
        coords = set_head_wear(self.parts, mode)
        self.parts = coords
        for path in (os.path.join(self.work, "coords.json"),
                     os.path.join(self.work, "parts", "coords.json")):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(coords, f, ensure_ascii=False, indent=2)
            except OSError as ex:
                self.logbox.appendPlainText(f"写坐标失败: {ex}")
        # 合并/改分类之后重跑一次头饰修整: 它会按像素把"头盔"和"垂布"再拆开
        # (盔刚性、布可飘)。不跑的话, 换一次模式头盔就又会跟着布飘。
        if self.drop.path and os.path.isfile(self.drop.path):
            try:
                import importlib
                import fix_headwear
                importlib.reload(fix_headwear)
                fix_headwear.main(self.work, self.drop.path)
                with open(os.path.join(self.work, "parts", "coords.json"),
                          encoding="utf-8") as f:
                    self.parts = json.load(f)
                with open(os.path.join(self.work, "coords.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(self.parts, f, ensure_ascii=False, indent=2)
            except Exception as ex:                   # noqa: BLE001
                self.logbox.appendPlainText(f"头饰修整跳过: {ex}")
        txt = {"hair": "有头发: 按头发处理, 会飘",
               "scarf": "头巾: 包在头上的那圈钉死, 下摆会飘",
               "helmet": "头盔: 整块刚性跟随头部, 不飘"}.get(mode, mode)
        self.logbox.appendPlainText("头部装饰 → " + txt)
        n_hair = sum(1 for p in coords["parts"]
                     if any(k in p["name"].lower() for k in _HAIR_NAME_KW))
        if mode != "hair" and n_hair > 1:
            self.logbox.appendPlainText(
                f"提示: 现在还是 {n_hair} 块碎片分开的。点一次【① 重新拆图】"
                "会合并成一整块（AI 分层有缓存，几秒）")
        self._show_split_preview(True)

    # ------------------------------------------------------------ ① 拆图
    def _copy_step(self, label, src, dst):
        def fn(log, a=src, b=dst):
            shutil.copyfile(a, b)
            log(b)
        return {"type": "py", "label": label, "fn": fn}

    def _split_steps(self, src):
        """按素材类型拆图 -> work/parts + work/coords.json。不生成任何动作。"""
        ext = os.path.splitext(src)[1].lower()
        wc = os.path.join(self.work, "coords.json")
        pc = os.path.join(self.work, "parts", "coords.json")
        if os.path.isdir(src):
            return [{"type": "py", "label": "导入分层文件夹",
                     "fn": lambda log, s=src: import_layer_dir(s, self.work, log)}]
        if ext == ".psd":
            return [
                {"type": "cmd", "label": "拆分 PSD 图层", "cmd": [
                    sys.executable, os.path.join(HERE, "psd_split.py"),
                    "--psd", src,
                    "--parts", os.path.join(self.work, "parts"),
                    "--coords", wc]},
                self._copy_step("整理坐标 (parts/coords.json)", wc, pc),
            ]
        if ext in IMG_EXT:
            cmd, env, cwd, out_dir = ST.infer_cmd(src)
            # AI 推理是唯一慢的一步(2-4 分钟)。同一张图重拆时, 只要上次的分层结果
            # 还在且比原图新, 就直接复用它, 只重跑"分层转部件"(改图层白名单后必用)。
            layers = []
            if os.path.isdir(out_dir):
                layers = [f for f in os.listdir(out_dir)
                          if f.endswith(".png") and not f.endswith("_depth.png")
                          and not f.startswith(("src_", "reconstruction"))]
            fresh = (len(layers) >= 6
                     and os.path.getmtime(out_dir) >= os.path.getmtime(src) - 2)
            if fresh:
                steps = [{"type": "py",
                          "label": "复用上次 AI 分层结果 (跳过 GPU 推理)",
                          "fn": lambda log, d=out_dir, n=len(layers):
                              log(f"命中缓存: {d}  ({n} 层)\n"
                                  "跳过 2-4 分钟的 AI 拆图, 直接重新组装部件")}]
            else:
                if not ST.ready():
                    self.status.setText(ST.status_text())
                    self.logbox.appendPlainText(
                        "平面图需要 AI 拆图组件: " + ST.status_text() +
                        "\n点下面的【安装 AI 拆图组件】按钮先装好再来。")
                    return None
                steps = [{"type": "cmd", "label": "AI 自动拆图 (GPU, 约 2-4 分钟)",
                          "cmd": cmd, "env": env, "cwd": cwd}]
            steps.append({"type": "cmd", "label": "分层转部件", "cmd": [
                sys.executable, os.path.join(HERE, "build_from_seethrough.py"),
                "--src", out_dir, "--out", os.path.join(self.work, "parts"),
                "--orig-image", src,
                "--head-wear", str(self.head_wear.currentData())]})
            # 腿**不再**在膝高切上下两片(--leg-halves)。切了之后两片在膝部有 8px
            # 重叠, 渲染时两块都画一遍, alpha 叠加出一条比上下都亮的横带 ——
            # 用户:"腿部与躯体相接处一条与上下明显区别的区域"。
            # 整条腿 + 顶点权重渐变(髋->膝->踝)一样能屈膝, 而且没有接缝。
            steps.append(self._copy_step("整理坐标 (coords.json)", pc, wc))
            return steps
        self.status.setText("不支持的文件类型")
        return None

    def run_split(self):
        src = self.drop.path
        if not src:
            self.status.setText("先拖入素材")
            return
        if self.parts:
            r = QMessageBox.question(
                self, "重新拆图",
                f"已有拆图结果（{len(self.parts['parts'])} 部件）。\n"
                "重新拆图会覆盖 parts/ 和坐标，已生成的动作目录不动。\n\n要继续吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        self.base = os.path.splitext(os.path.basename(src.rstrip("\\/")))[0]
        self.work = os.path.join(OUT_ROOT, self.base)
        os.makedirs(self.work, exist_ok=True)
        self.logbox.clear()

        steps = self._split_steps(src)
        if steps is None:
            return
        steps.append({"type": "py", "label": "拆图体检单 (预览图)",
                      "fn": self._make_preview})

        self.mode = "split"
        self.parts = None
        self._set_busy(True)
        self.preview.setText("拆图中…")
        self.worker = Worker(steps)
        self.worker.log.connect(self.logbox.appendPlainText)
        self.worker.step.connect(self.status.setText)
        self.worker.done.connect(self.finish)
        self.worker.start()

    def _make_preview(self, log):
        from split_preview import make_preview
        info = make_preview(os.path.join(self.work, "parts"),
                            os.path.join(self.work, "coords.json"),
                            os.path.join(self.work, PREVIEW_PNG), log=log)
        if info["n"] < 3:
            log("警告: 部件太少，检查素材是否分层/拆图是否失败")
        return info

    def _show_split_preview(self, rebuild=False):
        if not self.work:
            return
        p = os.path.join(self.work, PREVIEW_PNG)
        if rebuild or not os.path.isfile(p):
            try:
                self._make_preview(self.logbox.appendPlainText)
            except Exception as ex:  # noqa: BLE001
                self.status.setText(f"预览生成失败: {ex}")
                return
        if os.path.isfile(p):
            self.preview.show_image(p)

    # ------------------------------------------------------------ ② 动作
    def run_action(self):
        self._load_split()          # 生成前再验一次: 部件图被别人删了要拦住
        if not self.parts:
            self._refresh_controls()
            self.status.setText("先完成 ① 拆图（拆图正确后再生成动作）")
            return
        self.logbox.clear()
        self.logbox.appendPlainText(
            f"沿用已拆图结果: {self.work}\\parts  "
            f"({len(self.parts['parts'])} 部件) —— 不再重新拆图\n")
        keys = self._checked_actions() or [self.action]
        label = ("生成骨骼 + " + " / ".join(ACTION_LABEL[k] for k in keys) +
                 " 动画（合并导出一个 Spine 骨架）" if len(keys) > 1
                 else f"生成骨骼 + {ACTION_LABEL[self.action]} 动画")
        steps = [{"type": "cmd", "label": label, "cmd": self._gen_cmd()}]
        self.mode = "action"
        self._set_busy(True)
        self.worker = Worker(steps)
        self.worker.log.connect(self.logbox.appendPlainText)
        self.worker.step.connect(self.status.setText)
        self.worker.done.connect(self.finish)
        self.worker.start()

    def install_st(self):
        self.logbox.clear()
        self.logbox.appendPlainText(
            "安装 AI 拆图组件: PyTorch(CUDA12.8) + 推理依赖 + 模型, 合计约 12GB。\n"
            "需要 NVIDIA 显卡 (建议 8GB 显存以上), 全程可能十几分钟。")
        steps = [{"type": "cmd", "label": lb, "cmd": cmd, "env": env}
                 for lb, cmd, env in ST.install_steps()]
        self.mode = "install"
        self._set_busy(True)
        self.worker = Worker(steps)
        self.worker.log.connect(self.logbox.appendPlainText)
        self.worker.step.connect(self.status.setText)
        self.worker.done.connect(self.finish)
        self.worker.start()

    def _gen_cmd(self):
        d = dict(PARAMS)
        k_act = self.s_act.value() / 100.0
        k_wind = self.s_wind.value() / 100.0
        k_breath = self.s_breath.value() / 100.0
        k_arm_l = self.s_arm_l.value() / 100.0
        k_arm_r = self.s_arm_r.value() / 100.0
        k_head = self.s_head.value() / 100.0
        for k in ("wind-deg", "ribbon-deg"):
            d[k] = round(d[k] * k_wind, 3)
        # 行走/受击/死亡的发丝由 hair-lag 驱动, 也归到"风吹/发丝"这一档
        d["hair-lag"] = round(d["hair-lag"] * k_wind, 3)
        for k in ("breath", "bob"):
            d[k] = round(d[k] * k_breath, 4)
        for k in ("head-yaw", "head-turn"):
            d[k] = round(d[k] * k_head, 3)
        if not self.cb_blink.isChecked():
            d["blink-depth"] = 0.0
        keys = self._checked_actions() or [self.action]
        multi = len(keys) > 1
        cmd = [sys.executable, os.path.join(HERE, "gen_idle_bones.py"),
               "--parts", os.path.join(self.work, "parts"),
               "--coords", os.path.join(self.work, "coords.json")]
        if multi:
            # 多动作: 每个动作输出到 <work>/<动作>/, Spine 包合并到 <work>/spine/
            cmd += ["--out-dir", self.work]
            dur_map = {k: self._dur_memo.get(k, ACTION_DUR[k]) for k in keys}
            dur_map[self.action] = self.dur.value()      # 当前动作用界面上的时长
            cmd += ["--duration-map",
                    ",".join(f"{k}={dur_map[k]:g}" for k in keys)]
        else:
            cmd += ["--out", os.path.join(self.work, keys[0], "result.json"),
                    "--duration", str(self.dur.value())]
        cmd += ["--action", ",".join(keys),
                "--act-amp", f"{k_act:.2f}",
                "--arm-l", f"{k_arm_l:.2f}",
                "--arm-r", f"{k_arm_r:.2f}",
                "--speed", f"{self.s_speed.value() / 100.0:.2f}",
                "--fps", self.fps.currentText(), "--preview"]
        if self.cb_keep_act.isChecked():
            cmd += ["--db-keep"]
        for k, v in d.items():
            cmd += ["--" + k, str(v)]
        return cmd

    # ------------------------------------------------------------ 收尾
    def finish(self, ok, msg):
        mode, self.mode = self.mode, ""
        self._set_busy(False)
        self.status.setText(msg)
        if not ok:
            if mode == "split":
                self.parts = None
                self._refresh_controls()
                self.preview.setText("拆图失败，看下面日志")
            return
        if mode == "split":
            self._load_split()
            self._refresh_controls()
            if self.parts:
                self._apply_head_wear()      # 拆完立刻套用「头部装饰」
                self.status.setText(
                    f"拆图完成 · {len(self.parts['parts'])} 部件 —— "
                    "确认无误后可以反复生成 待机 / 行走 / 受击 / 死亡")
                self.logbox.appendPlainText(
                    "\n拆图结果: " + os.path.join(self.work, "parts") +
                    "\n体检单:   " + os.path.join(self.work, PREVIEW_PNG))
            else:
                self.status.setText("拆图完成但读取不到坐标，请检查日志")
            return
        if mode == "action":
            keys = self._checked_actions() or [self.action]
            gif = os.path.join(self.work, self.action,
                               f"_{self.action}_"
                               f"{'loop' if ACTION_LOOP[self.action] else 'once'}.gif")
            if os.path.isfile(gif):
                self.preview.show_movie(gif)
            if len(keys) > 1:
                self.logbox.appendPlainText(
                    "\n输出: " + os.path.join(self.work, "<动作>", "result.json") +
                    f"  ({'、'.join(keys)})" +
                    "\nSpine: " + os.path.join(self.work, "spine") +
                    f"  ← 一个骨架 + {len(keys)} 条动作 (png+atlas+json)" +
                    "\n拆图可复用，改勾选直接再点生成即可。")
            else:
                self.logbox.appendPlainText(
                    "\n输出: " + os.path.join(self.work, self.action, "result.json") +
                    "\nSpine: " + os.path.join(self.work, "spine") +
                    "\n拆图可复用，换个动作直接再点生成即可。")

    def open_out(self):
        if self.work and os.path.isdir(self.work):
            os.startfile(self.work)  # noqa: S606


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = Main()
    w.show()
    sys.exit(app.exec())
