#!/usr/bin/env python3
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget, QComboBox,
    QMenuBar, QAction, QToolBar, QFileDialog,
    QLabel, QLineEdit, QPushButton, QProgressBar,
    QHBoxLayout, QVBoxLayout, QGridLayout, QScrollArea, QCheckBox,
    QMessageBox, QStyledItemDelegate, QListView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QPixmap

from UI_enhance import Enhancer
from UI_detect import Detector
from UI_stats import generate_stats_charts


# —— 后台线程：增强 —— #
class EnhanceThread(QThread):
    progress = pyqtSignal(float)
    finished = pyqtSignal(list)

    def __init__(self, inp, out):
        super().__init__()
        self.inp = inp
        self.out = out
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            enh = Enhancer()
            paths = enh.enhance(
                self.inp,
                self.out,
                progress_cb=lambda p: self.progress.emit(p),
                cancel_cb=lambda: self._abort,
            )
            self.finished.emit(paths)
        except FileNotFoundError as e:
            # 将错误消息通过信号传回主线程
            QMessageBox.critical(None, "文件错误", str(e))


# —— 后台线程：识别 —— #
class DetectThread(QThread):
    progress = pyqtSignal(float)
    finished = pyqtSignal(list, object)  # out_paths, info

    def __init__(self, use_onnx, mode, params):
        super().__init__()
        self.use_onnx = use_onnx
        self.mode = mode
        self.p = params
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            det = Detector(self.use_onnx)
            cb = lambda: self._abort
            out_paths, info = [], None

            if self.mode == "predict":
                res = det.predict_image(self.p["input"], crop=self.p["crop"], count=self.p["count"])
                target = self.p["save"]
                if os.path.isdir(target):
                    name = os.path.splitext(os.path.basename(self.p["input"]))[0] + ".png"
                    save_path = os.path.join(target, name)
                else:
                    root, ext = os.path.splitext(target)
                    save_path = root + (ext if ext else ".png")
                res.save(save_path, format="PNG")
                out_paths = [save_path]

            elif self.mode == "video":
                det.video_predict(
                    self.p["video_path"],
                    self.p["video_save"],
                    fps=float(self.p["video_fps"]),
                    progress_cb=lambda v: self.progress.emit(v),
                    cancel_cb=cb,
                )
                out_paths = [self.p["video_save"]]

            elif self.mode == "fps":
                t = det.fps_test(self.p["fps_image"], int(self.p["test_interval"]))
                info = f"Avg time: {t:.4f}s, FPS: {1/t:.2f}"

            elif self.mode == "dir_predict":
                out_paths = det.dir_predict(
                    self.p["origin"],
                    self.p["save_dir"],
                    progress_cb=lambda v: self.progress.emit(v),
                    cancel_cb=cb,
                )

            elif self.mode == "heatmap":
                img_path = self.p["heat_image"]
                target = self.p["heat_save"]
                if os.path.isdir(target):
                    base = os.path.splitext(os.path.basename(img_path))[0] + "_heatmap.png"
                    save_path = os.path.join(target, base)
                else:
                    root, ext = os.path.splitext(target)
                    save_path = root + (ext if ext else ".png")
                det.detect_heatmap(img_path, save_path)
                out_paths = [save_path]

            elif self.mode == "export_onnx":
                det.export_onnx(self.p["simplify"], self.p["onnx_path"])
                info = f"Exported to {self.p['onnx_path']}"

            elif self.mode == "predict_onnx":
                res = det.predict_image(self.p["input_onnx"])
                target = self.p["save_onnx"]
                if os.path.isdir(target):
                    name = os.path.splitext(os.path.basename(self.p["input_onnx"]))[0] + ".png"
                    save_path = os.path.join(target, name)
                else:
                    root, ext = os.path.splitext(target)
                    save_path = root + (ext if ext else ".png")
                res.save(save_path, format="PNG")
                out_paths = [save_path]

            self.finished.emit(out_paths, info)

        except FileNotFoundError as e:
            QMessageBox.critical(None, "文件错误", str(e))


# —— 后台线程：统计 —— #
class StatsThread(QThread):
    progress = pyqtSignal(float)
    finished = pyqtSignal(dict)

    def __init__(self, detector, origin_dir):
        super().__init__()
        self.det = detector
        self.origin_dir = origin_dir
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            counts = self.det.stats_predict(
                self.origin_dir,
                progress_cb=lambda p: self.progress.emit(p),
                cancel_cb=lambda: self._abort
            )
            self.finished.emit(counts)
        except FileNotFoundError as e:
            QMessageBox.critical(None, "文件错误", str(e))


# —— UI 页面：增强 —— #
class EnhancePage(QWidget):
    def __init__(self):
        super().__init__()
        # 只保留 输入/输出 目录 和 自集成 选项
        self.inp = QLineEdit(); b1 = QPushButton("…")
        self.out = QLineEdit(); b2 = QPushButton("…")

        b1.clicked.connect(lambda: self.browse(self.inp, True))
        b2.clicked.connect(lambda: self.browse(self.out, True))

        self.btn_start = QPushButton("开始增强")
        self.btn_abort = QPushButton("中止增强")
        self.btn_abort.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_abort.clicked.connect(self.abort)

        self.pb = QProgressBar()
        self.scroll = QScrollArea()
        self.preview = QWidget()
        self.grid = QGridLayout(self.preview)
        self.scroll.setWidget(self.preview)
        self.scroll.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        # 仅两行：输入目录 + 输出目录
        for lbl, ed, btn in [
            ("输入目录：", self.inp, b1),
            ("输出目录：", self.out, b2),
        ]:
            h = QHBoxLayout()
            h.addWidget(QLabel(lbl))
            h.addWidget(ed)
            h.addWidget(btn)
            layout.addLayout(h)

        h2 = QHBoxLayout()
        h2.addWidget(self.btn_start)
        h2.addWidget(self.btn_abort)
        layout.addLayout(h2)
        layout.addWidget(self.pb)
        layout.addWidget(QLabel("预览："))
        layout.addWidget(self.scroll)

        self.thread = None

    def browse(self, edit: QLineEdit, is_dir: bool):
        if is_dir:
            path = QFileDialog.getExistingDirectory(
                self, "选择目录", options=QFileDialog.DontUseNativeDialog, directory=os.getcwd()
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择文件", options=QFileDialog.DontUseNativeDialog, directory=os.getcwd()
            )
        if path:
            edit.setText(path)

    def start(self):
        inp = self.inp.text().strip()
        out = self.out.text().strip()
        if not inp or not out:
            QMessageBox.warning(self, "参数不全", "请填写输入目录和输出目录。")
            return

        try:
            self.thread = EnhanceThread(inp, out)
            self.thread.progress.connect(lambda v: self.pb.setValue(int(v * 100)))
            self.thread.finished.connect(self.on_done)
            self.btn_start.setEnabled(False)
            self.btn_abort.setEnabled(True)
            self.thread.start()
        except FileNotFoundError as e:
            QMessageBox.critical(self, "文件错误", str(e))
            self.btn_start.setEnabled(True)
            self.btn_abort.setEnabled(False)

    def abort(self):
        if self.thread:
            self.thread.abort()
        self.btn_start.setEnabled(True)
        self.btn_abort.setEnabled(False)

    def on_done(self, paths: list):
        self.btn_start.setEnabled(True)
        self.btn_abort.setEnabled(False)
        # 清空旧预览
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().deleteLater()
        # 显示前 5 张
        for idx, p in enumerate(paths[:5]):
            lbl = QLabel()
            lbl.setPixmap(QPixmap(p).scaled(100, 100, Qt.KeepAspectRatio))
            self.grid.addWidget(lbl, idx // 5, idx % 5)


# 自定义 Delegate，用于增大行高
class ComboDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        return QSize(s.width(), 32)


# —— UI 页面：识别 —— #
class DetectPage(QWidget):
    def __init__(self):
        super().__init__()
        mode_items = [
            ("单张图片预测",    "predict"),
            ("视频/摄像头检测","video"),
            ("FPS 性能测试",   "fps"),
            ("文件夹批量检测","dir_predict"),
            ("热力图生成",     "heatmap"),
            ("导出 ONNX",     "export_onnx"),
            ("ONNX 模型预测","predict_onnx"),
        ]
        self.mode_cb = QComboBox()
        for text, code in mode_items:
            self.mode_cb.addItem(text, userData=code)

        view = QListView()
        self.mode_cb.setView(view)
        view.setItemDelegate(ComboDelegate(view))
        view.setStyleSheet("QListView::item { padding: 4px 12px; }")
        self.mode_cb.setMaxVisibleItems(6)
        self.mode_cb.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.mode_cb.setMinimumContentsLength(8)

        self.pages = {}
        self.stack = QStackedWidget()
        for _, code in mode_items:
            w = QWidget(); g = QGridLayout(w)
            self.pages[code] = (w, g)
            self.stack.addWidget(w)

            # 为每种模式创建一个页面和布局
            self.pages = {}
            self.stack = QStackedWidget()
            for _, code in mode_items:
                w = QWidget()
                g = QGridLayout(w)
                self.pages[code] = (w, g)
                self.stack.addWidget(w)

            # ———— 填充 “单张图片预测” 参数区 ———— #
            w, g = self.pages["predict"]
            self.ip = QLineEdit();
            b1 = QPushButton("…")
            b1.clicked.connect(lambda: self.browse(self.ip, False))
            self.sp = QLineEdit();
            b2 = QPushButton("…")
            b2.clicked.connect(lambda: self.browse(self.sp, True))
            self.cb_crop = QCheckBox("crop");
            self.cb_cnt = QCheckBox("count")
            g.addWidget(QLabel("输入图像："), 0, 0);
            g.addWidget(self.ip, 0, 1);
            g.addWidget(b1, 0, 2)
            g.addWidget(QLabel("保存目录："), 1, 0);
            g.addWidget(self.sp, 1, 1);
            g.addWidget(b2, 1, 2)
            g.addWidget(self.cb_crop, 2, 0);
            g.addWidget(self.cb_cnt, 2, 1)

            # ———— 填充 “视频/摄像头检测” 参数区 ———— #
            w, g = self.pages["video"]
            self.vp = QLineEdit();
            b3 = QPushButton("…")
            b3.clicked.connect(lambda: self.browse(self.vp, False))
            self.vs = QLineEdit();
            b4 = QPushButton("…")
            b4.clicked.connect(lambda: self.browse(self.vs, True))
            self.vfps = QLineEdit("25")
            g.addWidget(QLabel("视频路径："), 0, 0);
            g.addWidget(self.vp, 0, 1);
            g.addWidget(b3, 0, 2)
            g.addWidget(QLabel("保存路径："), 1, 0);
            g.addWidget(self.vs, 1, 1);
            g.addWidget(b4, 1, 2)
            g.addWidget(QLabel("保存FPS："), 2, 0);
            g.addWidget(self.vfps, 2, 1)

            # ———— 填充 “FPS 性能测试” 参数区 ———— #
            w, g = self.pages["fps"]
            self.fp = QLineEdit();
            b5 = QPushButton("…")
            b5.clicked.connect(lambda: self.browse(self.fp, False))
            self.fti = QLineEdit("100")
            g.addWidget(QLabel("测试图像："), 0, 0);
            g.addWidget(self.fp, 0, 1);
            g.addWidget(b5, 0, 2)
            g.addWidget(QLabel("次数："), 1, 0);
            g.addWidget(self.fti, 1, 1)

            # ———— 填充 “文件夹批量检测” 参数区 ———— #
            w, g = self.pages["dir_predict"]
            self.dp = QLineEdit();
            b6 = QPushButton("…")
            b6.clicked.connect(lambda: self.browse(self.dp, True))
            self.ds = QLineEdit();
            b7 = QPushButton("…")
            b7.clicked.connect(lambda: self.browse(self.ds, True))
            g.addWidget(QLabel("输入目录："), 0, 0);
            g.addWidget(self.dp, 0, 1);
            g.addWidget(b6, 0, 2)
            g.addWidget(QLabel("保存目录："), 1, 0);
            g.addWidget(self.ds, 1, 1);
            g.addWidget(b7, 1, 2)

            # ———— 填充 “热力图生成” 参数区 ———— #
            w, g = self.pages["heatmap"]
            self.hp = QLineEdit();
            b8 = QPushButton("…")
            b8.clicked.connect(lambda: self.browse(self.hp, False))
            self.hs = QLineEdit();
            b9 = QPushButton("…")
            b9.clicked.connect(lambda: self.browse(self.hs, True))
            g.addWidget(QLabel("热力图源："), 0, 0);
            g.addWidget(self.hp, 0, 1);
            g.addWidget(b8, 0, 2)
            g.addWidget(QLabel("保存目录："), 1, 0);
            g.addWidget(self.hs, 1, 1);
            g.addWidget(b9, 1, 2)

            # ———— 填充 “导出 ONNX” 参数区 ———— #
            w, g = self.pages["export_onnx"]
            self.cb_simp = QCheckBox("Simplify")
            self.op = QLineEdit();
            b10 = QPushButton("…")
            b10.clicked.connect(lambda: self.browse(self.op, True))
            g.addWidget(self.cb_simp, 0, 0)
            g.addWidget(QLabel("ONNX 路径："), 1, 0);
            g.addWidget(self.op, 1, 1);
            g.addWidget(b10, 1, 2)

            # ———— 填充 “ONNX 模型预测” 参数区 ———— #
            w, g = self.pages["predict_onnx"]
            self.ipx = QLineEdit();
            b11 = QPushButton("…")
            b11.clicked.connect(lambda: self.browse(self.ipx, False))
            self.sox = QLineEdit();
            b12 = QPushButton("…")
            b12.clicked.connect(lambda: self.browse(self.sox, True))
            g.addWidget(QLabel("输入图像："), 0, 0);
            g.addWidget(self.ipx, 0, 1);
            g.addWidget(b11, 0, 2)
            g.addWidget(QLabel("保存目录："), 1, 0);
            g.addWidget(self.sox, 1, 1);
            g.addWidget(b12, 1, 2)

            # 切换页面
            self.mode_cb.currentIndexChanged.connect(self._on_mode_change)
            self.mode_cb.setCurrentIndex(0)

        self.btn_start = QPushButton("开始识别")
        self.btn_abort = QPushButton("中止识别"); self.btn_abort.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_abort.clicked.connect(self.abort)

        self.pb = QProgressBar()
        self.scroll = QScrollArea(); self.pre = QWidget()
        self.grid = QGridLayout(self.pre)
        self.scroll.setWidget(self.pre); self.scroll.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("模式：")); layout.addWidget(self.mode_cb)
        layout.addWidget(self.stack)
        hb = QHBoxLayout(); hb.addWidget(self.btn_start); hb.addWidget(self.btn_abort)
        layout.addLayout(hb); layout.addWidget(self.pb)
        layout.addWidget(QLabel("预览：")); layout.addWidget(self.scroll)

        self.thread = None
        self.mode_cb.setCurrentIndex(0)
        self.mode_cb.currentIndexChanged.connect(self._on_mode_change)
        self._on_mode_change(0)

    def _on_mode_change(self, idx):
        code = self.mode_cb.itemData(idx)
        page_widget, _ = self.pages[code]
        self.stack.setCurrentWidget(page_widget)

    def browse(self, edit, is_dir):
        if is_dir:
            p = QFileDialog.getExistingDirectory(self, "选择目录",
                options=QFileDialog.DontUseNativeDialog, directory=os.getcwd())
        else:
            p, _ = QFileDialog.getOpenFileName(self, "选择文件",
                options=QFileDialog.DontUseNativeDialog, directory=os.getcwd())
        if p:
            edit.setText(p)

    def start(self):
        mode = self.mode_cb.currentData()
        params = {}
        # —— 根据模式收集参数 —— #
        if mode == "predict":
            params = {
                "input": self.ip.text(),
                "save": self.sp.text(),
                "crop": self.cb_crop.isChecked(),
                "count": self.cb_cnt.isChecked(),
            }
        elif mode == "video":
            params = {
                "video_path": self.vp.text(),
                "video_save": self.vs.text(),
                "video_fps": self.vfps.text(),
            }
        elif mode == "fps":
            params = {
                "fps_image": self.fp.text(),
                "test_interval": self.fti.text(),
            }
        elif mode == "dir_predict":
            params = {
                "origin": self.dp.text(),
                "save_dir": self.ds.text(),
            }
        elif mode == "heatmap":
            params = {
                "heat_image": self.hp.text(),
                "heat_save": self.hs.text(),
            }
        elif mode == "export_onnx":
            params = {
                "simplify": self.cb_simp.isChecked(),
                "onnx_path": self.op.text(),
            }
        elif mode == "predict_onnx":
            params = {
                "input_onnx": self.ipx.text(),
                "save_onnx": self.sox.text(),
            }
        else:
            return

        # 参数完整性校验
        if any(v in (None, "") for v in params.values()):
            QMessageBox.warning(self, "参数不全", "请填写完整参数后再开始。")
            return

        # 文件/目录预检查
        try:
            if mode == "predict":
                if not os.path.exists(params["input"]):
                    raise FileNotFoundError(f"输入图像不存在：{params['input']}")
                save = params["save"]
                if not os.path.isdir(save) and not os.path.exists(os.path.dirname(save)):
                    raise FileNotFoundError(f"保存路径不可用：{save}")

            elif mode == "video":
                if not os.path.exists(params["video_path"]):
                    raise FileNotFoundError(f"视频文件不存在：{params['video_path']}")

            elif mode == "fps":
                if not os.path.exists(params["fps_image"]):
                    raise FileNotFoundError(f"测试图像不存在：{params['fps_image']}")

            elif mode == "dir_predict":
                if not os.path.exists(params["origin"]):
                    raise FileNotFoundError(f"输入目录不存在：{params['origin']}")

            elif mode == "heatmap":
                if not os.path.exists(params["heat_image"]):
                    raise FileNotFoundError(f"热力图源文件不存在：{params['heat_image']}")

            elif mode in ("export_onnx",):
                # 导出 ONNX 时，父目录应存在
                out_dir = os.path.dirname(params["onnx_path"]) or "."
                if not os.path.exists(out_dir):
                    raise FileNotFoundError(f"ONNX 保存目录不存在：{out_dir}")

            elif mode == "predict_onnx":
                if not os.path.exists(params["input_onnx"]):
                    raise FileNotFoundError(f"输入 ONNX 图像不存在：{params['input_onnx']}")
                save = params["save_onnx"]
                if not os.path.isdir(save) and not os.path.exists(os.path.dirname(save)):
                    raise FileNotFoundError(f"ONNX 预测保存路径不可用：{save}")

            # 启动线程
            self.thread = DetectThread(False, mode, params)
            self.thread.progress.connect(lambda v: self.pb.setValue(int(v * 100)))
            self.thread.finished.connect(self.on_done)
            self.btn_start.setEnabled(False)
            self.btn_abort.setEnabled(True)
            self.thread.start()

        except FileNotFoundError as e:
            QMessageBox.critical(self, "文件错误", str(e))

    def abort(self):
        if self.thread:
            self.thread.abort()
        self.btn_start.setEnabled(True)
        self.btn_abort.setEnabled(False)

    def on_done(self, paths, info):
        self.btn_start.setEnabled(True)
        self.btn_abort.setEnabled(False)

        # 更新预览
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().deleteLater()
        for idx, p in enumerate(paths):
            lbl = QLabel()
            lbl.setPixmap(QPixmap(p).scaled(100,100,Qt.KeepAspectRatio))
            self.grid.addWidget(lbl, idx//5, idx%5)

        if info:
            QMessageBox.information(self, "结果信息", str(info))


# —— UI 页面：统计 —— #
class StatsPage(QWidget):
    def __init__(self):
        super().__init__()
        self.origin = QLineEdit(); b1 = QPushButton("…")
        self.save_dir = QLineEdit(); b2 = QPushButton("…")
        b1.clicked.connect(lambda: self.browse(self.origin, True))
        b2.clicked.connect(lambda: self.browse(self.save_dir, True))

        self.btn_start = QPushButton("开始统计")
        self.btn_abort = QPushButton("中止统计"); self.btn_abort.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_abort.clicked.connect(self.abort)

        self.pb = QProgressBar()
        self.scroll = QScrollArea(); self.pre = QWidget()
        self.grid = QGridLayout(self.pre)
        self.scroll.setWidget(self.pre); self.scroll.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        for lbl, ed, btn in [
            ("输入目录：", self.origin, b1),
            ("保存目录：", self.save_dir, b2),
        ]:
            h = QHBoxLayout(); h.addWidget(QLabel(lbl)); h.addWidget(ed); h.addWidget(btn)
            layout.addLayout(h)
        h2 = QHBoxLayout(); h2.addWidget(self.btn_start); h2.addWidget(self.btn_abort)
        layout.addLayout(h2); layout.addWidget(self.pb)
        layout.addWidget(QLabel("统计图预览：")); layout.addWidget(self.scroll)

        self.thread = None

    def browse(self, edit, is_dir):
        p = QFileDialog.getExistingDirectory(self,
            options=QFileDialog.DontUseNativeDialog, directory=os.getcwd()) if is_dir else \
            QFileDialog.getSaveFileName(self,
            options=QFileDialog.DontUseNativeDialog, directory=os.getcwd())[0]
        if p:
            edit.setText(p)

    def start(self):
        origin = self.origin.text()
        save_dir = self.save_dir.text()
        if not origin or not save_dir:
            QMessageBox.warning(self, "参数不全", "请填写输入目录和保存目录。")
            return
        if not os.path.exists(origin):
            QMessageBox.critical(self, "文件错误", f"输入目录不存在：{origin}")
            return

        try:
            det = Detector(use_onnx=False)
            self.thread = StatsThread(det, origin)
            self.thread.progress.connect(lambda v: self.pb.setValue(int(v * 100)))
            self.thread.finished.connect(lambda counts: self.on_done(counts, save_dir))
            self.btn_start.setEnabled(False)
            self.btn_abort.setEnabled(True)
            self.thread.start()

        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def abort(self):
        if self.thread:
            self.thread.abort()
        self.btn_start.setEnabled(True)
        self.btn_abort.setEnabled(False)

    def on_done(self, counts, save_dir):
        self.btn_start.setEnabled(True)
        self.btn_abort.setEnabled(False)

        try:
            pie, bar, line = generate_stats_charts(counts, save_dir)
        except Exception as e:
            QMessageBox.critical(self, "统计图生成错误", str(e))
            return

        # 清空旧预览
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().deleteLater()
        for idx, p in enumerate([pie, bar, line]):
            lbl = QLabel()
            lbl.setPixmap(QPixmap(p).scaled(300, 300, Qt.KeepAspectRatio))
            self.grid.addWidget(lbl, idx // 3, idx % 3)


# —— 主窗口 —— #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水面漂浮物系统")
        self.resize(1024, 768)
        self.setWindowIcon(QIcon("UI/ui.png"))

        mb = self.menuBar()
        for title in ["文件", "图像浏览", "数据分析"]:
            mb.addMenu(title)

        tb = QToolBar()
        for icon, tip, page in [
            ("UI/exit_system.png", "退出", None),
            ("UI/enhance.png", "增强", "enhance"),
            ("UI/detect.png", "识别", "detect"),
            ("UI/stats.png", "统计", "stats")
        ]:
            act = QAction(QIcon(icon), tip, self)
            if tip == "退出":
                act.triggered.connect(self.close)
            tb.addAction(act)
            if page == "enhance":
                act.triggered.connect(lambda _, p="enhance": self.switch(p))
            if page == "detect":
                act.triggered.connect(lambda _, p="detect": self.switch(p))
            if page == "stats":
                act.triggered.connect(lambda _, p="stats": self.switch(p))
        self.addToolBar(tb)

        self.enh_page = EnhancePage()
        self.det_page = DetectPage()
        self.stats_page = StatsPage()
        self.central_widget = QStackedWidget()
        self.central_widget.addWidget(self.enh_page)
        self.central_widget.addWidget(self.det_page)
        self.central_widget.addWidget(self.stats_page)
        self.setCentralWidget(self.central_widget)
        self.switch("enhance")

    def switch(self, name):
        if name == "enhance":
            self.central_widget.setCurrentWidget(self.enh_page)
        elif name == "detect":
            self.central_widget.setCurrentWidget(self.det_page)
        elif name == "stats":
            self.central_widget.setCurrentWidget(self.stats_page)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
