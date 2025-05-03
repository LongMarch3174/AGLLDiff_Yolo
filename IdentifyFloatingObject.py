#!/usr/bin/env python3
import sys, os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget, QComboBox,
    QMenuBar, QAction, QToolBar, QFileDialog,
    QLabel, QLineEdit, QPushButton, QProgressBar,
    QHBoxLayout, QVBoxLayout, QGridLayout, QScrollArea, QCheckBox, QMessageBox, QStyledItemDelegate, QListView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QPixmap

from UI_enhance import Enhancer
from UI_detect import Detector
from UI_stats import generate_stats_charts


def resource_path(relative_path: str) -> str:
    """
    PyInstaller 打包后，会把所有资源解压到 sys._MEIPASS 目录，
    否则就在源码目录(__file__的同级)下寻找。
    """
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base, relative_path)


# —— 后台线程：增强 —— #
class EnhanceThread(QThread):
    progress = pyqtSignal(float)
    finished = pyqtSignal(list)

    def __init__(self, opt, weights, inp, out, ensemble):
        super().__init__()
        self.opt = opt
        self.weights = weights
        self.inp = inp
        self.out = out
        self.ensemble = ensemble
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        enh = Enhancer(self.opt, self.weights, self_ensemble=self.ensemble)
        paths = enh.enhance(
            self.inp,
            self.out,
            progress_cb=lambda p: self.progress.emit(p),
            cancel_cb=lambda: self._abort,
        )
        self.finished.emit(paths)


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
        det = Detector(self.use_onnx)
        cb = lambda: self._abort
        out_paths, info = [], None

        if self.mode == "predict":
            res = det.predict_image(self.p["input"], crop=self.p["crop"], count=self.p["count"])
            # normalize save path
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
        counts = self.det.stats_predict(
            self.origin_dir,
            progress_cb=lambda p: self.progress.emit(p),
            cancel_cb=lambda: self._abort
        )
        self.finished.emit(counts)


# —— UI 页面：增强 —— #
class EnhancePage(QWidget):
    def __init__(self):
        super().__init__()
        self.inp  = QLineEdit(); b1=QPushButton("…")
        self.out  = QLineEdit(); b2=QPushButton("…")
        self.opt  = QLineEdit(); b3=QPushButton("…")
        self.wt   = QLineEdit(); b4=QPushButton("…")
        self.cbEn = QCheckBox("自集成")
        for edit,btn,is_dir in [(self.inp,b1,True),(self.out,b2,True),
                                (self.opt,b3,False),(self.wt,b4,False)]:
            btn.clicked.connect(lambda _,e=edit,d=is_dir: self.browse(e,d))

        self.btn_start = QPushButton("开始增强")
        self.btn_abort = QPushButton("中止增强"); self.btn_abort.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_abort.clicked.connect(self.abort)

        self.pb = QProgressBar()
        self.scroll = QScrollArea()
        self.preview = QWidget()
        self.grid = QGridLayout(self.preview)
        self.scroll.setWidget(self.preview)
        self.scroll.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        for lbl,ed,btn in [
            ("输入目录：", self.inp, b1),
            ("输出目录：", self.out, b2),
            ("配置文件：", self.opt, b3),
            ("权重文件：", self.wt, b4),
        ]:
            h = QHBoxLayout(); h.addWidget(QLabel(lbl)); h.addWidget(ed); h.addWidget(btn)
            layout.addLayout(h)
        layout.addWidget(self.cbEn)
        h2 = QHBoxLayout(); h2.addWidget(self.btn_start); h2.addWidget(self.btn_abort)
        layout.addLayout(h2)
        layout.addWidget(self.pb)
        layout.addWidget(QLabel("预览："))
        layout.addWidget(self.scroll)

        self.thread = None

    def browse(self, edit, is_dir):
        if is_dir:
            p = QFileDialog.getExistingDirectory(self, "选择目录", options=QFileDialog.DontUseNativeDialog, directory=os.getcwd())
        else:
            p, _ = QFileDialog.getOpenFileName(self, "选择文件", options=QFileDialog.DontUseNativeDialog, directory=os.getcwd())
        if p:
            edit.setText(p)

    def start(self):
        if not all([self.inp.text(), self.out.text(), self.opt.text(), self.wt.text()]):
            return
        self.thread = EnhanceThread(
            self.opt.text(), self.wt.text(),
            self.inp.text(), self.out.text(),
            self.cbEn.isChecked()
        )
        self.thread.progress.connect(lambda v: self.pb.setValue(int(v * 100)))
        self.thread.finished.connect(self.on_done)
        self.btn_start.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.thread.start()

    def abort(self):
        if self.thread: self.thread.abort()
        self.btn_start.setEnabled(True)
        self.btn_abort.setEnabled(False)

    def on_done(self, paths):
        self.btn_start.setEnabled(True)
        self.btn_abort.setEnabled(False)
        # 清空旧预览
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().deleteLater()
        # 加载前 5 张
        for idx, p in enumerate(paths[:5]):
            lbl = QLabel()
            lbl.setPixmap(QPixmap(p).scaled(100,100,Qt.KeepAspectRatio))
            self.grid.addWidget(lbl, idx//5, idx%5)


# 自定义 Delegate，用于增大行高
class ComboDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        # 强制每一行高度为 32px
        return QSize(s.width(), 32)


# —— UI 页面：识别 —— #
class DetectPage(QWidget):
    def __init__(self):
        super().__init__()

        # 1. 中英文对照列表
        mode_items = [
            ("单张图片预测",    "predict"),
            ("视频/摄像头检测","video"),
            ("FPS 性能测试",   "fps"),
            ("文件夹批量检测","dir_predict"),
            ("热力图生成",     "heatmap"),
            ("导出 ONNX",     "export_onnx"),
            ("ONNX 模型预测","predict_onnx"),
        ]

        # 2. 下拉框：显示中文，userData 存英文代号
        self.mode_cb = QComboBox()
        for text, code in mode_items:
            self.mode_cb.addItem(text, userData=code)

        # —— 在这里直接做美观调整 —— #
        view = QListView()
        self.mode_cb.setView(view)
        # 1) 设置自定义 Delegate，增高每行
        view.setItemDelegate(ComboDelegate(view))
        # 2) 加一点左右 padding
        view.setStyleSheet("""
                    QListView::item {
                        padding: 4px 12px;
                    }
                """)
        # 3) 最多显示 6 行，超过滚动
        self.mode_cb.setMaxVisibleItems(6)
        # 4) 让下拉框宽度自适应最长内容
        self.mode_cb.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.mode_cb.setMinimumContentsLength(8)
        # —— 美观调整结束 —— #

        # 3. 准备各英文模式对应的页面容器
        self.pages = {}
        self.stack = QStackedWidget()
        for _, code in mode_items:
            w = QWidget()
            g = QGridLayout(w)
            self.pages[code] = (w, g)
            self.stack.addWidget(w)

        # —— 接下来按各模式填充参数区，和之前完全一样 —— #
        # predict
        w, g = self.pages["predict"]
        self.ip = QLineEdit(); b1 = QPushButton("…")
        b1.clicked.connect(lambda: self.browse(self.ip, False))
        self.sp = QLineEdit(); b2 = QPushButton("…")
        b2.clicked.connect(lambda: self.browse(self.sp, True))
        self.cb_crop = QCheckBox("crop"); self.cb_cnt = QCheckBox("count")
        g.addWidget(QLabel("输入图像："), 0, 0); g.addWidget(self.ip, 0, 1); g.addWidget(b1, 0, 2)
        g.addWidget(QLabel("保存目录："), 1, 0); g.addWidget(self.sp, 1, 1); g.addWidget(b2, 1, 2)
        g.addWidget(self.cb_crop, 2, 0);          g.addWidget(self.cb_cnt, 2, 1)

        # video
        w, g = self.pages["video"]
        self.vp = QLineEdit(); b3 = QPushButton("…")
        b3.clicked.connect(lambda: self.browse(self.vp, False))
        self.vs = QLineEdit(); b4 = QPushButton("…")
        b4.clicked.connect(lambda: self.browse(self.vs, True))
        self.vfps = QLineEdit("25")
        g.addWidget(QLabel("视频路径："), 0, 0); g.addWidget(self.vp, 0, 1); g.addWidget(b3, 0, 2)
        g.addWidget(QLabel("保存路径："), 1, 0); g.addWidget(self.vs, 1, 1); g.addWidget(b4, 1, 2)
        g.addWidget(QLabel("保存FPS："), 2, 0); g.addWidget(self.vfps, 2, 1)

        # fps
        w, g = self.pages["fps"]
        self.fp = QLineEdit(); b5 = QPushButton("…")
        b5.clicked.connect(lambda: self.browse(self.fp, False))
        self.fti = QLineEdit("100")
        g.addWidget(QLabel("测试图像："), 0, 0); g.addWidget(self.fp, 0, 1); g.addWidget(b5, 0, 2)
        g.addWidget(QLabel("次数："), 1, 0); g.addWidget(self.fti, 1, 1)

        # dir_predict
        w, g = self.pages["dir_predict"]
        self.dp = QLineEdit(); b6 = QPushButton("…")
        b6.clicked.connect(lambda: self.browse(self.dp, True))
        self.ds = QLineEdit(); b7 = QPushButton("…")
        b7.clicked.connect(lambda: self.browse(self.ds, True))
        g.addWidget(QLabel("输入目录："), 0, 0); g.addWidget(self.dp, 0, 1); g.addWidget(b6, 0, 2)
        g.addWidget(QLabel("保存目录："), 1, 0); g.addWidget(self.ds, 1, 1); g.addWidget(b7, 1, 2)

        # heatmap
        w, g = self.pages["heatmap"]
        self.hp = QLineEdit(); b8 = QPushButton("…")
        b8.clicked.connect(lambda: self.browse(self.hp, False))
        self.hs = QLineEdit(); b9 = QPushButton("…")
        b9.clicked.connect(lambda: self.browse(self.hs, True))
        g.addWidget(QLabel("热力图源："), 0, 0); g.addWidget(self.hp, 0, 1); g.addWidget(b8, 0, 2)
        g.addWidget(QLabel("保存目录："), 1, 0); g.addWidget(self.hs, 1, 1); g.addWidget(b9, 1, 2)

        # export_onnx
        w, g = self.pages["export_onnx"]
        self.cb_simp = QCheckBox("Simplify")
        self.op      = QLineEdit(); b10 = QPushButton("…")
        b10.clicked.connect(lambda: self.browse(self.op, True))
        g.addWidget(self.cb_simp, 0, 0)
        g.addWidget(QLabel("ONNX 路径："), 1, 0); g.addWidget(self.op, 1, 1); g.addWidget(b10, 1, 2)

        # predict_onnx
        w, g = self.pages["predict_onnx"]
        self.ipx = QLineEdit(); b11 = QPushButton("…")
        b11.clicked.connect(lambda: self.browse(self.ipx, False))
        self.sox = QLineEdit(); b12 = QPushButton("…")
        b12.clicked.connect(lambda: self.browse(self.sox, True))
        g.addWidget(QLabel("输入图像："), 0, 0); g.addWidget(self.ipx, 0, 1); g.addWidget(b11, 0, 2)
        g.addWidget(QLabel("保存目录："), 1, 0); g.addWidget(self.sox, 1, 1); g.addWidget(b12, 1, 2)

        # 4. 切换页面
        self.mode_cb.currentIndexChanged.connect(self._on_mode_change)

        # 启动/中止按钮、进度、预览…
        self.btn_start = QPushButton("开始识别")
        self.btn_abort = QPushButton("中止识别")
        self.btn_abort.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_abort.clicked.connect(self.abort)

        self.pb = QProgressBar()
        self.scroll = QScrollArea()
        self.pre = QWidget()
        self.grid = QGridLayout(self.pre)
        self.scroll.setWidget(self.pre)
        self.scroll.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("模式："))
        layout.addWidget(self.mode_cb)
        layout.addWidget(self.stack)
        hb = QHBoxLayout(); hb.addWidget(self.btn_start); hb.addWidget(self.btn_abort)
        layout.addLayout(hb)
        layout.addWidget(self.pb)
        layout.addWidget(QLabel("预览："))
        layout.addWidget(self.scroll)

        self.thread = None

        # 5. 默认激活第一项
        self.mode_cb.setCurrentIndex(0)
        self._on_mode_change(0)

    def _on_mode_change(self, idx):
        code = self.mode_cb.itemData(idx)
        page_widget, _ = self.pages[code]
        self.stack.setCurrentWidget(page_widget)

    def browse(self, edit, is_dir):
        if is_dir:
            p = QFileDialog.getExistingDirectory(self, "选择目录", options=QFileDialog.DontUseNativeDialog, directory=os.getcwd())
        else:
            p, _ = QFileDialog.getOpenFileName(self, "选择文件", options=QFileDialog.DontUseNativeDialog, directory=os.getcwd())
        if p:
            edit.setText(p)

    def start(self):
        # 拿到英文模式代号
        mode = self.mode_cb.currentData()

        # 根据模式收集参数
        params = {}
        if mode == "predict":
            params = {
                "input": self.ip.text(),
                "save":  self.sp.text(),
                "crop":  self.cb_crop.isChecked(),
                "count": self.cb_cnt.isChecked(),
            }
        elif mode == "video":
            params = {
                "video_path": self.vp.text(),
                "video_save": self.vs.text(),
                "video_fps":  self.vfps.text(),
            }
        elif mode == "fps":
            params = {
                "fps_image":     self.fp.text(),
                "test_interval": self.fti.text(),
            }
        elif mode == "dir_predict":
            params = {
                "origin":   self.dp.text(),
                "save_dir": self.ds.text(),
            }
        elif mode == "heatmap":
            params = {
                "heat_image": self.hp.text(),
                "heat_save":  self.hs.text(),
            }
        elif mode == "export_onnx":
            params = {
                "simplify":  self.cb_simp.isChecked(),
                "onnx_path": self.op.text(),
            }
        elif mode == "predict_onnx":
            params = {
                "input_onnx": self.ipx.text(),
                "save_onnx":  self.sox.text(),
            }
        else:
            return

        # 简单校验一下
        if any(v in (None, "") for v in params.values()):
            QMessageBox.warning(self, "参数不全", "请填写完整参数后再开始。")
            return

        # 创建并启动后台线程
        self.thread = DetectThread(False, mode, params)
        self.thread.progress.connect(lambda v: self.pb.setValue(int(v * 100)))
        self.thread.finished.connect(self.on_done)
        self.btn_start.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.thread.start()

    def abort(self):
        if self.thread:
            self.thread.abort()
        self.btn_start.setEnabled(True)
        self.btn_abort.setEnabled(False)

    def on_done(self, paths, info):
        # … 预览更新逻辑 …
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
        self.scroll = QScrollArea()
        self.pre = QWidget()
        self.grid = QGridLayout(self.pre)
        self.scroll.setWidget(self.pre)
        self.scroll.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        for lbl, ed, btn in [
            ("输入目录：", self.origin, b1),
            ("保存目录：", self.save_dir, b2),
        ]:
            h = QHBoxLayout(); h.addWidget(QLabel(lbl)); h.addWidget(ed); h.addWidget(btn)
            layout.addLayout(h)
        h2 = QHBoxLayout(); h2.addWidget(self.btn_start); h2.addWidget(self.btn_abort)
        layout.addLayout(h2)
        layout.addWidget(self.pb)
        layout.addWidget(QLabel("统计图预览："))
        layout.addWidget(self.scroll)

        self.thread = None

    def browse(self, edit, is_dir):
        p = QFileDialog.getExistingDirectory(self, options=QFileDialog.DontUseNativeDialog, directory=os.getcwd()) \
            if is_dir else QFileDialog.getSaveFileName(self, options=QFileDialog.DontUseNativeDialog, directory=os.getcwd())[0]
        if p:
            edit.setText(p)

    def start(self):
        origin = self.origin.text()
        save_dir = self.save_dir.text()
        if not origin or not save_dir:
            return
        det = Detector(use_onnx=False)
        self.thread = StatsThread(det, origin)
        self.thread.progress.connect(lambda v: self.pb.setValue(int(v * 100)))
        self.thread.finished.connect(lambda counts: self.on_done(counts, save_dir))
        self.btn_start.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.thread.start()

    def abort(self):
        if self.thread:
            self.thread.abort()
        self.btn_start.setEnabled(True)
        self.btn_abort.setEnabled(False)

    def on_done(self, counts, save_dir):
        self.btn_start.setEnabled(True)
        self.btn_abort.setEnabled(False)
        pie, bar, line = generate_stats_charts(counts, save_dir)
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

        self.setWindowIcon(QIcon(resource_path("UI/ui.png")))

        # 菜单栏
        mb = self.menuBar()
        for title in ["文件", "图像浏览", "数据分析"]:
            mb.addMenu(title)

        # 工具栏
        tb = QToolBar()
        for icon, tip, page in [
            (resource_path("UI/exit_system.png"), "退出", None),
            (resource_path("UI/enhance.png"), "增强", "enhance"),
            (resource_path("UI/detect.png"), "识别", "detect"),
            (resource_path("UI/stats.png"), "统计", "stats")
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

        # 中央页面
        self.enh_page = EnhancePage()
        self.det_page = DetectPage()
        self.stats_page = StatsPage()
        self.central_widget = QStackedWidget()
        self.central_widget.addWidget(self.enh_page)
        self.central_widget.addWidget(self.det_page)
        self.central_widget.addWidget(self.stats_page)
        self.setCentralWidget(self.central_widget)

        # 默认显示增强页
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
