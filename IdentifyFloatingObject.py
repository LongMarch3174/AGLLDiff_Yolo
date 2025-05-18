#!/usr/bin/env python3
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget, QAction, QToolBar,
    QFileDialog, QLabel, QLineEdit, QPushButton, QProgressBar,
    QHBoxLayout, QVBoxLayout, QGridLayout, QScrollArea, QCheckBox,
    QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.uic import loadUi

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
        enh = Enhancer()
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
    finished = pyqtSignal(list, object)  # list of output paths, extra info

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
            result = det.predict_image(self.p["input"], crop=self.p["crop"], count=self.p["count"])
            if isinstance(result, tuple):
                img, info = result
            else:
                img = result

            target = self.p["save"]
            if os.path.isdir(target):
                name = os.path.splitext(os.path.basename(self.p["input"]))[0] + ".png"
                save_path = os.path.join(target, name)
            else:
                root, ext = os.path.splitext(target)
                save_path = root + (ext or ".png")
            img.save(save_path)
            out_paths = [save_path]

        elif self.mode == "video":
            det.video_predict(
                self.p["video_path"],
                self.p["video_save"],
                fps=float(self.p["video_fps"]),
                progress_cb=lambda v: self.progress.emit(v),
                cancel_cb=cb
            )
            out_paths = [self.p["video_save"]]

        elif self.mode == "fps":
            t = det.fps_test(self.p["fps_image"], int(self.p["test_interval"]))
            info = f"Avg time: {t:.4f}s, FPS: {1 / t:.2f}"

        elif self.mode == "dir_predict":
            out_paths = det.dir_predict(
                self.p["origin"],
                self.p["save_dir"],
                progress_cb=lambda v: self.progress.emit(v),
                cancel_cb=cb
            )

        elif self.mode == "heatmap":
            hm = det.detect_heatmap(self.p["heat_image"], self.p["heat_save"])
            out_paths = [self.p["heat_save"]]

        elif self.mode == "export_onnx":
            det.export_onnx(self.p["simplify"], self.p["onnx_path"])
            info = f"Exported ONNX to {self.p['onnx_path']}"

        elif self.mode == "predict_onnx":
            img = det.predict_image(self.p["input_onnx"])
            target = self.p["save_onnx"]
            if os.path.isdir(target):
                name = os.path.splitext(os.path.basename(self.p["input_onnx"]))[0] + ".png"
                save_path = os.path.join(target, name)
            else:
                root, ext = os.path.splitext(target)
                save_path = root + (ext or ".png")
            img.save(save_path)
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
        # 加载 .ui 文件
        ui_path = os.path.join(os.path.dirname(__file__), 'ui', 'enhance_page.ui')
        loadUi(ui_path, self)

        # 绑定浏览按钮
        self.inpBrowseButton.clicked.connect(lambda: self.browse(self.inpLineEdit, True))
        self.outBrowseButton.clicked.connect(lambda: self.browse(self.outLineEdit, True))

        # 绑定开始/中止
        self.startButton.clicked.connect(self.start)
        self.abortButton.clicked.connect(self.abort)

        self.thread = None

    def browse(self, edit: QLineEdit, is_dir: bool):
        if is_dir:
            path = QFileDialog.getExistingDirectory(
                self, "选择目录", directory=os.getcwd(),
                options=QFileDialog.DontUseNativeDialog
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择文件", directory=os.getcwd(),
                options=QFileDialog.DontUseNativeDialog
            )
        if path:
            edit.setText(path)

    def start(self):
        inp = self.inpLineEdit.text().strip()
        out = self.outLineEdit.text().strip()
        if not inp or not out:
            QMessageBox.warning(self, "参数不全", "请填写输入和输出目录。")
            return
        self.thread = EnhanceThread(inp, out)
        self.thread.progress.connect(lambda v: self.progressBar.setValue(int(v * 100)))
        self.thread.finished.connect(self.on_done)
        self.startButton.setEnabled(False)
        self.abortButton.setEnabled(True)
        self.thread.start()

    def abort(self):
        if self.thread:
            self.thread.abort()
        self.startButton.setEnabled(True)
        self.abortButton.setEnabled(False)

    def on_done(self, paths: list):
        self.startButton.setEnabled(True)
        self.abortButton.setEnabled(False)
        # 清除旧预览
        for i in reversed(range(self.gridLayout.count())):
            widget = self.gridLayout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        # 显示新预览（最多 5 张）
        for idx, p in enumerate(paths[:5]):
            lbl = QLabel()
            lbl.setPixmap(QPixmap(p).scaled(100, 100, Qt.KeepAspectRatio))
            self.gridLayout.addWidget(lbl, idx // 5, idx % 5)


# —— UI 页面：识别 —— #
class DetectPage(QWidget):
    def __init__(self):
        super().__init__()
        ui_path = os.path.join(
            os.path.dirname(__file__), 'ui', 'detect_page.ui'
        )
        loadUi(ui_path, self)

        # 先隐藏所有可选控件，等切模式时再显示
        for w in (
            self.cropCheckBox, self.countCheckBox,
            self.simplifyCheckBox, self.fpsLabel,
            self.fpsLineEdit
        ):
            w.setVisible(False)

        # 绑定浏览 & 按钮
        self.inputBrowseButton.clicked.connect(lambda: self.browse(
            self.inputLineEdit, self._is_dir_mode()
        ))
        self.outputBrowseButton.clicked.connect(lambda: self.browse(
            self.outputLineEdit, True
        ))
        self.startButton.clicked.connect(self.start)
        self.abortButton.clicked.connect(self.abort)

        self.thread = None
        self.current_mode = None

    def set_mode(self, mode: str):
        """由 MainWindow.toolbar 调用，切换识别模式"""
        self.current_mode = mode

        # 文本 & 子控件显隐映射
        mapping = {
            'predict':    ('输入图像：', '保存目录：'),
            'video':      ('视频路径：', '保存路径：'),
            'fps':        ('测试图像：', ''),            # 不需要保存行
            'dir_predict':('输入目录：', '保存目录：'),
            'heatmap':    ('热力图源：', '保存路径：'),
            'export_onnx':('','ONNX 路径：'),
            'predict_onnx':('输入 ONNX：','保存目录：'),
        }
        in_txt, out_txt = mapping.get(mode, ('输入：','保存：'))
        # 设置标签
        self.inputLabel.setText(in_txt)
        self.outputLabel.setText(out_txt)

        # 控制行可见性
        has_input = bool(in_txt)
        has_output = bool(out_txt)
        for w in (self.inputLabel, self.inputLineEdit, self.inputBrowseButton):
            w.setVisible(has_input)
        for w in (self.outputLabel, self.outputLineEdit, self.outputBrowseButton):
            w.setVisible(has_output)

        # 根据模式显示额外控件
        show_crop    = (mode == 'predict')
        show_count   = (mode == 'predict')
        show_simplify= (mode == 'export_onnx')
        show_fps     = (mode in ('video','fps'))

        self.cropCheckBox.setVisible(show_crop)
        self.countCheckBox.setVisible(show_count)
        self.simplifyCheckBox.setVisible(show_simplify)

        # FPS 文本根据模式调整
        if mode == 'video':
            self.fpsLabel.setText('视频 FPS：')
        elif mode == 'fps':
            self.fpsLabel.setText('测试次数：')
        self.fpsLabel.setVisible(show_fps)
        self.fpsLineEdit.setVisible(show_fps)

    def _is_dir_mode(self):
        """dir_predict 模式下输入是目录，其它都是文件"""
        return self.current_mode == 'dir_predict'

    def browse(self, edit: QLineEdit, is_dir: bool):
        if is_dir:
            path = QFileDialog.getExistingDirectory(
                self, "选择目录", os.getcwd(),
                options=QFileDialog.DontUseNativeDialog
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择文件", os.getcwd(),
                options=QFileDialog.DontUseNativeDialog
            )
        if path:
            edit.setText(path)

    def start(self):
        m = self.current_mode
        params = {}
        # 收集参数
        if m == 'predict':
            params = {
                "input": self.inputLineEdit.text(),
                "save":  self.outputLineEdit.text(),
                "crop":  self.cropCheckBox.isChecked(),
                "count": self.countCheckBox.isChecked()
            }
        elif m == 'video':
            params = {
                "video_path": self.inputLineEdit.text(),
                "video_save": self.outputLineEdit.text(),
                "video_fps":  self.fpsLineEdit.text()
            }
        elif m == 'fps':
            params = {
                "fps_image":    self.inputLineEdit.text(),
                "test_interval": int(self.fpsLineEdit.text() or 100)
            }
        elif m == 'dir_predict':
            params = {
                "origin":   self.inputLineEdit.text(),
                "save_dir": self.outputLineEdit.text()
            }
        elif m == 'heatmap':
            params = {
                "heat_image": self.inputLineEdit.text(),
                "heat_save":  self.outputLineEdit.text()
            }
        elif m == 'export_onnx':
            params = {
                "simplify":   self.simplifyCheckBox.isChecked(),
                "onnx_path":  self.outputLineEdit.text()
            }
        elif m == 'predict_onnx':
            params = {
                "input_onnx": self.inputLineEdit.text(),
                "save_onnx":  self.outputLineEdit.text()
            }

        # 基本校验
        if not self.inputLineEdit.text() or (m != 'fps' and not self.outputLineEdit.text()):
            QMessageBox.warning(self, "参数不全", "请填写必要的输入/输出。")
            return

        # 启动线程
        self.thread = DetectThread(False, m, params)
        self.thread.progress.connect(lambda v: self.progressBar.setValue(int(v*100)))
        self.thread.finished.connect(self.on_done)
        self.startButton.setEnabled(False)
        self.abortButton.setEnabled(True)
        self.thread.start()

    def abort(self):
        if self.thread:
            self.thread.abort()
        self.startButton.setEnabled(True)
        self.abortButton.setEnabled(False)

    def on_done(self, paths: list, info):
        self.startButton.setEnabled(True)
        self.abortButton.setEnabled(False)
        # 清空旧预览
        for i in reversed(range(self.gridLayout_preview.count())):
            w = self.gridLayout_preview.itemAt(i).widget()
            if w: w.deleteLater()
        # 显示新结果
        for idx, p in enumerate(paths):
            lbl = QLabel()
            lbl.setPixmap(QPixmap(p).scaled(100,100,Qt.KeepAspectRatio))
            self.gridLayout_preview.addWidget(lbl, idx//5, idx%5)
        if info:
            QMessageBox.information(self, "信息", str(info))


# —— UI 页面：统计 —— #
class StatsPage(QWidget):
    def __init__(self):
        super().__init__()
        # 加载 .ui
        ui_path = os.path.join(os.path.dirname(__file__), 'ui', 'stats_page.ui')
        loadUi(ui_path, self)

        # 绑定浏览按钮
        self.originBrowseButton.clicked.connect(lambda: self.browse(self.originLineEdit, True))
        self.saveBrowseButton.clicked.connect(lambda: self.browse(self.saveLineEdit, True))

        # 绑定开始/中止
        self.startButton.clicked.connect(self.start)
        self.abortButton.clicked.connect(self.abort)
        self.abortButton.setEnabled(False)

        self.thread = None

    def browse(self, edit: QLineEdit, is_dir: bool):
        if is_dir:
            path = QFileDialog.getExistingDirectory(
                self, "选择目录", directory=os.getcwd(),
                options=QFileDialog.DontUseNativeDialog
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择文件", directory=os.getcwd(),
                options=QFileDialog.DontUseNativeDialog
            )
        if path:
            edit.setText(path)

    def start(self):
        origin = self.originLineEdit.text().strip()
        save_dir = self.saveLineEdit.text().strip()
        if not origin or not save_dir:
            QMessageBox.warning(self, "参数不全", "请填写输入和保存目录。")
            return

        det = Detector(False)
        self.thread = StatsThread(det, origin)
        self.thread.progress.connect(lambda v: self.progressBar.setValue(int(v * 100)))
        self.thread.finished.connect(lambda counts: self.on_done(counts, save_dir))

        self.startButton.setEnabled(False)
        self.abortButton.setEnabled(True)
        self.thread.start()

    def abort(self):
        if self.thread:
            self.thread.abort()
        self.startButton.setEnabled(True)
        self.abortButton.setEnabled(False)

    def on_done(self, counts: dict, save_dir: str):
        self.startButton.setEnabled(True)
        self.abortButton.setEnabled(False)

        # 生成图表
        pie, bar, line = generate_stats_charts(counts, save_dir)

        # 清空旧图
        for i in reversed(range(self.chartGridLayout.count())):
            w = self.chartGridLayout.itemAt(i).widget()
            if w:
                w.deleteLater()

        # 添加新图（3 张）
        for idx, img_path in enumerate((pie, bar, line)):
            lbl = QLabel()
            pix = QPixmap(img_path).scaled(300, 300, Qt.KeepAspectRatio)
            lbl.setPixmap(pix)
            self.chartGridLayout.addWidget(lbl, idx // 3, idx % 3)


# —— 主窗口 —— #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 加载 .ui
        ui_path = os.path.join(os.path.dirname(__file__), 'ui', 'mainwindow.ui')
        loadUi(ui_path, self)

        # 创建各功能页面
        self.enh_page = EnhancePage()
        self.det_page = DetectPage()
        self.stats_page = StatsPage()

        # 把页面加入 stackedWidget
        self.stackedWidget.addWidget(self.enh_page)
        self.stackedWidget.addWidget(self.det_page)
        self.stackedWidget.addWidget(self.stats_page)

        # 连接工具栏按钮
        self.actionExit.triggered.connect(self.close)
        self.actionEnhance.triggered.connect(lambda: self._switch_main('enhance'))
        self.actionStats.triggered.connect(lambda: self._switch_main('stats'))
        self.actionDetectPredict.triggered.connect(lambda: self._switch_detect('predict'))
        self.actionDetectVideo.triggered.connect(lambda: self._switch_detect('video'))
        self.actionFPSTest.triggered.connect(lambda: self._switch_detect('fps'))
        self.actionDirPredict.triggered.connect(lambda: self._switch_detect('dir_predict'))
        self.actionHeatmap.triggered.connect(lambda: self._switch_detect('heatmap'))
        self.actionExportONNX.triggered.connect(lambda: self._switch_detect('export_onnx'))
        self.actionPredictONNX.triggered.connect(lambda: self._switch_detect('predict_onnx'))

        # 默认显示“增强”页
        self._switch_main('enhance')

    def _switch_main(self, name: str):
        """切换到 EnhancePage 或 StatsPage"""
        if name == 'enhance':
            self.stackedWidget.setCurrentWidget(self.enh_page)
        elif name == 'stats':
            self.stackedWidget.setCurrentWidget(self.stats_page)

    def _switch_detect(self, mode: str):
        """切换到 DetectPage 并设置具体识别模式"""
        self.stackedWidget.setCurrentWidget(self.det_page)
        self.det_page.set_mode(mode)


def main():
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
