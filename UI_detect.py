#!/usr/bin/env python3
"""
detector.py — 与之前基本相同，只是 YOLO/YOLO_ONNX 的实例化
也完全留在每个方法内部，保证 __init__ 轻量。
"""

import os
import cv2
import time
from typing import Callable, List, Optional, Union, Dict

import numpy as np
from PIL import Image

from yolo import YOLO, YOLO_ONNX


class Detector:
    def __init__(self, use_onnx: bool = False):
        """
        只存储标志，不加载模型
        """
        self.use_onnx = use_onnx

    def _load(self):
        model = YOLO_ONNX() if self.use_onnx else YOLO()
        if not hasattr(model, 'detect_image'):
            raise FileNotFoundError("未能加载模型，请检查模型文件是否存在。")
        return model

    def predict_image(
        self,
        img: Union[str, Image.Image],
        crop: bool = False,
        count: bool = False,
    ) -> Image.Image:
        yolo = self._load()
        if isinstance(img, str):
            if not os.path.exists(img):
                raise FileNotFoundError(f"图像文件不存在：{img}")
            img = Image.open(img)
        return yolo.detect_image(img, crop=crop, count=count)

    def dir_predict(
        self,
        origin_dir: str,
        save_dir: str,
        progress_cb: Optional[Callable[[float], None]] = None,
        cancel_cb: Optional[Callable[[], bool]] = None,
    ) -> List[str]:
        if not os.path.exists(origin_dir):
            raise FileNotFoundError(f"输入目录不存在：{origin_dir}")

        yolo = self._load()
        os.makedirs(save_dir, exist_ok=True)
        files = [
            f for f in os.listdir(origin_dir)
            if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))
        ]
        total = len(files)
        out_paths = []

        for idx, fn in enumerate(files, start=1):
            if cancel_cb and cancel_cb():
                break
            path = os.path.join(origin_dir, fn)
            img = Image.open(path)
            res = yolo.detect_image(img)
            out_name = os.path.splitext(fn)[0] + ".png"
            save_path = os.path.join(save_dir, out_name)
            res.save(save_path, quality=95, subsampling=0)
            out_paths.append(save_path)
            if progress_cb:
                progress_cb(idx / total)

        return out_paths

    def video_predict(
        self,
        video_path: Union[str, int],
        save_path: Optional[str] = None,
        fps: float = 25.0,
        progress_cb: Optional[Callable[[float], None]] = None,
        cancel_cb: Optional[Callable[[], bool]] = None,
    ) -> None:
        if isinstance(video_path, str) and not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在：{video_path}")

        yolo = self._load()
        cap = cv2.VideoCapture(video_path)
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(save_path, fourcc, fps, (w,h))
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        idx = 0

        while True:
            if cancel_cb and cancel_cb(): break
            ret, frame = cap.read()
            if not ret: break
            idx += 1
            t0 = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            out_pil = yolo.detect_image(pil)
            out_np = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
            fps_val = 1.0 / (time.time() - t0)
            cv2.putText(out_np, f"FPS:{fps_val:.2f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imshow("Detection", out_np)
            if writer: writer.write(out_np)
            if progress_cb and total>0: progress_cb(idx/total)
            if cv2.waitKey(1)&0xFF==27: break

        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()

    def fps_test(
        self, img_path: str, test_interval: int = 100
    ) -> float:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"测试图像不存在：{img_path}")

        yolo = self._load()
        img = Image.open(img_path)
        return yolo.get_FPS(img, test_interval)

    def detect_heatmap(self, img_path: str, save_path: str) -> None:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"热力图图像不存在：{img_path}")

        yolo = self._load()
        img = Image.open(img_path)
        yolo.detect_heatmap(img, save_path)

    def export_onnx(self, simplify: bool, onnx_path: str) -> None:
        yolo = self._load()
        yolo.convert_to_onnx(simplify, onnx_path)

    def stats_predict(
            self,
            origin_dir: str,
            progress_cb: Optional[Callable[[float], None]] = None,
            cancel_cb: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, int]:
        """
        批量对 origin_dir 下所有图片进行检测，统计每个类别出现次数。
        :param progress_cb: 接收 [0,1] 进度
        :param cancel_cb: 返回 True 时中止
        :return: {类别: 数量}
        """
        from collections import Counter

        if not os.path.exists(origin_dir):
            raise FileNotFoundError(f"输入目录不存在：{origin_dir}")

        yolo = self._load()
        files = [
            f for f in os.listdir(origin_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        total = len(files)
        counter = Counter()

        for idx, fn in enumerate(files, start=1):
            if cancel_cb and cancel_cb():
                break
            path = os.path.join(origin_dir, fn)
            img = Image.open(path)
            _, counts = yolo.detect_image(img, count=True)
            counter.update(counts)
            if progress_cb:
                progress_cb(idx / total)

        return dict(counter)
