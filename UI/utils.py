#!/usr/bin/env python3
"""
utils.py

提供图像读写函数，支持 RGB 格式
"""

import os
import cv2
import numpy as np


def load_img(path: str) -> np.ndarray:
    """
    读取图像并返回 HWC 格式的 RGB numpy 数组（dtype uint8 或 float）
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    # 若有 alpha 通道，舍弃
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    # 若灰度图，转换为 RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_img(path: str, img: np.ndarray) -> None:
    """
    保存 HWC 格式的 RGB numpy 数组到路径，自动创建目录
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 确保 uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # RGB -> BGR
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
