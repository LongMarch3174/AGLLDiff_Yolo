#!/usr/bin/env python3
# UI_enhance.py - 基于属性引导扩散模型的图像增强接口，用于 UI 调用

import os
import cv2
from glob import glob
from typing import Callable, List, Optional

import torch as th
import numpy as np

from guided_diffusion.script_util import create_model_and_diffusion, args_to_dict, model_and_diffusion_defaults
from utils.Attribute import (
    calculate_spatially_varying_exposure,
    calculate_color_map,
    check_image_size,
    L_structure2, L_exp2, L_fft_multiscale, AdaptiveLossWeighting
)
from guided_diffusion import logger
from Rnet import net as Rnet


class Enhancer:
    def __init__(self, opt_path=None, weights_path=None, gpus="0", self_ensemble=False):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

        # === 检查模型文件是否存在 ===
        required_files = {
            "model_data/256x256_diffusion_uncond.pt": "主模型权重",
            "model_data/RNet_1688_step.ckpt": "Retinex 模型",
            "model_data/weight_epoch7.pth": "Loss 权重"
        }
        for path, name in required_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"未找到{name}：{path}")

        # === 加载模型 ===
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(self._default_args(), model_and_diffusion_defaults().keys())
        )
        self.model.load_state_dict(th.load("model_data/256x256_diffusion_uncond.pt", map_location="cpu"))
        self.model.to(self.device).eval()

        self.retinex_model = Rnet().to(self.device)
        ckpt = th.load("model_data/RNet_1688_step.ckpt", map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        self.retinex_model.load_state_dict(ckpt, strict=True)
        self.retinex_model.eval()

        self.loss_weighter = AdaptiveLossWeighting(num_losses=5, mode='softmax', tau=0.7, ema_beta=0.9).to(self.device)
        self.loss_weighter.load_state_dict(th.load("model_data/weight_epoch7.pth", map_location=self.device))
        self.loss_weighter.eval()

        self.L_spa = L_structure2()
        self.L_exp = L_exp2(1)
        self.L_fft = L_fft_multiscale()

    def _default_args(self):
        return dict(
            seed=12345678, task="LIE",
            clip_denoised=True, num_samples=1, batch_size=1, use_ddim=False,
            model_path="model_data/256x256_diffusion_uncond.pt",
            retinex_model="model_data/RNet_1688_step.ckpt",
            loss_weight_path="model_data/weight_epoch7.pth",
            guidance_scale=2.3, structure_weight=10, color_map_weight=0.03,
            exposure_weight=1000, fft_weight=10,
            base_exposure=0.46, adjustment_amplitude=0.25, N=2
        )

    def enhance(self, input_path: str, output_dir: str,
                progress_cb: Optional[Callable[[float], None]] = None,
                cancel_cb: Optional[Callable[[], bool]] = None) -> List[str]:

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入路径不存在：{input_path}")

        os.makedirs(output_dir, exist_ok=True)
        paths = [input_path] if os.path.isfile(input_path) else sorted(
            glob(os.path.join(input_path, "*.png")) +
            glob(os.path.join(input_path, "*.jpg")) +
            glob(os.path.join(input_path, "*.jpeg"))
        )
        total = len(paths)
        out_paths = []

        def attribute_guidance(x, t, y=None, pred_xstart=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                pred = pred_xstart.detach().requires_grad_(True)
                y_pred = (pred + 1) * 0.5
                y_true = (y + 1) * 0.5
                l_structure = th.mean(self.L_spa(y_true, y_pred))
                l_exposure = self.L_exp(y_pred, kwargs["exposure_map"])
                l_reflect = th.nn.functional.mse_loss(kwargs["reflectence_map"], y_pred, reduction="sum")
                l_fft = self.L_fft(y_pred, y_true)
                loss_all = [th.zeros_like(l_structure), l_structure, l_exposure, l_reflect, l_fft]
                _, weights = self.loss_weighter(loss_all)
                weights = weights[1:] / weights[1:].sum()
                total_loss = sum(w * l for w, l in zip(weights, [l_structure, l_exposure, l_reflect, l_fft]))
                return th.autograd.grad(total_loss, pred)[0], None

        for idx, p in enumerate(paths, start=1):
            if cancel_cb and cancel_cb():
                break
            img_name = os.path.basename(p)
            raw = cv2.imread(p).astype(np.float32)[:, :, [2, 1, 0]]
            y00 = th.tensor(raw / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            y0 = th.tensor(raw / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

            exposure_map = calculate_spatially_varying_exposure(p, 0.46, 0.25)
            reflect_map = calculate_color_map(y00, self.retinex_model)

            model_kwargs = {
                "task": "LIE",
                "y": check_image_size(y0),
                "exposure_map": check_image_size(exposure_map),
                "reflectence_map": check_image_size(reflect_map),
                "scale": 2.3,
                "N": 2
            }
            _, _, h, w = model_kwargs["y"].shape
            sample = self.diffusion.p_sample_loop(
                lambda x, t, **kw: self.model(x, t, kw.get("y")),
                (1, 3, h, w),
                clip_denoised=True,
                model_kwargs=model_kwargs,
                cond_fn=attribute_guidance,
                device=self.device,
                seed=12345678,
                inference_step=5
            )
            sample = ((sample[:, :, :h, :w] + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1).cpu().numpy()[0][..., [2, 1, 0]]
            out_path = os.path.join(output_dir, img_name)
            cv2.imwrite(out_path, sample)
            out_paths.append(out_path)
            if progress_cb:
                progress_cb(idx / total)

        return out_paths
