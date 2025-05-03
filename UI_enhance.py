#!/usr/bin/env python3
"""
enhancer.py — 与之前完全一致，无需做任何变动。
模型加载、推理、协作式中止都在 enhance() 方法中完成。
"""

import os
from glob import glob
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.util import img_as_ubyte

from basicsr.models import create_model
from basicsr.utils.options import parse
import UI.utils  # load_img, save_img


class Enhancer:
    def __init__(
        self,
        opt_path: str,
        weights_path: str,
        gpus: str = "0",
        self_ensemble: bool = False,
    ):
        """
        这里只存储参数，模型真正的创建会在 enhance() 中进行，
        所以 __init__ 本身非常轻量，不会阻塞 UI。
        """
        self.opt_path = opt_path
        self.weights_path = weights_path
        self.gpus = gpus
        self.use_ensemble = self_ensemble

    def _self_ensemble(self, x: torch.Tensor, model: nn.Module) -> torch.Tensor:
        def transform(t, h, v, r):
            if h: t = torch.flip(t, (-2,))
            if v: t = torch.flip(t, (-1,))
            if r: t = torch.rot90(t, (-2, -1))
            out = model(t)
            if r: out = torch.rot90(out, (-2, -1), k=3)
            if v: out = torch.flip(out, (-1,))
            if h: out = torch.flip(out, (-2,))
            return out

        outs = []
        for h in (False, True):
            for v in (False, True):
                for r in (False, True):
                    outs.append(transform(x, h, v, r))
        return torch.mean(torch.stack(outs), dim=0, keepdim=True)

    def enhance(
        self,
        input_path: str,
        output_dir: str,
        progress_cb: Optional[Callable[[float], None]] = None,
        cancel_cb: Optional[Callable[[], bool]] = None,
    ) -> List[str]:
        """
        在此方法里加载模型，将阻塞控制在后台线程中。
        """
        # 1. 加载模型
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
        opt = parse(self.opt_path, is_train=False)
        opt["dist"] = False
        net = create_model(opt).net_g
        ckpt = torch.load(self.weights_path)
        try:
            net.load_state_dict(ckpt["params"])
        except RuntimeError:
            net.load_state_dict({f"module.{k}": v for k, v in ckpt["params"].items()})
        model = nn.DataParallel(net.cuda()).eval()

        # 2. 收集文件
        os.makedirs(output_dir, exist_ok=True)
        if os.path.isfile(input_path):
            paths = [input_path]
        else:
            paths = sorted(
                glob(os.path.join(input_path, "*.png")) +
                glob(os.path.join(input_path, "*.jpg")) +
                glob(os.path.join(input_path, "*.jpeg"))
            )
        total = len(paths)
        out_paths = []

        # 3. 推理循环
        with torch.no_grad():
            for idx, p in enumerate(paths, start=1):
                print(p)
                if cancel_cb and cancel_cb():
                    break

                img = np.float32(UI.utils.load_img(p)) / 255.0
                t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
                _, _, h, w = t.shape
                H = ((h + 3) // 4) * 4
                W = ((w + 3) // 4) * 4
                t = F.pad(t, (0, W - w, 0, H - h), mode="reflect")

                if cancel_cb and cancel_cb():
                    break

                if self.use_ensemble:
                    out_t = self._self_ensemble(t, model)
                else:
                    out_t = model(t)

                if cancel_cb and cancel_cb():
                    break

                out_t = out_t[..., :h, :w].clamp(0, 1)
                out_img = out_t.cpu().squeeze(0).permute(1, 2, 0).numpy()
                name = os.path.splitext(os.path.basename(p))[0] + ".png"
                save_path = os.path.join(output_dir, name)
                UI.utils.save_img(save_path, img_as_ubyte(out_img))
                out_paths.append(save_path)

                if progress_cb:
                    progress_cb(idx / total)

        return out_paths
