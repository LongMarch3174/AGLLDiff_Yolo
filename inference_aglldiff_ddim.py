import argparse
import os
import os.path as osp
import collections

import cv2
import numpy as np
import torch as th
import torch.nn.functional as F

from Rnet import net as Rnet
from utils.Attribute import *
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion import logger


def main(inference_step=None):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # 各种损失
    L_spa = L_structure2()
    L_exp = L_exp2(1)
    L_fft_loss = L_fft()

    def attribute_guidance(x, t, y=None, pred_xstart=None, **kwargs):
        assert y is not None
        with th.enable_grad():
            predicted_start = pred_xstart.detach().requires_grad_(True)
            total_loss = 0

            print(f'[t={str(t.cpu().numpy()[0]).zfill(3)}]', end=' ')

            # 归一化权重计算
            tau = t.float() / (inference_step - 1)
            tau = tau.unsqueeze(0)
            structure_weight = args.structure_weight * (1 - tau)
            exposure_weight = args.exposure_weight * tau
            color_map_weight = args.color_map_weight
            fft_weight = args.fft_weight * (1 - (2 * tau - 1).abs())

            # 损失计算
            spatial_loss = th.mean(L_spa(((y + 1) * 0.5), ((predicted_start + 1) * 0.5))) * structure_weight
            illum_loss = L_exp(((predicted_start + 1) * 0.5), kwargs.get('exposure_map')) * exposure_weight
            refl_loss = F.mse_loss(kwargs.get('reflectence_map'), ((predicted_start + 1) * 0.5),
                                   reduction='sum') * color_map_weight
            fft_loss = L_fft_loss(((predicted_start + 1) * 0.5), ((y + 1) * 0.5)) * fft_weight
            total_loss = spatial_loss + illum_loss + refl_loss + fft_loss

            print(f'loss (structure): {spatial_loss.item()};', end=' ')
            print(f'loss (exposure): {illum_loss.item()};', end=' ')
            print(f'loss (color): {refl_loss};', end=' ')
            print(f'loss (fft): {fft_loss.item()};', end=' ')
            print(f'loss (total): {total_loss.item()};')

            gradient = th.autograd.grad(total_loss, predicted_start)[0]
        return gradient

    def model_fn(x, t, **kwargs):
        return model(x, t, kwargs.get('y') if args.class_cond else None)

    args = create_argparser().parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = f"{args.out_dir}/ddim_s{args.guidance_scale}_steps{args.timestep_respacing}_seed{args.seed}"
    logger.configure(dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger.log("Creating model and diffusion.")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    state_dict = th.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 加载Retinex模型
    retinex_model = Rnet().to(device)
    if args.retinex_model.endswith('.ckpt'):
        print("Loading Rnet checkpoint")
        ckpt = th.load(args.retinex_model, map_location=lambda storage, loc: storage)
        new_state_dict = collections.OrderedDict()
        for k in ckpt['state_dict']:
            if k.startswith('model.'):
                new_state_dict[k[6:]] = ckpt['state_dict'][k]
        retinex_model.load_state_dict(new_state_dict, strict=True)
    else:
        retinex_model.load_state_dict(th.load(args.retinex_model, map_location=lambda storage, loc: storage))
    print('Pre-trained retinex model is loaded.')
    retinex_model.eval()

    seed = args.seed

    logger.log("Sampling with DDIM.")
    for img_name in sorted(os.listdir(args.in_dir)):
        img_path = osp.join(args.in_dir, img_name)
        raw = cv2.imread(img_path).astype(np.float32)[:, :, [2, 1, 0]]
        y0 = th.tensor(raw / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to(device)

        print(img_name)

        model_kwargs = {
            'task': args.task,
            'scale': args.guidance_scale,
            'N': args.N,
            'exposure_map': check_image_size(
                calculate_spatially_varying_exposure(img_path, args.base_exposure, args.adjustment_amplitude)),
            'y': check_image_size(y0),
            'reflectence_map': check_image_size(
                calculate_color_map(th.tensor(raw / 255).permute(2, 0, 1).unsqueeze(0).to(device), retinex_model))
        }
        b, c, h, w = model_kwargs['y'].shape
        # 根据 use_ddim 选择采样方法，无需向 ddim_sample_loop 传递 seed 或 inference_step
        if args.use_ddim:
            sample = diffusion.ddim_sample_loop(
                model_fn,
                (args.batch_size, 3, h, w),
                noise=None,
                clip_denoised=args.clip_denoised,
                denoised_fn=None,
                cond_fn=attribute_guidance,
                model_kwargs=model_kwargs,
                device=device,
                eta=args.eta
            )

        out = ((sample[:, :, :raw.shape[0], :raw.shape[1]] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        out = out.permute(0, 2, 3, 1).cpu().numpy()[0][:, :, ::-1]
        cv2.imwrite(f"{out_dir}/{img_name}", out)

    logger.log("DDIM sampling complete.")


def create_argparser():
    defaults = dict(
        seed=12345678,
        task='LIE',
        in_dir='./examples',
        out_dir='./results',
        clip_denoised=True,
        batch_size=1,
        use_ddim=True,
        eta=0.0,
        timestep_respacing="10",
        inference_step=10,
        model_path='./ckpt/256x256_diffusion_uncond.pt',
        retinex_model='./ckpt/RNet_1688_step.ckpt',
        guidance_scale=5,
        structure_weight=10,
        color_map_weight=0.003,
        exposure_weight=1000,
        fft_weight=10,
        base_exposure=0.46,
        adjustment_amplitude=0.25,
        N=2,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults["timestep_respacing"] = "10"
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()
    main(inference_step=int(args.timestep_respacing))
