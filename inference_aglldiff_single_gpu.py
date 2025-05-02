import argparse
import os
import cv2
import os.path as osp
import numpy as np
import torch as th
import torch.nn.functional as F
import collections

from Rnet import net as Rnet
from utils.Attribute import *
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion import logger


"""
单GPU推理
"""


def main(inference_step=None):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    L_spa = L_structure2()
    L_exp = L_exp2(1)
    L_fft_loss = L_fft_multiscale()
    loss_weighter = AdaptiveLossWeighting(
        num_losses=4,
        mode='uncertainty',  # 或 'softmax'
        tau=0.5,
        ema_beta=0.9
    ).to(device)

    args = create_argparser().parse_args()

    if os.path.isfile(args.loss_weight_path):
        loss_weighter.load_state_dict(th.load(args.loss_weight_path, map_location=device))
        print(f"Loaded loss weights from {args.loss_weight_path}")
    else:
        print(f"Warning: loss_weight_path not found at {args.loss_weight_path}")

    loss_weighter.eval()  # ✅ 推理阶段关闭梯度更新

    def attribute_guidance(x, t, y=None, pred_xstart=None, target=None, ref=None, mask=None,
                           task="LIE", scale=0, N=None, exposure_map=None, reflectence_map=None):
        assert y is not None
        with torch.enable_grad():
            predicted_start = pred_xstart.detach().requires_grad_(True)

            predicted_start_norm = (predicted_start + 1) * 0.5
            target_norm = (y + 1) * 0.5

            # individual losses
            loss_structure = torch.mean(L_spa(target_norm, predicted_start_norm))
            loss_exposure = L_exp(predicted_start_norm, exposure_map)
            loss_reflect = F.mse_loss(reflectence_map, predicted_start_norm, reduction='sum')
            loss_fft = L_fft_loss(predicted_start_norm, target_norm)

            # combine with learnable weights
            total_loss, w = loss_weighter([loss_structure, loss_exposure, loss_reflect, loss_fft])

            print("W=[%s]  Ls=[%.3f, %.3f, %.3f, %.3f]  Total=%.3f" %
                  (", ".join(["%.3f" % x for x in w]),
                   loss_structure.item(), loss_exposure, loss_reflect.item(), loss_fft, total_loss))

            gradient = torch.autograd.grad(total_loss, predicted_start)[0]

        return gradient, None

    def model_fn(x, t, y=None, target=None, ref=None, mask=None, task=None, scale=0, N=1,
                 exposure_map=None, reflectence_map=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = f'{args.out_dir}/s{args.guidance_scale}_sw{args.structure_weight}_cw{args.color_map_weight}_ew{args.exposure_weight}_be{args.base_exposure}_aa{args.adjustment_amplitude}_seed{args.seed}'
    logger.configure(dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    state_dict = th.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print('===> Building retinex model')
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

    print("=================== Summary (Sampling) ===================")
    print(f'Task: {args.task}; Guidance scale: {args.guidance_scale}')
    print(f'structure weight: {args.structure_weight}')
    print(f'color weight: {args.color_map_weight}')
    print(f'exposure weight: {args.exposure_weight}')
    print("==========================================================")

    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    all_images = []
    lr_images = sorted(os.listdir(args.in_dir))

    logger.log("Sampling...")

    for img_name in lr_images:
        path_lq = osp.join(args.in_dir, img_name)
        raw = cv2.imread(path_lq).astype(np.float32)[:, :, [2, 1, 0]]
        y00 = th.tensor(raw / 255).permute(2, 0, 1).unsqueeze(0).to(device)
        y0 = th.tensor(raw / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to(device)

        print(img_name)
        _, _, H, W = y0.shape

        model_kwargs = {
            "task": args.task,
            "target": None,
            "scale": args.guidance_scale,
            "N": args.N,
            "exposure_map": check_image_size(
                calculate_spatially_varying_exposure(path_lq, args.base_exposure, args.adjustment_amplitude)),
            "y": check_image_size(y0),
            "reflectence_map": check_image_size(calculate_color_map(y00, retinex_model))
        }

        b, c, h, w = model_kwargs["y"].shape

        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, h, w),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=attribute_guidance,
            device=device,
            seed=seed,
            inference_step=inference_step
        )

        sample = ((sample[:, :, :H, :W] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()

        all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} sample")

        cv2.imwrite(f'{out_dir}/{img_name}', all_images[-1][0][..., [2, 1, 0]])
        th.cuda.empty_cache()

    logger.log("Sampling complete!")


def create_argparser():
    defaults = dict(
        seed=12345678,
        task='LIE',
        in_dir='./examples',
        out_dir='./results',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="./ckpt/256x256_diffusion_uncond.pt",
        retinex_model="./ckpt/RNet_1688_step.ckpt",
        loss_weight_path="./ckpt/weight_epoch4.pth",
        guidance_scale=2.3,
        structure_weight=10,
        color_map_weight=0.03,
        exposure_weight=1000,
        fft_weight=10,  # 新增FFT频域loss权重
        base_exposure=0.46,
        adjustment_amplitude=0.25,
        N=2,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main(inference_step=10)
