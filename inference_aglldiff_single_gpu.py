#!/usr/bin/env python
# inference_aglldiff_single_gpu.py
# -----------------------------------------------------------
# Single‑GPU inference with attribute‑guided diffusion
# -----------------------------------------------------------

import argparse
import os
import cv2
import os.path as osp
import numpy as np
import torch as th
import torch.nn.functional as F
import collections

# local modules
from Rnet import net as Rnet
from utils.Attribute import (
    L_structure2, L_exp2, L_fft_multiscale,
    AdaptiveLossWeighting,                 # 自适应权重
    calculate_spatially_varying_exposure,
    calculate_color_map,
    check_image_size
)
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion import logger


# ------------------------------  inference  ------------------------------ #
def main(inference_step=None):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    # ── loss modules ────────────────────────────────────────────────────── #
    L_spa       = L_structure2()
    L_exp       = L_exp2(1)
    L_fft_loss  = L_fft_multiscale()

    # 自适应权重：5 个 loss（noise + structure + exposure + reflect + fft）
    loss_weighter = AdaptiveLossWeighting(
        num_losses=5,
        mode='softmax',      # or 'softmax'
        tau=0.5,
        ema_beta=0.9
    ).to(device)

    # ── CLI args ───────────────────────────────────────────────────────── #
    args = create_argparser().parse_args()

    # 载入训练阶段保存的自适应权重
    if os.path.isfile(args.loss_weight_path):
        loss_weighter.load_state_dict(
            th.load(args.loss_weight_path, map_location=device)
        )
        print(f"[✓] Loaded loss‑weights from {args.loss_weight_path}")
    else:
        print(f"[!] loss_weight_path not found: {args.loss_weight_path}")
    loss_weighter.eval()   # 推理期冻结

    # -------------------------------------------------------------------- #
    #  attribute guidance  (cond_fn)
    # -------------------------------------------------------------------- #
    def attribute_guidance(x, t, y=None, pred_xstart=None,
                           target=None, ref=None, mask=None,
                           task="LIE", scale=0, N=None,
                           exposure_map=None, reflectence_map=None):
        """
        cond_fn: returns ∂loss/∂x0  (gradient wrt predicted clean image)
        """
        assert y is not None
        with th.enable_grad():
            predicted_start = pred_xstart.detach().requires_grad_(True)

            print(f'[t={str(t.cpu().numpy()[0]).zfill(3)}]', end=' ')

            predicted_start_norm = (predicted_start + 1) * 0.5
            target_norm          = (y + 1) * 0.5

            # ── individual losses ──────────────────────────────────── #
            # 占位噪声 loss（无梯度，仅对齐向量）
            loss_noise = th.zeros_like(predicted_start_norm.mean())

            loss_structure = th.mean(L_spa(target_norm, predicted_start_norm))
            loss_exposure = L_exp(predicted_start_norm, exposure_map)
            loss_reflect = F.mse_loss(reflectence_map, predicted_start_norm, reduction='sum')
            loss_fft = L_fft_loss(predicted_start_norm, target_norm)

            # ── 自适应权重（5个） ───────────────────────────────────── #
            losses_all = [loss_noise, loss_structure, loss_exposure, loss_reflect, loss_fft]
            _, weights_all = loss_weighter(losses_all)

            # === 仅用第1~4项（去掉占位项），并归一化权重 === #
            w_active = weights_all[1:] / weights_all[1:].sum()
            loss_active = [loss_structure, loss_exposure, loss_reflect, loss_fft]

            # 组合真实 loss
            total_loss = sum(wi * li for wi, li in zip(w_active, loss_active))

            # 打印信息
            print("W=[%s]  Ls=[structure: %.3f, exposure: %.3f, color: %.3f, fft: %.3f]  Total=%.3f" %
                  (", ".join([f"{v:.3f}" for v in w_active]),
                   loss_structure.item(), loss_exposure.item(),
                   loss_reflect.item(), loss_fft.item(),
                   total_loss.item()))

            # gradient wrt predicted_start
            gradient = th.autograd.grad(total_loss, predicted_start)[0]
        return gradient, None

    # model_fn: unconditional forward (class_cond 可选)
    def model_fn(x, t, y=None, target=None, ref=None, mask=None,
                 task=None, scale=0, N=1,
                 exposure_map=None, reflectence_map=None):
        return model(x, t, y if args.class_cond else None)

    # ── output dir ─────────────────────────────────────────────────────── #
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = (
        f"{args.out_dir}/"
        f"s{args.guidance_scale}_"
        f"sw{args.structure_weight}_cw{args.color_map_weight}_"
        f"ew{args.exposure_weight}_"
        f"be{args.base_exposure}_aa{args.adjustment_amplitude}_"
        f"seed{args.seed}"
    )
    logger.configure(dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ── create model & diffusion ───────────────────────────────────────── #
    logger.log("Creating model and diffusion ...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    state_dict = th.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # ── Retinex model for color map ────────────────────────────────────── #
    print("===> Building Retinex model")
    retinex_model = Rnet().to(device)
    if args.retinex_model.endswith(".ckpt"):
        ckpt = th.load(args.retinex_model, map_location="cpu")
        new_sd = collections.OrderedDict(
            (k[6:], v) for k, v in ckpt["state_dict"].items() if k.startswith("model.")
        )
        retinex_model.load_state_dict(new_sd, strict=True)
    else:
        retinex_model.load_state_dict(
            th.load(args.retinex_model, map_location="cpu")
        )
    retinex_model.eval()
    print("[✓] Pre‑trained Retinex model loaded")

    # ── summary print ──────────────────────────────────────────────────── #
    print("===================  Summary (Sampling)  ===================")
    print(f" Task: {args.task}")
    print(f" Guidance scale:   {args.guidance_scale}")
    print(f" structure weight: {args.structure_weight}")
    print(f" color   weight:   {args.color_map_weight}")
    print(f" exposure weight:  {args.exposure_weight}")
    print("============================================================")

    # ── seed ──────────────────────────────────────────────────────────── #
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    # ── image list ─────────────────────────────────────────────────────── #
    lr_images = sorted(os.listdir(args.in_dir))
    all_images = []

    logger.log("Sampling ...")
    for img_name in lr_images:
        print(img_name)
        path_lq = osp.join(args.in_dir, img_name)
        raw = cv2.imread(path_lq).astype(np.float32)[:, :, [2, 1, 0]]
        y00 = th.tensor(raw / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        y0  = th.tensor(raw / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
        _, _, H, W = y0.shape

        # —— prepare per‑image maps —— #
        exposure_map = calculate_spatially_varying_exposure(
            path_lq, args.base_exposure, args.adjustment_amplitude
        )
        exposure_map = check_image_size(exposure_map)
        reflect_map  = calculate_color_map(y00, retinex_model)
        reflect_map  = check_image_size(reflect_map)

        model_kwargs = dict(
            task=args.task,
            target=None,
            scale=args.guidance_scale,
            N=args.N,
            exposure_map=exposure_map,
            reflectence_map=reflect_map,
            y=check_image_size(y0),
        )

        b, c, h, w = model_kwargs["y"].shape
        sample_fn = (
            diffusion.ddim_sample_loop if args.use_ddim
            else diffusion.p_sample_loop
        )

        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, h, w),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=attribute_guidance,
            device=device,
            seed=seed,
            inference_step=inference_step,
        )

        # —— save —— #
        sample = ((sample[:, :, :H, :W] + 1) * 127.5
                  ).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()

        all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} sample")

        cv2.imwrite(
            f"{out_dir}/{img_name}",
            all_images[-1][0][..., [2, 1, 0]]
        )
        th.cuda.empty_cache()

    logger.log("Sampling complete!")


# ------------------------------  CLI  ------------------------------ #
def create_argparser():
    defaults = dict(
        seed=12345678,
        task="LIE",
        in_dir="./examples",
        out_dir="./results",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="./ckpt/256x256_diffusion_uncond.pt",
        retinex_model="./ckpt/RNet_1688_step.ckpt",
        loss_weight_path="./ckpt/weight_epoch7.pth",
        guidance_scale=2.3,
        structure_weight=10,
        color_map_weight=0.03,
        exposure_weight=1000,
        fft_weight=10,            # weight for FFT loss
        base_exposure=0.46,
        adjustment_amplitude=0.25,
        N=2,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


# ------------------------------------------------------------------- #
if __name__ == "__main__":
    # inference_step 可选：None = full steps；整数 = DDIM/DPMSolver step
    main(inference_step=5)
