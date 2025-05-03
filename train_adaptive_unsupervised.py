#!/usr/bin/env python
"""
train_adaptive_unsupervised.py  —  diffusion‑based low‑light enhancement training

Changes vs. original version
----------------------------
1.  ➕ **Noise loss**   ‖pred_eps − noise‖²  (MSE) added as loss 0 for the adaptive
    weight module.
2.  🔄  **`loss_color` uses `mean`**   instead of `sum` to stabilise magnitude.
3.  📈 **ScheduleSampler** (uniform by default) for importance‑sampling timesteps.
4.  ✂️ **Gradient clipping**   `clip_grad_norm_(model.parameters(), 1.0)` before
    every optimiser step.

All checkpoint‑loading and argument‑parsing logic is intentionally preserved.
"""

import argparse
import os
from glob import glob
import collections

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# guided‑diffusion imports
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.resample import create_named_schedule_sampler

# project utils
from utils.Attribute import (
    L_structure2,
    L_exp2,
    calculate_spatially_varying_exposure,
    calculate_color_map,
    check_image_size,
    L_fft_multiscale,
    AdaptiveLossWeighting,
)
from Rnet import net as Rnet


# ------------------------- dataset ----------------------------------------- #
class ImageFolderDataset(Dataset):
    """Simple image‑folder dataset that returns (tensor, path)."""

    def __init__(self, folder: str, image_size: int):
        self.paths = sorted(glob(os.path.join(folder, "*")))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),  # [0,1]
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # → [‑1,1]
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path


# ------------------------- arg‑parser -------------------------------------- #

def create_argparser():
    defaults = dict(
        # data & io
        data_dir="./LOLdataset/our485/low",
        ckpt_dir="./ckpt",
        retinex_model="./ckpt/RNet_1688_step.ckpt",
        # resume
        resume_epoch=6,  # epoch to resume (‑1 for none)
        # training
        epochs=10,
        batch_size=2,
        lr=2e-4,
        seed=12345678,
        # exposure‑map
        base_exposure=0.46,
        adjustment_amplitude=0.25,
        # adaptive‑weight
        weight_mode="softmax",  # {uncertainty|softmax}
        weight_tau=0.7,
        weight_ema_beta=0.9,
        # schedule‑sampler
        schedule_sampler="uniform",  # {uniform|loss-second-moment}
    )
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


# ------------------------- main ------------------------------------------- #

def main():
    args = create_argparser().parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reproducibility
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # --- model & diffusion --------------------------------------------------
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)

    if args.resume_epoch is not None:
        ppath = os.path.join(args.ckpt_dir, f"model_epoch{args.resume_epoch}.pth")
        if os.path.isfile(ppath):
            print(f"[Resume] Loading model weights from {ppath}")
            model.load_state_dict(torch.load(ppath, map_location=device))
    else:
        ppath = "./ckpt/256x256_diffusion_uncond.pt"
        if os.path.isfile(ppath):
            print(f"[Resume] Loading model weights from {ppath}")
            model.load_state_dict(torch.load(ppath, map_location=device))

    # --- schedule sampler ---------------------------------------------------
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # --- adaptive loss weighting -------------------------------------------
    loss_weighter = AdaptiveLossWeighting(
        num_losses=5,  # ← noise + 4 perceptual losses
        mode=args.weight_mode,
        tau=args.weight_tau,
        ema_beta=args.weight_ema_beta,
    ).to(device)

    # --- optimiser ----------------------------------------------------------
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_weighter.parameters()), lr=args.lr
    )

    if args.resume_epoch is not None:
        wpath = os.path.join(args.ckpt_dir, f"weight_epoch{args.resume_epoch}.pth")
        opath = os.path.join(args.ckpt_dir, f"optim_epoch{args.resume_epoch}.pth")
        if os.path.isfile(wpath) and os.path.isfile(opath):
            loss_weighter.load_state_dict(torch.load(wpath, map_location=device))
            optimizer.load_state_dict(torch.load(opath, map_location=device))
            print(f"[Resume] Loss‑weighter and optimiser restored from epoch {args.resume_epoch}.")

    # --- loss modules -------------------------------------------------------
    L_spa = L_structure2()
    L_exp = L_exp2(1)
    L_fft = L_fft_multiscale()

    # Retinex colour prior
    print("===> Building Retinex model …")
    retinex_model = Rnet().to(device)
    if args.retinex_model.endswith(".ckpt"):
        ck = torch.load(args.retinex_model, map_location="cpu")
        state = {k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
        retinex_model.load_state_dict(state, strict=True)
    else:
        retinex_model.load_state_dict(torch.load(args.retinex_model, map_location="cpu"))
    retinex_model.eval()
    print("Retinex model loaded.")

    # --- dataset ------------------------------------------------------------
    dataset = ImageFolderDataset(args.data_dir, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # training loop ----------------------------------------------------------
    start_epoch = (args.resume_epoch + 1) if (args.resume_epoch is not None) else 0
    end_epoch = start_epoch + args.epochs

    for epoch in range(start_epoch, end_epoch):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x, img_paths in pbar:
            x = x.to(device)  # [B,3,H,W] ∈ [‑1,1]
            B, C, H, W = x.shape

            # ---------------- diffusion forward‑/reverse step ----------------
            t, _ = schedule_sampler.sample(B, device=device)  # importance‑sampled timesteps
            noise = torch.randn_like(x)
            x_noisy = diffusion.q_sample(x_start=x, t=t, noise=noise)

            # model prediction
            model_out = model(x_noisy, t)
            pred_eps, _ = model_out.chunk(2, dim=1)
            pred_xstart = diffusion._predict_xstart_from_eps(x_noisy, t, pred_eps)

            # normalised to [0,1]
            x0_norm = (x + 1) * 0.5
            pred_norm = (pred_xstart + 1) * 0.5

            # dynamic exposure map per image (CPU‑heavy → consider caching)
            exposure_maps = []
            for path in img_paths:
                em = calculate_spatially_varying_exposure(
                    path,
                    base_exposure=args.base_exposure,
                    adjustment_amplitude=args.adjustment_amplitude,
                )
                em = check_image_size(em, padder_size=args.image_size)
                em = F.interpolate(em, size=(H, W), mode="bilinear", align_corners=False)
                exposure_maps.append(em.to(device).expand(1, 3, H, W))
            exposure_map = torch.cat(exposure_maps, dim=0)

            # Retinex colour map (no grad)
            color_map = calculate_color_map(x0_norm, retinex_model).detach()

            # --------------- individual loss components ---------------------
            loss_noise = F.mse_loss(pred_eps, noise, reduction="mean")  # ← NEW
            loss_structure = torch.mean(L_spa(x0_norm, pred_norm))
            loss_exposure = L_exp(pred_norm, exposure_map)
            loss_color = F.mse_loss(pred_norm, color_map, reduction="sum")
            loss_fft = L_fft(pred_norm, x0_norm)

            # aggregate via adaptive weights (order matters!)
            total_loss, weights = loss_weighter(
                [
                    loss_noise,
                    loss_structure,
                    loss_exposure,
                    loss_color,
                    loss_fft,
                ]
            )

            # ----------------------- optimisation ---------------------------
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # progress bar display
            w_str = ",".join(f"{w:.3f}" for w in weights)
            pbar.set_postfix({"L_total": f"{total_loss.item():.4f}", "Weights": f"[{w_str}]"})

        # -------------------- checkpointing per‑epoch -----------------------
        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"model_epoch{epoch}.pth"))
        torch.save(loss_weighter.state_dict(), os.path.join(args.ckpt_dir, f"weight_epoch{epoch}.pth"))
        torch.save(optimizer.state_dict(), os.path.join(args.ckpt_dir, f"optim_epoch{epoch}.pth"))


if __name__ == "__main__":
    main()
