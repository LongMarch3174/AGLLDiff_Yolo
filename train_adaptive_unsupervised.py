#!/usr/bin/env python
# train_adaptive_unsupervised.py

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
from tqdm import tqdm
import collections

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from utils.Attribute import (
    L_structure2,
    L_exp2,
    calculate_spatially_varying_exposure,
    calculate_color_map,
    check_image_size,                     # ← 新增
    L_fft_multiscale,
    AdaptiveLossWeighting
)
from Rnet import net as Rnet


class ImageFolderDataset(Dataset):
    """Simple image folder dataset returning (tensor, path)."""
    def __init__(self, folder, image_size):
        self.paths = sorted(glob(os.path.join(folder, '*')))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),            # [0,1]
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # to [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        img_t = self.transform(img)
        return img_t, path


def create_argparser():
    defaults = dict(
        # dataset & I/O
        data_dir='./LOLdataset/our485/low',
        ckpt_dir='./ckpt',
        retinex_model='./ckpt/RNet_1688_step.ckpt',
        # training hyperparams
        pretrained_model='./ckpt/256x256_diffusion_uncond.pt',
        epochs=5,
        batch_size=4,
        lr=2e-4,
        seed=12345678,
        # exposure map params
        base_exposure=0.46,
        adjustment_amplitude=0.25,
        # adaptive weight params
        weight_mode='uncertainty',  # or 'softmax'
        weight_tau=1.0,
        weight_ema_beta=0.9,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    parser = create_argparser()
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # model & diffusion
    md_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**md_kwargs)
    model.to(device)
    if args.pretrained_model:
        print(f"Loading pretrained model from {args.pretrained_model}")
        ckpt = torch.load(args.pretrained_model, map_location='cpu')
        model.load_state_dict(ckpt)
        print("Pretrained weights loaded.")

    # adaptive loss weight
    loss_weighter = AdaptiveLossWeighting(
        num_losses=4,
        mode=args.weight_mode,
        tau=args.weight_tau,
        ema_beta=args.weight_ema_beta
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_weighter.parameters()),
        lr=args.lr
    )

    # loss modules
    L_spa = L_structure2()
    L_exp = L_exp2(1)
    L_fft = L_fft_multiscale()
    # pretrained Retinex
    print('===> Building retinex model')
    retinex_model = Rnet().to(device)
    if args.retinex_model.endswith('.ckpt'):
        print("Loading Rnet checkpoint")
        ck = torch.load(args.retinex_model, map_location='cpu')
        new_sd = collections.OrderedDict()
        for k, v in ck['state_dict'].items():
            if k.startswith('model.'):
                new_sd[k[6:]] = v
        retinex_model.load_state_dict(new_sd, strict=True)
    else:
        retinex_model.load_state_dict(torch.load(args.retinex_model, map_location='cpu'))
    print('Pre-trained retinex model is loaded.')
    retinex_model.eval()

    # dataset & loader
    dataset = ImageFolderDataset(args.data_dir, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x, img_paths in pbar:
            x = x.to(device)               # [B,3,256,256]
            B, C, H, W = x.shape

            # sample timesteps & noise
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device).long()
            noise = torch.randn_like(x)
            x_noisy = diffusion.q_sample(x_start=x, t=t, noise=noise)

            # model prediction & x0 estimate
            model_out = model(x_noisy, t)
            pred_eps, _ = model_out.chunk(2, dim=1)
            pred_xstart = diffusion._predict_xstart_from_eps(x_noisy, t, pred_eps)

            # normalize to [0,1]
            x0_norm   = (x + 1) * 0.5
            pred_norm = (pred_xstart + 1) * 0.5

            # === dynamic exposure map per image ===
            exposure_maps = []
            for path in img_paths:
                em = calculate_spatially_varying_exposure(
                    path,
                    base_exposure=args.base_exposure,
                    adjustment_amplitude=args.adjustment_amplitude
                )                              # [1,1,h_raw,w_raw]
                # pad 到能被 image_size 整除
                em = check_image_size(em, padder_size=args.image_size)
                # 再 resize 到 [1,1, H, W]
                em = F.interpolate(em, size=(H, W),
                                   mode='bilinear', align_corners=False)
                # expand 到 3 通道 [1,3,H,W]
                em = em.to(device).expand(1, 3, H, W)
                exposure_maps.append(em)
            exposure_map = torch.cat(exposure_maps, dim=0)  # [B,3,H,W]

            # Retinex color map
            color_map = calculate_color_map(x0_norm, retinex_model).detach()  # [B,3,H,W]

            # compute individual losses
            loss_structure = torch.mean(L_spa(x0_norm, pred_norm))
            loss_exposure  = L_exp(pred_norm, exposure_map)
            loss_color     = F.mse_loss(pred_norm, color_map, reduction='sum')
            loss_fft       = L_fft(pred_norm, x0_norm)

            # aggregate with adaptive weights
            total_loss, weights = loss_weighter([
                loss_structure, loss_exposure, loss_color, loss_fft
            ])

            # backward & step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # update progress bar
            w_str = ','.join(f"{w:.3f}" for w in weights)
            pbar.set_postfix({
                'L_total': f"{total_loss.item():.4f}",
                'Weights': f"[{w_str}]"
            })

        # save checkpoints
        torch.save(model.state_dict(),
                   os.path.join(args.ckpt_dir, f"model_epoch{epoch}.pth"))
        torch.save(loss_weighter.state_dict(),
                   os.path.join(args.ckpt_dir, f"weight_epoch{epoch}.pth"))
        torch.save(optimizer.state_dict(),
                   os.path.join(args.ckpt_dir, f"optim_epoch{epoch}.pth"))


if __name__ == '__main__':
    main()
