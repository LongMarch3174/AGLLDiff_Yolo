import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import os.path as osp
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torch
import collections
from guided_diffusion import dist_util, logger
from Rnet import net as Rnet
from utils.Attribute import *
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main(inference_step=None):
    L_spa = L_structure2()
    L_exp = L_exp2(1)
    
    def attribute_guidance(x, t, y=None, pred_xstart=None, target=None, ref=None, mask=None, 
                         task="LIE", scale = 0, N = None, exposure_map = None, reflectence_map=None): 
        assert y is not None
        with th.enable_grad():
            predicted_start = pred_xstart.detach().requires_grad_(True)
            total_loss = 0
            
            print(f'[t={str(t.cpu().numpy()[0]).zfill(3)}]', end=' ')

            # Normalize inputs to [0,1] range
            predicted_start_norm = ((predicted_start + 1) * 0.5)
            target_norm = ((y + 1) * 0.5)
            
            # Calculate component losses
            spatial_structure_loss = torch.mean(L_spa(target_norm, predicted_start_norm)) * args.structure_weight
            illumination_loss = L_exp(predicted_start_norm, exposure_map) * args.exposure_weight
            reflectance_loss = F.mse_loss(reflectence_map, predicted_start_norm, reduction='sum') * args.color_map_weight
            
            # Combine all losses
            total_loss = spatial_structure_loss + illumination_loss + reflectance_loss
            
            # Print loss components
            print(f'loss (structure): {spatial_structure_loss};', end=' ')
            print(f'loss (exposure): {illumination_loss};', end=' ') 
            print(f'loss (color): {reflectance_loss};', end=' ')
            print(f'loss (total): {total_loss};')

            if t.cpu().numpy()[0] > 0:
                print(end='\r')
            else:
                print('\n')
                
            gradient = th.autograd.grad(total_loss, predicted_start)[0]

        return gradient, None

    def model_fn(x, t, y=None, target=None, ref=None, mask=None, task=None, scale=0, N=1,
                 exposure_map=None, reflectence_map=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    
    # Parse args and setup
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = f'{args.out_dir}/s{args.guidance_scale}_sw{args.structure_weight}_cw{args.color_map_weight}_ew{args.exposure_weight}_be{args.base_exposure}_aa{args.adjustment_amplitude}_seed{args.seed}'
    logger.configure(dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(dist_util.dev())
    model.eval()

    # Load retinex model
    print('===> Building retinex model')
    retinex_model = Rnet().to(dist_util.dev())
    
    if args.retinex_model.split('.')[-1] == 'ckpt':
        print("Loading Rnet checkpoint")
        ckpt = torch.load(args.retinex_model, map_location=lambda storage, loc: storage)
        new_state_dict = collections.OrderedDict()
        for k in ckpt['state_dict']:
            if k[:6] != 'model.':
                continue
            name = k[6:]
            new_state_dict[name] = ckpt['state_dict'][k]           
        retinex_model.load_state_dict(new_state_dict, strict=True)
    else:
        retinex_model.load_state_dict(torch.load(args.retinex_model, map_location=lambda storage, loc: storage))
    print('Pre-trained retinex model is loaded.')
    retinex_model.eval()

    # Print sampling summary
    print("=================== Summary (Sampling) ===================")
    print(f'Task: {args.task}; Guidance scale: {args.guidance_scale}')
    print(f'structure weight (w1={args.structure_weight}).')
    print(f'color weight (w2={args.color_map_weight}).')
    print(f'exposure weight (w3={args.exposure_weight}).') 
    print("==========================================================")
    
    # Set random seeds
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    
    # Process images
    all_images = []
    lr_folder = args.in_dir
    lr_images = sorted(os.listdir(lr_folder))

    logger.log("Sampling...")

    for img_name in lr_images:
        # Load and preprocess image
        path_lq = osp.join(lr_folder, img_name)
        raw = cv2.imread(path_lq).astype(np.float32)[:, :, [2, 1, 0]]
        y00 = th.as_tensor(raw/255).permute(2,0,1).unsqueeze(0).to(dist_util.dev())
        y0 = th.tensor(raw/127.5 - 1).permute(2,0,1).unsqueeze(0).to(dist_util.dev())

        print(img_name)
        _,_,H,W = y0.shape
        
        # Setup model kwargs
        model_kwargs = {
            "task": args.task,
            "target": None,
            "scale": args.guidance_scale,
            "N": args.N,
            "exposure_map": check_image_size(calculate_spatially_varying_exposure(path_lq, args.base_exposure, args.adjustment_amplitude)),
            "y": check_image_size(y0),
            "reflectence_map": check_image_size(calculate_color_map(y00, retinex_model))
        }
        b,c,h,w = model_kwargs["y"].shape

        # Sample
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, h, w),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=attribute_guidance,
            device=dist_util.dev(),
            seed=seed,
            inference_step=inference_step
        )

        sample = ((sample[:,:,:H,:W] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} sample")

        cv2.imwrite(f'{out_dir}/{img_name}', all_images[-1][0][...,[2,1,0]])
        torch.cuda.empty_cache()

    dist.barrier()
    logger.log("Sampling complete!")

def create_argparser():
    defaults = dict(
        seed=12345678,
        task='LIE',
        in_dir='./examples',
        out_dir='results',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="./ckpt/256x256_diffusion_uncond.pt",
        retinex_model = "./ckpt/RNet_1688_step.ckpt",
        guidance_scale=2.3,        # Overall guidance scale for attribute guidance
        structure_weight=10,       # Weight for structure preservation
        color_map_weight=0.03,    # Weight for color mapping guidance
        exposure_weight=1000,     # Weight for exposure adjustment guidance
        base_exposure=0.46,       # Base exposure value for image enhancement
        adjustment_amplitude=0.25, # Amplitude of contrast adjustment
        N=2,                      # number of gradient steps at each time t
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main(inference_step=10)
