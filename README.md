<div align="center">

<!-- <div class="logo">
    <img src="assets/aglldiff_logo.png" style="width:180px">
</div> -->

<h1>AGLLDiff: Guiding Diffusion Models Towards Unsupervised Training-free Real-world Low-light Image Enhancement</h1>

<div>
    <a href='https://lyl1015.github.io/' target='_blank'>Yunlong Lin</a><sup>1*</sup>&emsp;
    <a href='https://owen718.github.io/' target='_blank'>Tian Ye</a><sup>2*</sup>&emsp;
    <a href='https://ephemeral182.github.io/' target='_blank'>Sixiang Chen</a><sup>2*</sup>&emsp;
    <a href='https://zhenqifu.github.io/' target='_blank'>Zhenqi Fu</a><sup>4</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=fDVgLA0AAAAJ&hl=en' target='_blank'>Yingying Wang</a><sup>1</sup>&emsp;
    <a href='https://rese1f.github.io/' target='_blank'>Wenhao Chai</a><sup>5</sup>&emsp;
    <a href='https://ge-xing.github.io/' target='_blank'>Zhaohu Xing</a><sup>2</sup>&emsp;
    <a href='https://sites.google.com/site/indexlzhu/home/' target='_blank'>Lei Zhu</a><sup>2,3</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=k5hVBfMAAAAJ&hl=zh-CN/' target='_blank'>Xinghao Ding</a><sup>1</sup>
</div>
<div>
    <sup>1</sup>Xiamen University, China&emsp; 
    <sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou), China&emsp; 
    <sup>3</sup>The Hong Kong University of Science and Technology, Hong Kong SAR, China&emsp; 
    <sup>4</sup>Tsinghua University, China&emsp; 
    <sup>5</sup>University of Washington
</div>
<div>
    <em>*denotes equal contribution</em>
</div>

<div>
    :triangular_flag_on_post: <strong>Accepted to AAAI 2025</strong>
</div>

<div>
    <h4 align="center">
        • <a href="https://arxiv.org/pdf/2407.14900" target='_blank'>[arXiv]</a> • 
        <a href="https://aglldiff.github.io/" target='_blank'>[Project Page]</a> •
    </h4>
</div>

<img src="assets/teaser.png" width="800px"/>

<strong>AGLLDiff provides a training-free framework for enhancing low-light images using diffusion models.</strong>

<div>
    If you find AGLLDiff useful for your projects, please consider ⭐ this repo. Thank you! 😉
</div>



---

</div>

## :postbox: Updates
<!-- - 2023.12.04: Add an option to speed up the inference process by adjusting the number of denoising steps. -->
- 2024.2.9: Release our demo codes and models. Have fun! :yum:
- 2023.12.31: This repo is created.

## :diamonds: Installation

### Codes and Environment

```
# git clone this repository
git clone https://github.com/LYL1015/AGLLDiff.git
cd AGLLDiff

# create new anaconda env
conda create -n aglldiff python=3.8 -y
conda activate aglldiff

# install python dependencies
conda install mpi4py
pip3 install -r requirements.txt
pip install -e .
```

### Pretrained Model
Download the pretrained diffusion model from [guided-diffusion](https://github.com/openai/guided-diffusion?tab=readme-ov-file) and the pretrained Rnet model from [Google Drive](https://drive.google.com/file/d/1PCJX_6j3NIqmDHy55P3yAcX9ze1EVRwJ/view?usp=sharing). Place both models in the `ckpt` folder.
```
mkdir ckpt
cd ckpt
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
cd ..
```

## :circus_tent: Inference
### Example usage:
```
python inference_aglldiff.py --task LIE --in_dir ./examples/ --out_dir ./results/
```
There are other arguments you may want to change. You can change the hyperparameters using the command line.

For example, you can use the following command to run inference with customized settings:


```
python inference_aglldiff.py \
--in_dir ./examples/ \
--out_dir ./results/ \
--model_path "./ckpt/256x256_diffusion_uncond.pt" \
--retinex_model "./ckpt/RNet_1688_step.ckpt" \
--guidance_scale 2.3 \
--structure_weight 10 \
--color_map_weight 0.03 \
--exposure_weight 1000 \
--base_exposure 0.46 \
--adjustment_amplitude 0.25 \
--N 2 
```

Explanation of important arguments:
- `in_dir`: Path to the folder containing input images.
- `out_dir`: Path to the folder where results will be saved.
- `model_path`: Path to the pretrained diffusion model checkpoint.
- `retinex_model`: Path to the pretrained Retinex model checkpoint.
- `guidance_scale`: Overall guidance scale for attribute control.
- `structure_weight`: Weight for structure preservation.
- `color_map_weight`: Weight for color mapping guidance.
- `exposure_weight`: Weight for exposure adjustment.
- `base_exposure`: Base exposure value for image enhancement.
- `adjustment_amplitude`: Amplitude of contrast adjustment.
- `N`: Number of gradient descent steps at each timestep.


## :love_you_gesture: Citation
If you find our work useful for your research, please consider citing the paper:
```
@misc{lin2024aglldiff,
  Author = {Yunlong Lin and Tian Ye and Sixiang Chen and Zhenqi Fu and Yingying Wang and Wenhao Chai and Zhaohu Xing and Lei Zhu and Xinghao Ding},
  Title  = {AGLLDiff: Guiding Diffusion Models Towards Unsupervised Training-free Real-world Low-light Image Enhancement},
  year      ={2024}, 
  eprint    ={2407.14900}, 
  archivePrefix={arXiv}, 
  primaryClass={cs.CV},
}
```

### Contact
If you have any questions, please feel free to reach out at `linyl@stu.xmu.edu.cn`. 