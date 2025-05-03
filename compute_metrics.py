import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# —— 用户配置 —— #
# 硬编码文件夹路径
FOLDER_BASE = "./results/GT"
FOLDER_COMPARE_1 = "./results/base10"
FOLDER_COMPARE_2 = "./results/base5"

# 选择设备：优先 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 设置随机种子
torch.manual_seed(123)

# —— 指标初始化 —— #
psnr_metric = PeakSignalNoiseRatio().to(DEVICE)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(DEVICE)

def load_image(path: str) -> torch.Tensor:
    """
    读取图像并转为 [1,3,H,W] 的 FloatTensor (0~1)，直接加载到 DEVICE 上。
    """
    img = Image.open(path).convert('RGB')
    tensor = ToTensor()(img).unsqueeze(0).to(DEVICE)
    return tensor

def compute_metrics(img1: torch.Tensor, img2: torch.Tensor) -> dict:
    """
    在 DEVICE 上计算 PSNR（公式 & Metric）、SSIM、LPIPS。
    """
    with torch.no_grad():
        mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1)
        psnr_formula = 20 * torch.log10(1.0 / torch.sqrt(mse))

        psnr_m = psnr_metric(img1, img2)
        ssim_m = ssim_metric(img1, img2)
        lpips_m = lpips_metric(img1, img2)

    return {
        'psnr_metric':  psnr_m.item(),
        'ssim':         ssim_m.item(),
        'lpips':        lpips_m.item()
    }

def compare_folders(base_folder, compare_folder, label=""):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    base_files = sorted(f for f in os.listdir(base_folder) if f.lower().endswith(exts))
    comp_files = sorted(f for f in os.listdir(compare_folder) if f.lower().endswith(exts))

    if len(base_files) != len(comp_files):
        print(f"❌ 文件数不一致：{len(base_files)} vs {len(comp_files)}")
        return

    n = len(base_files)
    sums = {'psnr_metric': 0.0, 'ssim': 0.0, 'lpips': 0.0}

    for f1, f2 in zip(base_files, comp_files):
        img_base = load_image(os.path.join(base_folder, f1))
        img_comp = load_image(os.path.join(compare_folder, f2))
        m = compute_metrics(img_base, img_comp)
        for k in sums:
            sums[k] += m[k]

    avgs = {k: sums[k] / n for k in sums}
    print(f"\n=== 与 {label} 对比的平均指标（共 {n} 对图像） ===")
    print(f"PSNR(metric) = {avgs['psnr_metric']:.4f} dB")
    print(f"SSIM          = {avgs['ssim']:.4f}")
    print(f"LPIPS         = {avgs['lpips']:.4f}")

def main():
    compare_folders(FOLDER_BASE, FOLDER_COMPARE_1, label="origin")
    compare_folders(FOLDER_BASE, FOLDER_COMPARE_2, label="opt")

if __name__ == "__main__":
    main()
