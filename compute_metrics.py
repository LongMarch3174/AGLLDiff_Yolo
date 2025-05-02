import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# —— 用户配置 —— #
# 硬编码文件夹路径
FOLDER1 = "./results/base"
FOLDER2 = "./results/opt"
# 选择设备：优先 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 设置随机种子
torch.manual_seed(123)

# —— 指标初始化 —— #
# 将这些指标对象移动到 GPU（如果可用）
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
        # PSNR 公式计算
        mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1)
        psnr_formula = 20 * torch.log10(1.0 / torch.sqrt(mse))

        # TorchMetrics 计算
        psnr_m = psnr_metric(img1, img2)
        ssim_m = ssim_metric(img1, img2)
        lpips_m = lpips_metric(img1, img2)

    return {
        'psnr_formula': psnr_formula.item(),
        'psnr_metric':  psnr_m.item(),
        'ssim':         ssim_m.item(),
        'lpips':        lpips_m.item()
    }

def main():
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files1 = sorted(f for f in os.listdir(FOLDER1) if f.lower().endswith(exts))
    files2 = sorted(f for f in os.listdir(FOLDER2) if f.lower().endswith(exts))

    if len(files1) != len(files2):
        print(f"❌ 两个文件夹图像数量不一致：{len(files1)} vs {len(files2)}")
        return

    n = len(files1)
    sums = {'psnr_formula': 0.0, 'psnr_metric': 0.0, 'ssim': 0.0, 'lpips': 0.0}

    for f1, f2 in zip(files1, files2):
        img1 = load_image(os.path.join(FOLDER1, f1))
        img2 = load_image(os.path.join(FOLDER2, f2))
        m = compute_metrics(img1, img2)
        for k in sums:
            sums[k] += m[k]

    # 计算并打印平均值
    avgs = {k: sums[k] / n for k in sums}
    print(f"\n=== 平均指标（共 {n} 对图像）===")
    print(f"PSNR(formula) = {avgs['psnr_formula']:.4f} dB")
    print(f"PSNR(metric ) = {avgs['psnr_metric']:.4f} dB")
    print(f"SSIM          = {avgs['ssim']:.4f}")
    print(f"LPIPS         = {avgs['lpips']:.4f}")

if __name__ == "__main__":
    main()
