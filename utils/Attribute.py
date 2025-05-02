import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


def check_image_size(x, padder_size=256):
    _, _, h, w = x.size()
    mod_pad_h = (padder_size - h % padder_size) % padder_size
    mod_pad_w = (padder_size - w % padder_size) % padder_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
    return x


def normalize_data(data):
    """Normalize the data to the range [-1, 1]."""
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    normalized_data = 2 * normalized_data - 1
    return normalized_data


def normalize_data_torch(data):
    """Normalize the data to the range [0, 1]."""
    min_val = torch.min(data)
    max_val = torch.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def calculate_spatially_varying_exposure(image_path, base_exposure=0.55, adjustment_amplitude=0.15):
    img = Image.open(image_path)
    ycbcr_img = img.convert('YCbCr')
    rgb_img = img.convert('RGB')

    y, _, _ = ycbcr_img.split()
    l_channel = np.array(y) / 255
    l_channl_rgb = np.array(rgb_img) / 255
    l_avg = np.mean(l_channel)
    
    norm_diff = normalize_data(l_avg - l_channel)
    exposure_map = base_exposure + adjustment_amplitude * norm_diff
    exposure_map = exposure_map.astype(np.float32)[:, :, np.newaxis]
    exposure_map = torch.tensor(exposure_map).permute(2, 0, 1).unsqueeze(0).cuda()
    
    return exposure_map


def calculate_color_map(input, Retinex):
    L, color_map = Retinex(input)
    return color_map


def calculate_color_map_fix(input, Retinex):
    input = torch.pow(input, 0.25)
    data_low = input.squeeze(0) / 20

    data_max_r = data_low[0].max()
    data_max_g = data_low[1].max()
    data_max_b = data_low[2].max()
    
    color_max = torch.zeros((data_low.shape[0], data_low.shape[1], data_low.shape[2])).cuda()
    color_max[0, :, :] = data_max_r * torch.ones((data_low.shape[1], data_low.shape[2])).cuda()
    color_max[1, :, :] = data_max_g * torch.ones((data_low.shape[1], data_low.shape[2])).cuda()
    color_max[2, :, :] = data_max_b * torch.ones((data_low.shape[1], data_low.shape[2])).cuda()
    
    data_color = data_low / (color_max + 1e-6)
    return data_color.unsqueeze(0)


class L_structure(nn.Module):
    def __init__(self):
        super(L_structure, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, org, enhance):
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        D_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_left - D_enhance_left, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        
        return D_left + D_right + D_up + D_down


class L_exp2(nn.Module):
    def __init__(self, patch_size):
        super(L_exp2, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x, y):
        x = torch.mean(x, 1, keepdim=True)
        mean_x = self.pool(x)
        mean_y = self.pool(y)
        d = torch.mean(torch.pow(mean_x - mean_y, 2))
        return d

class L_structure2(nn.Module):
    def __init__(self):
        super(L_structure2, self).__init__()

    def forward(self, input, target):
        H,W = input.shape[-2:]
        x_fft = torch.fft.rfft2(input+1e-8, norm='backward')
        x_amp = torch.abs(x_fft)
        x_pha = torch.angle(x_fft)
        real_uni = 1 * torch.cos(x_pha)+1e-8
        imag_uni = 1 * torch.sin(x_pha)+1e-8
        x_uni = torch.complex(real_uni, imag_uni)+1e-8
        x_uni = torch.abs(torch.fft.irfft2(x_uni, s=(H, W), norm='backward'))
        x_g = torch.gradient(x_uni,axis=(2,3),edge_order=2)
        x_g_x  = x_g[0];x_g_y = x_g[1]
        
        y_fft = torch.fft.rfft2(target+1e-8, norm='backward')
        y_amp = torch.abs(y_fft)
        y_pha = torch.angle(y_fft)
        real_uni = 1 * torch.cos(y_pha)+1e-8
        imag_uni = 1 * torch.sin(y_pha)+1e-8
        y_uni = torch.complex(real_uni, imag_uni)+1e-8
        y_uni = torch.abs(torch.fft.irfft2(y_uni, s=(H, W), norm='backward'))
        y_g = torch.gradient(y_uni,axis=(2,3),edge_order=2)
        y_g_x  = y_g[0];y_g_y =y_g[1]
        
        D_left = torch.pow(x_g_x - y_g_x,2)
        D_right = torch.pow(x_g_y - y_g_y,2)
        
        E = (D_left + D_right)
        
        return E


class L_fft_multiscale(nn.Module):
    def __init__(self):
        super(L_fft_multiscale, self).__init__()

    def fft_amp(self, img):
        fft = torch.fft.fft2(img, norm='ortho')
        return torch.abs(fft)

    def forward(self, input, target):
        loss = 0
        for scale in [1.0, 0.5, 0.25]:
            if scale != 1.0:
                input_scaled = F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                input_scaled, target_scaled = input, target

            input_amp = self.fft_amp(input_scaled)
            target_amp = self.fft_amp(target_scaled)
            loss += F.mse_loss(input_amp, target_amp)

        return loss


class AdaptiveLossWeighting(nn.Module):
    """
    自适应多任务 loss 加权器

    Args:
        num_losses (int): loss 项数量
        mode        (str): 'softmax' or 'uncertainty'
        tau       (float): softmax 温度 (仅 softmax 模式有效)
        ema_beta  (float): EMA 系数 (0=不用 EMA)
    """
    def __init__(self, num_losses: int, mode: str = "softmax",
                 tau: float = 1.0, ema_beta: float = 0.0):
        super().__init__()
        assert mode in ("softmax", "uncertainty")
        self.mode = mode
        self.tau  = tau
        self.ema_beta = ema_beta
        # learnable params
        if mode == "softmax":
            self.logits = nn.Parameter(torch.zeros(num_losses))  # w_i = softmax(logits/τ)
        else:  # uncertainty
            self.log_vars = nn.Parameter(torch.zeros(num_losses))  # s_i = log σ_i^2

        # for EMA
        if ema_beta > 0:
            self.register_buffer("ema_loss", torch.zeros(num_losses), persistent=False)

    def forward(self, loss_list):
        """
        输入: list[Tensor] 各子 loss (标量或 batch 平均)
        输出: total_loss, weight_tensor
        """
        losses = torch.stack(loss_list)   # [K]
        if self.ema_beta > 0:
            # 更新 loss 的滑动均值 (不参与梯度)
            with torch.no_grad():
                self.ema_loss.mul_(self.ema_beta).add_(losses.detach() * (1 - self.ema_beta))
            norm_losses = losses / (self.ema_loss + 1e-8)
        else:
            norm_losses = losses

        if self.mode == "softmax":
            weights = torch.softmax(self.logits / self.tau, dim=0)
            total   = torch.dot(weights, norm_losses)
        else:  # uncertainty weighting
            inv_sigma2 = torch.exp(-self.log_vars)          # exp(-s_i)
            total = 0.5 * torch.sum(inv_sigma2 * losses + self.log_vars)  # Σ ½ e^{-s} L + ½ s
            weights = inv_sigma2 / inv_sigma2.sum()         # 仅用于监控

        return total, weights.detach()  # 返回权重的 detach 版方便打印


class L_fft(nn.Module):
    def __init__(self):
        super(L_fft, self).__init__()

    def forward(self, input, target):
        """
        计算输入图像和目标图像在频域（幅度谱）上的均方误差。
        """
        _, _, H, W = input.shape
        input_fft = torch.fft.fft2(input, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')

        input_amp = torch.abs(input_fft)
        target_amp = torch.abs(target_fft)

        loss = F.mse_loss(input_amp, target_amp)
        return loss


