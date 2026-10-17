import torch
import torch.nn as nn
from einops import rearrange

class MGAM(nn.Module):
    """
    Multiscale Gaussian Attention Mechanism (MGAM)
    优化版: 向量化提速 + 解决 YOLO 验证阶段 FP16 数据类型冲突
    """
    def __init__(self):
        super(MGAM, self).__init__()
        
        # 1. Multiscale Feature Extraction Module (MFEM)
        self.pool_1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.pool_3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_5 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.pool_7 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

    # ------------------ 提速重构的注意力计算核心逻辑 ------------------
    
    def normal(self, x):
        x_min = torch.min(x)
        x_max = torch.max(x)
        # 添加 1e-7 防止在 FP16 半精度下除以 0 导致 NaN
        x = (x - x_min) / (x_max - x_min + 1e-7)
        return x

    def g_channel(self, x):
        b, c, w, h = x.shape
        
        x_g = x.mean(dim=(2, 3), keepdim=True) + x.amax(dim=(2, 3), keepdim=True)
        
        x_g_flat = x_g.view(b, c)
        _, index = torch.topk(x_g_flat, 1, dim=1)
        
        # 强制在 FP32 下计算分布 (防止底层不支持 FP16)
        mean = index.float() 
        covariance = (c / 2) ** 2 
        
        gaussian_distribution = torch.distributions.Normal(mean, covariance)
        x_g_index = torch.linspace(0, c, c, device=x.device, dtype=torch.float32).unsqueeze(0)
        
        z_g = gaussian_distribution.log_prob(x_g_index).exp()
        z_g_out = z_g.view(b, c, 1, 1)
        
        # 修复关键：与 x 相乘前，将数据类型对齐回 x 的类型 (如 FP16)
        z_g_out = self.normal(z_g_out).to(x.dtype)
        return x * z_g_out

    def g_spatial(self, x):
        b, c, w, h = x.shape
        x_g = torch.mean(x, dim=1, keepdim=True) + torch.max(x, dim=1, keepdim=True)[0]
        
        x_g_flat = x_g.view(b, w * h)
        _, index = torch.topk(x_g_flat, 1, dim=1)
        
        r_index = torch.div(index, w, rounding_mode='trunc').float() 
        c_index = (index % w).float() 
        
        mean = torch.cat([c_index, r_index], dim=1)
        
        # 强制在 FP32 下计算多元分布
        covariance = torch.tensor([[1.0 * (w / 2) ** 2, 0.0], 
                                   [0.0, 1.0 * (h / 2) ** 2]], device=x.device, dtype=torch.float32)
        covariance = covariance.unsqueeze(0).expand(b, 2, 2)
        
        gaussian_distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance)
        
        x_g_index = torch.linspace(0, w, w, device=x.device, dtype=torch.float32)
        y_g_index = torch.linspace(0, h, h, device=x.device, dtype=torch.float32)
        x_g_index, y_g_index = torch.meshgrid(x_g_index, y_g_index, indexing='ij')
        
        x_y_g_index = torch.stack([x_g_index.flatten(), y_g_index.flatten()], dim=1)
        x_y_g_index = x_y_g_index.unsqueeze(1) 
        
        z_g = gaussian_distribution.log_prob(x_y_g_index) 
        
        z_g_out = z_g.transpose(0, 1).exp().view(b, 1, w, h)
        
        # 修复关键：与 x 相乘前，将数据类型对齐回 x 的类型
        z_g_out = self.normal(z_g_out).to(x.dtype)
        return x * z_g_out

    def get_heat_map(self, x, b):
        x = torch.mean(x, dim=1, keepdim=True)
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        # 添加 1e-7 防止在 FP16 下出现 0 除
        x_out = (x - x_min) / (x_max - x_min + 1e-7)
        return x_out

    def com_loss_by_heat(self, x, y):
        b, c, w, h = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x_heat = self.get_heat_map(x, b)
        y_heat = self.get_heat_map(y, b)
        loss = torch.sum(torch.abs(x_heat - y_heat), dim=(2, 3)) / (w * h)
        loss = loss.view(b, 1, 1, 1) 
        return loss

    # ------------------ 前向传播逻辑封装 ------------------
    def forward(self, x):
        """Forward function for MGAM Module"""
        # 1. 提取多尺度特征 (MFEM)
        x_1 = self.pool_1(x)
        x_3 = self.pool_3(x)
        x_5 = self.pool_5(x)
        x_7 = self.pool_7(x)

        # 2. Gaussian Attention Module (GAM)
        c_1 = self.g_channel(x_1)
        s_1 = self.g_spatial(x_1)
        
        c_3 = self.g_channel(x_3)
        s_3 = self.g_spatial(x_3)
        
        c_5 = self.g_channel(x_5)
        s_5 = self.g_spatial(x_5)
        
        c_7 = self.g_channel(x_7)
        s_7 = self.g_spatial(x_7)

        # 3. Dynamic Weighting Module (DWM)
        sim_c_1 = self.com_loss_by_heat(x, c_1)
        sim_c_3 = self.com_loss_by_heat(x, c_3)
        sim_c_5 = self.com_loss_by_heat(x, c_5)
        sim_c_7 = self.com_loss_by_heat(x, c_7)
        
        sim_s_1 = self.com_loss_by_heat(x, s_1)
        sim_s_3 = self.com_loss_by_heat(x, s_3)
        sim_s_5 = self.com_loss_by_heat(x, s_5)
        sim_s_7 = self.com_loss_by_heat(x, s_7)

        sum_sim_c = sim_c_1 + sim_c_3 + sim_c_5 + sim_c_7
        sum_sim_s = sim_s_1 + sim_s_3 + sim_s_5 + sim_s_7

        out = (x + (sim_c_1 / sum_sim_c) * c_1 + (sim_c_3 / sum_sim_c) * c_3 +
             (sim_c_5 / sum_sim_c) * c_5 + (sim_c_7 / sum_sim_c) * c_7 +
             (sim_s_1 / sum_sim_s) * s_1 + (sim_s_3 / sum_sim_s) * s_3 +
             (sim_s_5 / sum_sim_s) * s_5 + (sim_s_7 / sum_sim_s) * s_7)
             
        return out


class RFMGAMConv(nn.Module):
    """
    基于 RFAConv 思想重构的 RFMGAMConv
    结合 Receptive-Field Spatial Feature 提取与 MGAM 高斯注意力特征增强
    """
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super(RFMGAMConv, self).__init__()
        if kernel_size % 2 == 0:
            assert False, "the kernel_size must be odd."
            
        self.kernel_size = kernel_size
        
        # 1. 采用分组卷积生成感受野空间特征 (Receptive-Field Spatial Feature)
        self.generate = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size**2), kernel_size, 
                      padding=kernel_size//2, stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size**2)),
            nn.ReLU()
        )
        
        # 2. 核心：MGAM 注意力机制
        self.mgam = MGAM()

        # 3. 参数不共享卷积计算特征融合 (解决感受野内部的参数共享问题)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        
    def forward(self, x):
        b, c = x.shape[0:2]
        
        # 第一步：提取初始感受野特征
        generate_feature = self.generate(x)
        h, w = generate_feature.shape[2:]
        
        # 第二步：将通道维度中隐藏的感受野局部信息 unfold (展开) 到空间维度上
        generate_feature = generate_feature.view(b, c, self.kernel_size**2, h, w)
        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', 
                                     n1=self.kernel_size, n2=self.kernel_size)
        
        # 第三步：在展开的不重叠空间特征上应用 MGAM 计算多尺度高斯注意力
        attention_feature = self.mgam(generate_feature)
        
        # 第四步：使用 kernel_size=k, stride=k 的卷积操作还原空间尺寸并融合多通道信息
        out = self.conv(attention_feature)
        
        return out