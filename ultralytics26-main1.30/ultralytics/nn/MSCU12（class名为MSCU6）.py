import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAwareMultiScaleExtractor(nn.Module):
    """
    轻量化 Scale-aware 多尺度特征提取器 (剔除 SE 平均池化毒药版)
    利用纯粹的感受野金字塔 + 像素级动态空间融合，死保小目标空间特征
    """
    def __init__(self, channels):
        super().__init__()
        
        # 分支1: 感受野 3x3 (rate=1) -> 关注极小目标和高频细节
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels, bias=False),
            # [已移除 SE Block，防止全局平均池化稀释小目标]
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), # PWConv 跨通道交互
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # 分支2: 感受野 5x5 (rate=3) -> 关注中等目标
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # 分支3: 感受野 7x7 (rate=5) -> 关注大目标和广阔上下文
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # Scale-aware 像素级空间注意力生成器 (核心保留：这是安全的，因为它不破坏空间分辨率)
        self.attention_generator = nn.Sequential(
            # 将 1x1 卷积替换为 3x3 进一步平滑局部注意力响应
            nn.Conv2d(channels, 3, kernel_size=3, padding=1, bias=False),
            nn.Softmax(dim=1) # 在3个尺度(通道维度)进行归一化
        )

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        
        f_sum = f1 + f2 + f3
        
        # 生成动态尺度权重 (B, 3, H, W)，完全保留空间分辨率
        attn_weights = self.attention_generator(f_sum)
        
        # 权重切分
        alpha = attn_weights[:, 0:1, :, :]
        beta  = attn_weights[:, 1:2, :, :]
        gamma = attn_weights[:, 2:3, :, :]
        
        # 像素级自适应尺度融合，精准叠加
        f_fusion = alpha * f1 + beta * f2 + gamma * f3
        
        return f_fusion


class MSCU6(nn.Module):
    """
    Hybrid MSCU + SCPP (YOLO26 冲分版)
    先用 SCPP 的多尺度自适应提取丰富上下文，再用 MSCU 的通道位移捕捉高频边缘。
    """
    def __init__(self, c1, scale=2, groups=4):
        super().__init__()
        self.scale = scale
        self.groups = groups 
        
        self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')
        
        # 【核心替换】：用 SCPP 的多尺度感知模块替换原本死板的单层 3x3 DWConv
        self.multiscale_extractor = ScaleAwareMultiScaleExtractor(c1)
        
        # D_pw: 1x1 Pointwise Conv (用于位移后的跨通道全局融合)
        self.pw_conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True)
        )

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x

    def shift_channel_mix(self, x):
        b, c, h, w = x.size()
        c_split = c // 4
        x1, x2, x3, x4 = torch.split(x, c_split, dim=1)
        
        x1 = torch.roll(x1, shifts=-1, dims=3)
        x2 = torch.roll(x2, shifts=1, dims=3)
        x3 = torch.roll(x3, shifts=-1, dims=2)
        x4 = torch.roll(x4, shifts=1, dims=2)
        
        return torch.cat([x1, x2, x3, x4], dim=1)

    def forward(self, x):
        # 1. 尺度放大
        x = self.upsample(x)
        
        # 2. 【多尺度上下文自适应提取 (SCPP思想)】
        # 这一步彻底解决了遥感图像中巨大尺度差异和相似背景引发的"空洞"和误检问题
        x = self.multiscale_extractor(x)
        
        # 3. 通道重排，促进组间交互
        x = self.channel_shuffle(x, self.groups)
        
        
        # 5. 全局特征凝练
        x = self.pw_conv(x)
        
        return x