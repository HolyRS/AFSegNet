from unetmamba_model.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重
class CRF_RNN(nn.Module):
    def __init__(self, num_classes, num_iterations=5, spatial_ker_weight=1.0, bilateral_ker_weight=1.0, theta_alpha=80.0, theta_beta=13.0, theta_gamma=3.0):
        super(CRF_RNN, self).__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations

        # 可学习参数
        self.spatial_ker_weight = nn.Parameter(torch.tensor(spatial_ker_weight, requires_grad=True))
        self.bilateral_ker_weight = nn.Parameter(torch.tensor(bilateral_ker_weight, requires_grad=True))

        # 可调高斯核参数
        self.theta_alpha = nn.Parameter(torch.tensor(theta_alpha, requires_grad=True))  # RGB强度差权重
        self.theta_beta = nn.Parameter(torch.tensor(theta_beta, requires_grad=True))    # 空间差异权重
        self.theta_gamma = nn.Parameter(torch.tensor(theta_gamma, requires_grad=True))  # 空间高斯核的标准差

    def forward(self, unary_logits, image):
        """
        unary_logits: [B, C, H, W] - TransUNet 输出的原始预测（未softmax）
        image:        [B, 3, H, W] - 原始图像
        """
        softmax_unary = F.softmax(unary_logits, dim=1)  # 初始 Q 分布

        B, C, H, W = unary_logits.size()

        # 坐标信息
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=image.device), torch.arange(W, device=image.device), indexing='ij')
        pos = torch.stack([grid_y, grid_x], dim=0).float()  # [2, H, W]
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

        for _ in range(self.num_iterations):
            # 1. 高斯空间核（spatial kernel）
            spatial_out = F.avg_pool2d(softmax_unary, kernel_size=3, stride=1, padding=1)

            # 2. 双边核（bilateral kernel）：基于像素颜色差异 + 空间位置
            bilateral_out = self._bilateral_filter(softmax_unary, image, pos)

            # 3. Message Passing + Compatibility Transform
            pairwise = self.spatial_ker_weight * spatial_out + self.bilateral_ker_weight * bilateral_out

            # 减去 pairwise energy，重新归一化
            softmax_unary = F.softmax(unary_logits - pairwise, dim=1)

        return softmax_unary

    def _bilateral_filter(self, Q, image, pos, win=3):
        B, C, H, W = Q.shape
        pad = win // 2
        Q_pad = F.pad(Q, [pad] * 4, mode='reflect')
        img_pad = F.pad(image, [pad] * 4, mode='reflect')
        pos_pad = F.pad(pos, [pad] * 4, mode='reflect')

        out = torch.zeros_like(Q)

        for i in range(-pad, pad + 1):
            for j in range(-pad, pad + 1):
                shifted_Q = Q_pad[:, :, pad + i:H + pad + i, pad + j:W + pad + j]
                shifted_img = img_pad[:, :, pad + i:H + pad + i, pad + j:W + pad + j]
                shifted_pos = pos_pad[:, :, pad + i:H + pad + i, pad + j:W + pad + j]

                color_diff = ((image - shifted_img) / self.theta_alpha) ** 2
                pos_diff = ((pos - shifted_pos) / self.theta_beta) ** 2
                weight = torch.exp(- (color_diff.sum(1, keepdim=True) + pos_diff.sum(1, keepdim=True)))

                out += weight * shifted_Q
        return out / ((win ** 2) + 1e-8)
from .vit import ViT  # expected in your codebase
# =========================
# Utility modules
# =========================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        mid = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, mid, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class CRF_RNN(nn.Module):
    def __init__(self, num_classes, num_iterations=5, spatial_ker_weight=1.0, bilateral_ker_weight=1.0,
                 theta_alpha=80.0, theta_beta=13.0, theta_gamma=3.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.spatial_ker_weight = nn.Parameter(torch.tensor(spatial_ker_weight, requires_grad=True))
        self.bilateral_ker_weight = nn.Parameter(torch.tensor(bilateral_ker_weight, requires_grad=True))
        self.theta_alpha = nn.Parameter(torch.tensor(theta_alpha, requires_grad=True))
        self.theta_beta = nn.Parameter(torch.tensor(theta_beta, requires_grad=True))
        self.theta_gamma = nn.Parameter(torch.tensor(theta_gamma, requires_grad=True))
    def forward(self, unary_logits, image):
        Q = F.softmax(unary_logits, dim=1)
        B, C, H, W = unary_logits.size()
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=image.device),
                                        torch.arange(W, device=image.device), indexing='ij')
        pos = torch.stack([grid_y, grid_x], dim=0).float().unsqueeze(0).repeat(B, 1, 1, 1)
        for _ in range(self.num_iterations):
            spatial_out = F.avg_pool2d(Q, 3, 1, 1)
            bilateral_out = self._bilateral_filter(Q, image, pos)
            pairwise = self.spatial_ker_weight * spatial_out + self.bilateral_ker_weight * bilateral_out
            Q = F.softmax(unary_logits - pairwise, dim=1)
        return Q
    def _bilateral_filter(self, Q, image, pos, win=3):
        B, C, H, W = Q.shape
        pad = win // 2
        Q_pad   = F.pad(Q,     [pad]*4, mode='reflect')
        img_pad = F.pad(image, [pad]*4, mode='reflect')
        pos_pad = F.pad(pos,   [pad]*4, mode='reflect')
        out = torch.zeros_like(Q)
        for i in range(-pad, pad+1):
            for j in range(-pad, pad+1):
                shifted_Q   = Q_pad[:, :, pad+i:H+pad+i, pad+j:W+pad+j]
                shifted_img = img_pad[:, :, pad+i:H+pad+i, pad+j:W+pad+j]
                shifted_pos = pos_pad[:, :, pad+i:H+pad+i, pad+j:W+pad+j]
                color_diff = ((image - shifted_img) / self.theta_alpha)**2
                pos_diff   = ((pos   - shifted_pos) / self.theta_beta)**2
                weight = torch.exp(- (color_diff.sum(1, keepdim=True) + pos_diff.sum(1, keepdim=True)))
                out += weight * shifted_Q
        return out / ((win**2) + 1e-8)

# =========================
# Basic blocks
# =========================
class StemPatchify(nn.Module):
    """Stem to get H/4×W/4 feature map (ViT-style)."""
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=4, padding=0, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return F.gelu(self.bn(self.proj(x)))

class PatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))

class ConvDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class FuseConcat1x1(nn.Module):
    def __init__(self, c_a, c_b, c_out):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(c_a + c_b, c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )
    def forward(self, a, b):
        return self.fuse(torch.cat([a, b], dim=1))

# =========================
# Branch definitions
# =========================
class VSSStage1_NoDown(nn.Module):
    def __init__(self, ch, depth):
        super().__init__()
        self.blocks = nn.Sequential(*[VSSBlock(hidden_dim=ch, drop_path=0.1) for _ in range(depth)])
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x=self.blocks(x)
        x=x.permute(0,3,1,2)
        return x

class VSSStage_DownThenBlocks(nn.Module):
    def __init__(self, in_ch, out_ch, depth):
        super().__init__()
        self.down = PatchMerging(in_ch, out_ch)
        self.blocks = nn.Sequential(*[VSSBlock(hidden_dim=out_ch, drop_path=0.1) for _ in range(depth)])
    def forward(self, x):
        x = self.down(x)
        x = x.permute(0, 2, 3, 1)
        x = self.blocks(x)
        x = x.permute(0, 3, 1, 2)
        return x

class VSSBranch(nn.Module):
    def __init__(self, in_ch=3, dims=(64,128,256,512), depths=(2,2,2,2)):
        super().__init__()
        self.stem = StemPatchify(in_ch=in_ch, out_ch=dims[0])
        self.stage1 = VSSStage1_NoDown(dims[0], depths[0])      # H/4
        self.stage2 = VSSStage_DownThenBlocks(dims[0], dims[1], depths[1])  # H/8
        self.stage3 = VSSStage_DownThenBlocks(dims[1], dims[2], depths[2])  # H/16
        self.stage4 = VSSStage_DownThenBlocks(dims[2], dims[3], depths[3])  # H/32
    def forward_stem(self, x):
        return self.stem(x)
    def forward_stage1(self, x):
        return self.stage1(x)
    def forward_stage2(self, x):
        return self.stage2(x)
    def forward_stage3(self, x):
        return self.stage3(x)
    def forward_stage4(self, x):
        return self.stage4(x)

class ConvViTBranch(nn.Module):
    def __init__(self, in_ch=3, dims=(64,128,256,512), vit_blocks=8, vit_heads=4, vit_mlp=512):
        super().__init__()
        # Make C1 at H/4 like VSS stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], 3, 2, 1, bias=False), nn.BatchNorm2d(dims[0]), nn.ReLU(inplace=True),  # H/2
            nn.Conv2d(dims[0], dims[0], 3, 2, 1, bias=False), nn.BatchNorm2d(dims[0]), nn.ReLU(inplace=True), # H/4
        )
        self.down2 = ConvDown(dims[0], dims[1])  # H/8
        self.down3 = ConvDown(dims[1], dims[2])  # H/16
        self.down4 = ConvDown(dims[2], dims[3])  # H/32
        #self.vit_img_dim = 32  # 假设输入图像为256×256，对应H/32
        #self.vit = ViT(self.vit_img_dim, dims[3], dims[3],
                       #vit_heads, vit_mlp, vit_blocks, patch_dim=1, classification=False)
        #self.norm2 = nn.BatchNorm2d(dims[3])

    def forward(self, x):
        B, _, H, W = x.shape
        c1 = self.stem(x)            # H/4
        c2 = self.down2(c1)          # H/8
        c3 = self.down3(c2)          # H/16
        c4 = self.down4(c3)          # H/32
        #tokens = self.vit(c4)        # (B, (H/32*W/32), C4)
        #h4, w4 = max(H//32,1), max(W//32,1)
        #c4 = rearrange(tokens, 'b (h w) c -> b c h w', h=h4, w=w4)
        #c4 = self.norm2(c4)
        return c1, c2, c3, c4

# =========================
# Decoder with CA on VSS->Decoder skips
# =========================
class LocalSupervision(nn.Module):
    def __init__(self, in_channels=128, num_classes=6):
        super().__init__()
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(in_channels, num_classes, 1, 1, 0, bias=False)
    def forward(self, x, h, w):
        x = self.drop(self.conv3(x) + self.conv1(x))
        x = self.conv_out(x)
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

class PatchExpand(nn.Module):
    def __init__(self, dim_in, dim_out, scale=2):
        super().__init__()
        self.scale = scale
        self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(dim_out)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.bn(self.conv1x1(x))
        return x

class VSSLayerDecoder(nn.Module):
    def __init__(self, dim, depth=2):
        super().__init__()
        self.blocks = nn.Sequential(*[VSSBlock(hidden_dim=dim, drop_path=0.1) for _ in range(depth)])
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.blocks(x)
        x = x.permute(0, 3, 1, 2)
        return x

class MambaSegDecoder(nn.Module):
    def __init__(self, num_classes, encoder_channels=(64,128,256,512), depths=(2,2,2,2)):
        super().__init__()
        C1, C2, C3, C4 = encoder_channels
        # CA for VSS->Decoder skips
        #self.ca1 = ChannelAttention(C1)
        #self.ca2 = ChannelAttention(C2)
        #self.ca3 = ChannelAttention(C3)
        #self.ca4 = ChannelAttention(C4)
        #self.dec4 = VSSLayerDecoder(dim=C4, depth=depths[0])
        self.ls3 = LocalSupervision(C4, num_classes)

        self.up4 = PatchExpand(C4, C3, scale=2)  # H/32 -> H/16
        #self.fuse4 = nn.Conv2d(C3 + C3, C3, 1, 1, 0)
        #self.dec3 = VSSLayerDecoder(dim=C3, depth=depths[1])
        self.fuse4 = nn.Conv2d(C3 + C3, C3, 1, 1, 0)
        self.ls2   = LocalSupervision(C3, num_classes)

        self.up3 = PatchExpand(C3, C2, scale=2)  # H/16 -> H/8
        self.fuse3 = nn.Conv2d(C2 + C2, C2, 1, 1, 0)
        #self.dec2 = VSSLayerDecoder(dim=C2, depth=depths[2])
        #self.fuse3 = nn.Conv2d(C2 + C2, C2, 1, 1, 0)
        self.ls1   = LocalSupervision(C2, num_classes)

        self.up2 = PatchExpand(C2, C1, scale=2)  # H/8 -> H/4
        self.fuse2 = nn.Conv2d(C1 + C1, C1, 1, 1, 0)
        #self.dec1 = VSSLayerDecoder(dim=C1, depth=depths[3])
        #self.fuse2 = nn.Conv2d(C1 + C1, C1, 1, 1, 0)
        self.ls0   = LocalSupervision(C1, num_classes)

        self.up1 = PatchExpand(C1, C1//2 if C1>=2 else C1, scale=2)  # H/4 -> H/2
        self.up0 = PatchExpand(C1//2 if C1>=2 else C1, C1//2 if C1>=2 else C1, scale=2)  # H/2 -> H
        self.seg_head = nn.Conv2d(C1//2 if C1>=2 else C1, num_classes, 1, 1, 0)

    def forward(self, vss_skips, h, w):
        # vss_skips are already fused with CNN branch inside the encoder
        F1, F2, F3, F4 = vss_skips
        # apply CA to VSS->Decoder skips
        #F1 = self.ca1(F1) * F1
        #F2 = self.ca2(F2) * F2
        #F3 = self.ca3(F3) * F3
        #F4 = self.ca4(F4) * F4

        ls = []
        x = F4
        #x = self.dec4(x)
        ls.append(self.ls3(x,h,w))

        x = self.up4(x)
        x = torch.cat([x, F3], dim=1)
        x = self.fuse4(x)
        #x = self.dec3(x)
        ls.append(self.ls2(x, h, w))

        x = self.up3(x)
        x = torch.cat([x, F2], dim=1)
        x = self.fuse3(x)
        #x = self.dec2(x)
        ls.append(self.ls1(x, h, w))

        x = self.up2(x)
        x = torch.cat([x, F1], dim=1)
        x = self.fuse2(x)
        #x = self.dec1(x)
        ls.append(self.ls0(x, h, w))

        x = self.up1(x)
        x = self.up0(x)
        seg = self.seg_head(x)
        return seg, sum(ls)

# =========================
# Dual-branch encoder with CNN->VSS gated skips per stage (CA applied)
# =========================
class DualBranchEncoder(nn.Module):
    def __init__(self, in_ch=3, dims=(64,128,256,512), depths_vss=(2,2,2,2), vit_blocks=8, vit_heads=4, vit_mlp=512):
        super().__init__()
        #self.vss = VSSBranch(in_ch=in_ch, dims=dims, depths=depths_vss)
        self.cnn = ConvViTBranch(in_ch=in_ch, dims=dims, vit_blocks=vit_blocks, vit_heads=vit_heads, vit_mlp=vit_mlp)
        # CA on CNN->VSS per-stage jump
        #self.ca_c1 = ChannelAttention(dims[0])
        #self.ca_c2 = ChannelAttention(dims[1])
        #self.ca_c3 = ChannelAttention(dims[2])
        #self.ca_c4 = ChannelAttention(dims[3])
        # fusion (concat + 1x1) at each stage, output back to VSS stream
        #self.fuse1 = FuseConcat1x1(dims[0], dims[0], dims[0])
        #self.fuse2 = FuseConcat1x1(dims[1], dims[1], dims[1])
        #self.fuse3 = FuseConcat1x1(dims[2], dims[2], dims[2])
        #self.fuse4 = FuseConcat1x1(dims[3], dims[3], dims[3])
    def forward(self, x):
        # VSS stem
        #s0 = self.vss.forward_stem(x)         # H/4
        # CNN branch forward to get c1..c4
        c1, c2, c3, c4 = self.cnn(x)

        # Stage 1: VSS blocks then fuse CNN->VSS
        #s1_raw = self.vss.forward_stage1(s0)  # H/4
        #c1_g   = self.ca_c1(c1) * c1
        #c1_g = c1
        #s1 = self.fuse1(s1_raw, c1_g)         # H/4 (skip for decoder)

        # Stage 2
        #s2_raw = self.vss.forward_stage2(s1)  # H/8
        #c2_g   = self.ca_c2(c2) * c2
        #c2_g=c2
        #s2 = self.fuse2(s2_raw, c2_g)         # H/8

        # Stage 3
        #s3_raw = self.vss.forward_stage3(s2)  # H/16
        #c3_g   = self.ca_c3(c3) * c3
        #c3_g=c3
        #s3 = self.fuse3(s3_raw, c3_g)         # H/16

        # Stage 4
        #s4_raw = self.vss.forward_stage4(s3)  # H/32
        #c4_g   = self.ca_c4(c4) * c4
        #c4_g=c4
        #s4 = self.fuse4(s4_raw, c4_g)         # H/32

        return [c1, c2, c3, c4]

# =========================
# Top-level model
# =========================
class GatedMamba(nn.Module):
    def __init__(self,
                 in_ch=3,
                 num_classes=6,
                 dims=(64,128,256,512),
                 depths_vss=(2,2,2,1),
                 vit_blocks=2,
                 vit_heads=4,
                 vit_mlp=512,
                 use_crf=True):
        super().__init__()
        self.encoder = DualBranchEncoder(in_ch=in_ch, dims=dims, depths_vss=depths_vss,
                                         vit_blocks=vit_blocks, vit_heads=vit_heads, vit_mlp=vit_mlp)
        self.decoder = MambaSegDecoder(num_classes=num_classes, encoder_channels=dims, depths=(2,2,2,2))
        #self.use_crf = use_crf
        #self.crf = CRF_RNN(num_classes=num_classes)
    def forward(self, img, use_crf=None):
        h, w = img.size()[-2:]
        skips = self.encoder(img)
        if self.training:
            seg, aux = self.decoder(skips, h, w)
            #if (use_crf if use_crf is not None else self.use_crf):
                #seg = self.crf(seg, img)
            return seg, aux
        else:
            seg, _ = self.decoder(skips, h, w)
            #if (use_crf if use_crf is not None else self.use_crf):
                #seg = self.crf(seg, img)
            return seg

if __name__ == '__main__':
    model = GatedMamba(in_ch=3, num_classes=6)
    model=model.to('cuda:0')
    x = torch.randn(2,3,512,512)
    x=x.to('cuda:0')
    model.train()
    y, aux = model(x)
    print('train:', y.shape, aux.shape)
    model.eval()
    with torch.no_grad():
        y2 = model(x)
        print('eval:', y2.shape)
    from torchinfo import summary
    summary(model, input_size=(2, 3, 512, 512), col_names=("input_size", "output_size", "num_params"))