
import torch
import torch.nn as nn
from einops import rearrange
from .vit import ViT
from unetmamba_model.classification.models.vmamba_s import VSSBlock
import torch.nn.functional as F
import time
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



class FeatureEnhancementEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels))
    def forward(self, x):
        u = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        return u * x



class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)

        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        drop_path = 0.2
        use_checkpoint = False
        downsample = nn.Identity()
        channel_first = False
        # ===========================
        ssm_d_state = 16
        ssm_ratio = 2.0
        ssm_dt_rank = "auto"
        ssm_act_layer = nn.SiLU
        ssm_conv = 3
        ssm_conv_bias = True
        ssm_drop_rate = 0.0
        ssm_init = "v0"
        forward_type = "v3"
        # ===========================
        mlp_ratio = 4.0
        mlp_act_layer = nn.GELU
        mlp_drop_rate = 0.0
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.vss = VSSBlock(hidden_dim=out_channels, drop_path=drop_path)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)
        x= self.conv(x)
        #x=self.vss1(x)
        x=x.permute(0,2,3,1)
        x=self.vss(x)
        x = x.permute(0, 3, 1, 2)
        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()
        drop_path = 0.1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.vss1=VSSBlock(hidden_dim=out_channels,drop_path=drop_path)

        self.ca1 = ChannelAttention(out_channels)
        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.vss2 = VSSBlock(hidden_dim=out_channels*2, drop_path=drop_path)

        self.ca2 = ChannelAttention(out_channels * 2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.vss31 = VSSBlock(hidden_dim=out_channels*4, drop_path=drop_path)
        self.vss32 = VSSBlock(hidden_dim=out_channels*4, drop_path=drop_path)

        self.ca3 = ChannelAttention(out_channels * 4)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // 1
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        self.norm2 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)
        #print(x1.device)
        x1=x1.permute(0,2,3,1)
        x1=self.vss1(x1)
        x1 = x1.permute(0, 3, 1, 2)
        #print(x1.device)

        ca1 = self.ca1(x1)
        #print("ca1")
        out1 = x1*ca1
        #print("out1")
        x2 = self.encoder1(x1)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.vss2(x2)
        x2 = x2.permute(0, 3, 1, 2)
        ca2 = self.ca2(x2)
        #print("ca2")
        out2 = x2 * ca2
        #print("out2")
        x3 = self.encoder2(x2)
        x3 = x3.permute(0, 2, 3, 1)
        x3 = self.vss31(x3)
        x3 = self.vss32(x3)
        x3 = x3.permute(0, 3, 1, 2)
        ca3 = self.ca3(x3)
        #print("ca3")
        out3 = x3 * ca3
        #print("out3")
        x = self.encoder3(x3)

        x = self.vit(x)

        B, T, C = x.shape
        side = int(T ** 0.5)
        assert side * side == T, f"tokens={T} not square"
        x = rearrange(x, "b (h w) c -> b c h w", h=side, w=side)

        x = self.norm2(x)

        x = self.relu(x)
        #print("encoder finished")
        return x, out1,out2,out3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()
        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 4)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels*2)
        self.decoder3 = DecoderBottleneck(out_channels * 2, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(out_channels, class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):


        x = self.decoder1(x, x3)
        #print("decoder1 finished")
        x = self.decoder2(x, x2)
        #print("decoder2 finished")
        x = self.decoder3(x, x1)
        #print("decoder3 finished")
        x=self.upsample(x)
        #x=self.vss1(x)
        #x=self.vss2(x)
        x = self.conv1(x)
        #print("decoder finished")
        return x


class TransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, num_classes):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)
        self.decoder = Decoder(out_channels, num_classes)


    def forward(self, img):
        x, x1, x2, x3 = self.encoder(img)
        out = self.decoder(x, x1, x2, x3)
        return out







if __name__ == '__main__':
    import torch

    transunet = TransUNet(img_dim=128,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          num_classes=2)
    transunet=transunet.to("cuda:0")
    print(sum(p.numel() for p in transunet.parameters()))
    print(transunet(torch.randn(1, 3,128, 128).to("cuda:0")).shape)