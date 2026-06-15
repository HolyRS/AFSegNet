import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SegHead(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(x)


class UNetLikeDecoderWithAuxLoss(nn.Module):
    """
    输入:
        out_128 : [B,  64, 128, 128]
        out_64  : [B,  64,  64,  64]
        out_32  : [B, 128,  32,  32]
        out_16  : [B, 256,  16,  16]

    输出:
        main_out : [B, 2, H, W]
        aux16    : [B, 2, H, W]
        aux32    : [B, 2, H, W]
        aux64    : [B, 2, H, W]
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # ---------- aux16 ----------
        self.aux16_head = nn.Sequential(
            ConvBNReLU(256, 64, 3, 1, 1),
            SegHead(64, num_classes)
        )

        # ---------- 16 -> 32 ----------
        # 下层特征先变到和当前层 out_32 一样的通道数: 128
        self.proj16_to_32 = ConvBNReLU(256, 128, kernel_size=1, stride=1, padding=0)

        # concat后: 128 + 128 = 256，再融合回 128
        self.fuse32 = nn.Sequential(
            ConvBNReLU(256, 128, 3, 1, 1),
            ConvBNReLU(128, 128, 3, 1, 1)
        )

        self.aux32_head = nn.Sequential(
            ConvBNReLU(128, 64, 3, 1, 1),
            SegHead(64, num_classes)
        )

        # ---------- 32 -> 64 ----------
        # 下层特征先变到和当前层 out_64 一样的通道数: 64
        self.proj32_to_64 = ConvBNReLU(128, 64, kernel_size=1, stride=1, padding=0)

        # concat后: 64 + 64 = 128，再融合回 64
        self.fuse64 = nn.Sequential(
            ConvBNReLU(128, 64, 3, 1, 1),
            ConvBNReLU(64, 64, 3, 1, 1)
        )

        self.aux64_head = nn.Sequential(
            ConvBNReLU(64, 64, 3, 1, 1),
            SegHead(64, num_classes)
        )

        # ---------- 64 -> 128 ----------
        # 下层特征先变到和当前层 out_128 一样的通道数: 64
        self.proj64_to_128 = ConvBNReLU(64, 64, kernel_size=1, stride=1, padding=0)

        # concat后: 64 + 64 = 128，再融合回 64
        self.fuse128 = nn.Sequential(
            ConvBNReLU(128, 64, 3, 1, 1),
            ConvBNReLU(64, 64, 3, 1, 1)
        )

        self.main_head = nn.Sequential(
            ConvBNReLU(64, 64, 3, 1, 1),
            SegHead(64, num_classes)
        )

    def forward(self, out_128, out_64, out_32, out_16, target_size=None):
        """
        target_size: 最终分割输出大小，例如 (256, 256)
        如果为 None，则默认输出到 out_128 的 2 倍，即 (256, 256)
        """
        if target_size is None:
            target_size = (out_128.shape[-2] * 2, out_128.shape[-1] * 2)

        # =========================================================
        # 1) out_16 直接做辅助监督
        # =========================================================
        aux16 = self.aux16_head(out_16)
        aux16 = F.interpolate(aux16, size=target_size, mode='bilinear', align_corners=False)

        # =========================================================
        # 2) out_16 -> out_32
        # 先上采样到 32x32，再变通道到 128，与 out_32 对齐后 concat
        # =========================================================
        x16_up = F.interpolate(out_16, size=out_32.shape[-2:], mode='bilinear', align_corners=False)
        x16_up = self.proj16_to_32(x16_up)              # [B, 128, 32, 32]

        x32 = torch.cat([out_32, x16_up], dim=1)       # [B, 256, 32, 32]
        x32 = self.fuse32(x32)                         # [B, 128, 32, 32]

        aux32 = self.aux32_head(x32)
        aux32 = F.interpolate(aux32, size=target_size, mode='bilinear', align_corners=False)

        # =========================================================
        # 3) x32 -> out_64
        # 先上采样到 64x64，再变通道到 64，与 out_64 对齐后 concat
        # =========================================================
        x32_up = F.interpolate(x32, size=out_64.shape[-2:], mode='bilinear', align_corners=False)
        x32_up = self.proj32_to_64(x32_up)             # [B, 64, 64, 64]

        x64 = torch.cat([out_64, x32_up], dim=1)       # [B, 128, 64, 64]
        x64 = self.fuse64(x64)                         # [B, 64, 64, 64]

        aux64 = self.aux64_head(x64)
        aux64 = F.interpolate(aux64, size=target_size, mode='bilinear', align_corners=False)

        # =========================================================
        # 4) x64 -> out_128
        # 先上采样到 128x128，再变通道到 64，与 out_128 对齐后 concat
        # =========================================================
        x64_up = F.interpolate(x64, size=out_128.shape[-2:], mode='bilinear', align_corners=False)
        x64_up = self.proj64_to_128(x64_up)            # [B, 64, 128, 128]

        x128 = torch.cat([out_128, x64_up], dim=1)     # [B, 128, 128, 128]
        x128 = self.fuse128(x128)                      # [B, 64, 128, 128]

        # 主输出
        x128_up = F.interpolate(x128, size=target_size, mode='bilinear', align_corners=False)
        main_out = self.main_head(x128_up)             # [B, 2, H, W]
        return main_out,aux16,aux32,aus64

if __name__ == "__main__":
    out_128 = torch.randn(1, 64, 128, 128)
    out_64  = torch.randn(1, 64, 64, 64)
    out_32  = torch.randn(1, 128, 32, 32)
    out_16  = torch.randn(1, 256, 16, 16)

    model = UNetLikeDecoderWithAuxLoss(num_classes=2)
    outputs = model(out_128, out_64, out_32, out_16, target_size=(256, 256))
    print(outputs.shape)