import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRFasRNN_Module(nn.Module):
    def __init__(
        self,
        num_classes,
        num_iterations=5,
        win=3,
        spatial_ker_weight=1.0,
        bilateral_ker_weight=1.0,
        theta_alpha=80.0,
        theta_beta=13.0,
        theta_gamma=3.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.win = win

        self.spatial_ker_weight = nn.Parameter(torch.tensor(spatial_ker_weight, requires_grad=True))
        self.bilateral_ker_weight = nn.Parameter(torch.tensor(bilateral_ker_weight, requires_grad=True))

        self.theta_alpha = nn.Parameter(torch.tensor(theta_alpha, requires_grad=True))
        self.theta_beta = nn.Parameter(torch.tensor(theta_beta, requires_grad=True))
        self.theta_gamma = nn.Parameter(torch.tensor(theta_gamma, requires_grad=True))

    def _gaussian_kernel(self, device, dtype):
        k = self.win
        pad = k // 2

        y, x = torch.meshgrid(
            torch.arange(-pad, pad + 1, device=device, dtype=dtype),
            torch.arange(-pad, pad + 1, device=device, dtype=dtype),
            indexing="ij"
        )

        sigma = torch.clamp(self.theta_gamma.to(device=device, dtype=dtype), min=1e-6)
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / (kernel.sum() + 1e-8)

        # [C, 1, k, k] for depthwise convolution
        kernel = kernel.view(1, 1, k, k).repeat(self.num_classes, 1, 1, 1)
        return kernel

    def forward(self, unary_logits, image):
        Q = F.softmax(unary_logits, dim=1)
        B, C, H, W = unary_logits.size()

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=image.device),
            torch.arange(W, device=image.device),
            indexing="ij"
        )
        pos = torch.stack([grid_y, grid_x], dim=0).float().unsqueeze(0).repeat(B, 1, 1, 1)

        pad = self.win // 2

        for _ in range(self.num_iterations):
            # Gaussian spatial filtering, closer to CRFasRNN-style spatial message passing
            kernel = self._gaussian_kernel(Q.device, Q.dtype)
            spatial_out = F.conv2d(Q, kernel, padding=pad, groups=C)

            # Bilateral filtering
            bilateral_out = self._bilateral_filter(Q, image, pos, win=self.win)

            pairwise = self.spatial_ker_weight * spatial_out + self.bilateral_ker_weight * bilateral_out
            Q = F.softmax(unary_logits - pairwise, dim=1)

        return Q

    def _bilateral_filter(self, Q, image, pos, win=3):
        B, C, H, W = Q.shape
        pad = win // 2

        Q_pad = F.pad(Q, [pad] * 4, mode="reflect")
        img_pad = F.pad(image, [pad] * 4, mode="reflect")
        pos_pad = F.pad(pos, [pad] * 4, mode="reflect")

        out = torch.zeros_like(Q)

        for i in range(-pad, pad + 1):
            for j in range(-pad, pad + 1):
                shifted_Q = Q_pad[:, :, pad + i:H + pad + i, pad + j:W + pad + j]
                shifted_img = img_pad[:, :, pad + i:H + pad + i, pad + j:W + pad + j]
                shifted_pos = pos_pad[:, :, pad + i:H + pad + i, pad + j:W + pad + j]

                color_diff = ((image - shifted_img) / self.theta_alpha) ** 2
                pos_diff = ((pos - shifted_pos) / self.theta_beta) ** 2

                weight = torch.exp(
                    -(
                        color_diff.sum(dim=1, keepdim=True)
                        + pos_diff.sum(dim=1, keepdim=True)
                    )
                )

                out += weight * shifted_Q

        return out / ((win ** 2) + 1e-8)

def benchmark_crfasrnn(
    num_classes=2,
    h=256,
    w=256,
    batch_size=1,
    num_iterations=5,
    win=3,
    warmup=50,
    repeat=300
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    module = CRFasRNN_Module(
        num_classes=num_classes,
        num_iterations=num_iterations,
        win=win
    ).to(device)

    module.eval()

    unary_logits = torch.randn(batch_size, num_classes, h, w, device=device)
    image = torch.randn(batch_size, 3, h, w, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = module(unary_logits, image)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(repeat):
            _ = module(unary_logits, image)

    if device == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start
    avg_time = total_time / repeat
    fps = 1.0 / avg_time

    print(f"CRFasRNN-style, win={win}x{win}, iterations={num_iterations}")
    print(f"Input size: {h}x{w}")
    print(f"Average time: {avg_time * 1000:.3f} ms")
    print(f"FPS: {fps:.2f}")

    return avg_time, fps


benchmark_crfasrnn(win=3)