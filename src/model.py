import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    """Standard 4-level 3D U-Net. Input: (B, 1, D, H, W). Output: (B, num_classes, D, H, W)."""

    def __init__(self, in_channels=1, num_classes=6, base_channels=16):
        super().__init__()
        c = base_channels

        self.enc1 = conv_block(in_channels, c)
        self.enc2 = conv_block(c, c * 2)
        self.enc3 = conv_block(c * 2, c * 4)
        self.enc4 = conv_block(c * 4, c * 8)

        self.pool = nn.MaxPool3d(2)

        self.bottleneck = conv_block(c * 8, c * 16)

        self.up4 = nn.ConvTranspose3d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = conv_block(c * 16, c * 8)

        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(c * 8, c * 4)

        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(c * 4, c * 2)

        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = conv_block(c * 2, c)

        self.out_conv = nn.Conv3d(c, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


if __name__ == "__main__":
    # Quick sanity check: one forward pass with a small volume.
    model = UNet3D(in_channels=1, num_classes=6, base_channels=16)
    x = torch.randn(1, 1, 128, 128, 128)
    y = model(x)
    print(f"input: {x.shape}")
    print(f"output: {y.shape}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")