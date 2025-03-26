""" Full assembly of the parts to form the complete network """
import math

from models.Conv.line_prior_conv import MultiAngleGatedConv
from models.Unet.unet_parts import *


class UNet_Line2(nn.Module):
    """
    基于官方 UNet 改进 => 在编码器若干层插入多角度方向先验卷积
    """

    def __init__(self, n_channels, n_classes, bilinear=False, angles=None):
        super(UNet_Line2, self).__init__()
        if angles is None:
            angles = [i * math.pi / 12 for i in range(12)]
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 1) 输入 => DoubleConv => MultiAngleGatedConv => Down
        self.inc = DoubleConv(n_channels, 64)
        self.prior1 = MultiAngleGatedConv(64, 64, angle_list=angles)
        self.down1 = Down(64, 128)

        self.prior2 = MultiAngleGatedConv(128, 128, angle_list=angles)
        self.down2 = Down(128, 256)

        self.prior3 = MultiAngleGatedConv(256, 256, angle_list=angles)
        self.down3 = Down(256, 512)

        self.down4 = Down(512, 512)

        # 2) Decoder部分(可按需也加)
        self.up1 = Up(512, 256)
        self.prior_up1 = MultiAngleGatedConv(256, 256, angle_list=angles)

        self.up2 = Up(256, 128)
        self.prior_up2 = MultiAngleGatedConv(128, 128, angle_list=angles)

        self.up3 = Up(128, 64)
        self.prior_up3 = MultiAngleGatedConv(64, 64, angle_list=angles)

        self.up4 = Up(64, 64)
        # final conv
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)  # [B,64,H,W]
        x1p = self.prior1(x1)  # line prior conv
        x2 = self.down1(x1p)  # [B,128,H/2,W/2]
        x2p = self.prior2(x2)
        x3 = self.down2(x2p)  # [B,256,H/4,W/4]
        x3p = self.prior3(x3)
        x4 = self.down3(x3p)  # [B,512,H/8,W/8]
        x5 = self.down4(x4)  # [B,512,H/16,W/16]

        # Decoder
        u1 = self.up1(x5, x4)  # -> 256
        u1p = self.prior_up1(u1)
        u2 = self.up2(u1p, x3)  # -> 128
        u2p = self.prior_up2(u2)
        u3 = self.up3(u2p, x2)  # -> 64
        u3p = self.prior_up3(u3)
        u4 = self.up4(u3p, x1)  # -> 64
        logits = self.outc(u4)
        return logits


class UNet_Line(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, angles=None):
        super(UNet_Line, self).__init__()
        if angles is None:
            angles = [i * math.pi / 12 for i in range(12)]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.prior1 = MultiAngleGatedConv(64, 64, angle_list=angles)
        self.down1 = (Down(64, 128))

        self.prior2 = MultiAngleGatedConv(128, 128, angle_list=angles)
        self.down2 = (Down(128, 256))

        self.prior3 = MultiAngleGatedConv(256, 256, angle_list=angles)
        self.down3 = (Down(256, 512))

        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.prior_up1 = MultiAngleGatedConv(256, 256, angle_list=angles)

        self.up2 = (Up(512, 256 // factor, bilinear))
        self.prior_up2 = MultiAngleGatedConv(128, 128, angle_list=angles)

        self.up3 = (Up(256, 128 // factor, bilinear))
        self.prior_up3 = MultiAngleGatedConv(64, 64, angle_list=angles)

        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)  # [B,64,H,W]
        x1p = self.prior1(x1)  # line prior conv
        x2 = self.down1(x1p)  # [B,128,H/2,W/2]
        x2p = self.prior2(x2)
        x3 = self.down2(x2p)  # [B,256,H/4,W/4]
        x3p = self.prior3(x3)
        x4 = self.down3(x3p)  # [B,512,H/8,W/8]
        x5 = self.down4(x4)  # [B,512,H/16,W/16]

        # Decoder
        u1 = self.up1(x5, x4)  # -> 256
        u1p = self.prior_up1(u1)
        u2 = self.up2(u1p, x3)  # -> 128
        u2p = self.prior_up2(u2)
        u3 = self.up3(u2p, x2)  # -> 64
        u3p = self.prior_up3(u3)
        u4 = self.up4(u3p, x1)  # -> 64
        logits = self.outc(u4)
        return logits
