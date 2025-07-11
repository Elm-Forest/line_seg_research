""" Full assembly of the parts to form the complete network """

from models.Unet.unet_parts import *
from models.deep_hough_transform.HT_cuda import HTIHT_Cuda, CAT_HTIHT_Cuda
from models.deep_hough_transform.dht_module.dht_func import C_dht

from models.direction_mask.dmg import DirectionalMaskGenerator
from models.modules.cga import CGAFusion


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class HTMask(nn.Module):
    def __init__(self, in_channels, feat_out_channels, mask_out_channels, numAngle, numRho, img_size):
        super(HTMask, self).__init__()
        self.ht = C_dht(numAngle, numRho)
        self.mask_gen = DirectionalMaskGenerator(img_size, numAngle, numRho)
        self.conv1x1 = nn.Sequential(
            conv3x3(in_channels, in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(in_channels, mask_out_channels, kernel_size=1)
        self.mask_out_channels = mask_out_channels

    def forward(self, x):
        # h, w = x.size(-2), x.size(-1)
        x = self.conv1x1(x)
        out = self.conv2(x)
        hough_map = self.ht(out.to(torch.float32))
        mask = self.mask_gen(hough_map)
        mask = mask.expand(-1, self.mask_out_channels, -1, -1)
        return mask, out


class FusionBlock(nn.Module):
    def __init__(self, in_channel_x, in_channel_feat, out_channels):
        super(FusionBlock, self).__init__()
        # self.cma = CMA_Block(in_channel_x, in_channel_feat, in_channel_x, out_channels)
        self.cga = CGAFusion(in_channel_x)

    def forward(self, x, mask):
        x = self.cga(x, mask)
        return x


class FusionBlock2(nn.Module):
    def __init__(self, in_channel_x, in_channel_feat, out_channels):
        super(FusionBlock2, self).__init__()
        # self.cma = CMA_Block(in_channel_x, in_channel_feat, in_channel_x, out_channels)
        self.conv1 = conv1x1(in_channel_x * 2, out_channels)

    def forward(self, x, mask):
        x = self.conv1(torch.cat([x, mask], 1))
        return x


class UNetHT(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, angle_res=3, rho_res=1, img_size=512,
                 device=torch.device('cuda')):
        super(UNetHT, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.ht1 = HTIHT_Cuda(512 // factor, 512 // factor, img_size // 8, img_size // 8, angle_res, rho_res)
        self.fb1 = FusionBlock(512 // factor, 512 // factor, 512 // factor)

        self.up2 = (Up(512, 256 // factor, bilinear))
        self.ht2 = HTIHT_Cuda(256 // factor, 256 // factor, img_size // 4, img_size // 4, angle_res, rho_res)
        self.fb2 = FusionBlock(256 // factor, 256 // factor, 256 // factor)

        self.up3 = (Up(256, 128 // factor, bilinear))
        self.ht3 = HTIHT_Cuda(128 // factor, 128 // factor, img_size // 2, img_size // 2, angle_res, rho_res)
        self.fb3 = FusionBlock(128 // factor, 128 // factor, 128 // factor)

        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        mask = self.ht1(x.float())
        x = self.fb1(x, mask)

        x = self.up2(x, x3)
        mask = self.ht2(x.float())
        x = self.fb2(x, mask)

        x = self.up3(x, x2)
        mask = self.ht3(x.float())
        x = self.fb3(x, mask)

        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    in_channel = 3
    h = 512
    w = 512
    rgb_input = torch.rand(1, in_channel, h, w).cuda()
    unet = UNetHT(3, 2, True, 3, 1, 512).cuda()
    temp = unet(rgb_input)
    print(temp.shape)
