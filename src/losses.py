import torch
from torch import nn
from torch.nn import functional as F
from src.models.dewarp_net import unwarp, flow_to_grid


def ssim_loss(x, y):
    from pytorch_msssim import ssim
    return 1 - ssim(x, y, data_range=1, size_average=True, win_size=7)


def gradient_magnitude(x):
    kernel = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    channels = x.shape[1]
    gx = F.conv2d(x, kernel.expand(channels, 1, 3, 3), padding=1, groups=channels)
    gy = F.conv2d(x, kernel.T.expand(channels, 1, 3, 3), padding=1, groups=channels)
    return torch.sqrt(gx*gx+gy*gy+1e-8)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        self.features.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, prediction, target):
        a = self.features((prediction-self.mean)/self.std)
        with torch.no_grad():
            b = self.features((target-self.mean)/self.std)
        return F.l1_loss(a, b)


class CombinedLoss(nn.Module):
    def __init__(self, l1=1., ssim=.2, perceptual=0., edge=.1):
        super().__init__()
        self.weights = l1, ssim, perceptual, edge
        self.perceptual = PerceptualLoss() if perceptual else None

    def forward(self, prediction, target):
        a, b, c, d = self.weights
        loss = a*F.l1_loss(prediction, target)
        if b:
            loss = loss + b*ssim_loss(prediction, target)
        if c:
            loss = loss + c*self.perceptual(prediction, target)
        if d:
            loss = loss + d*F.l1_loss(gradient_magnitude(prediction), gradient_magnitude(target))
        return loss


class DewarpLoss(nn.Module):
    def __init__(self, map_weight=1., image_weight=1., ssim=.2, smooth=.01):
        super().__init__()
        self.map_weight, self.image_weight, self.ssim, self.smooth = map_weight, image_weight, ssim, smooth

    def forward(self, flow, source, target, bmap):
        restored = unwarp(source, flow)
        loss = self.map_weight*F.l1_loss(flow_to_grid(flow), bmap)
        loss = loss + self.image_weight*F.l1_loss(restored, target)
        if self.ssim:
            loss = loss + self.ssim*ssim_loss(restored, target)
        loss = loss + self.smooth*((flow[:, :, 1:]-flow[:, :, :-1]).abs().mean() +
                                  (flow[:, :, :, 1:]-flow[:, :, :, :-1]).abs().mean())
        return loss
