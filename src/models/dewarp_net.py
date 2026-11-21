import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from src.models.unet_attention import ConvBlock


def identity_grid(batch, height, width, device=None, dtype=torch.float32):
    y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=device, dtype=dtype),
                          torch.linspace(-1, 1, width, device=device, dtype=dtype), indexing="ij")
    return torch.stack([x, y], -1).unsqueeze(0).expand(batch, -1, -1, -1)


def flow_to_grid(flow):
    b, _, h, w = flow.shape
    return identity_grid(b, h, w, flow.device, flow.dtype) + flow.permute(0, 2, 3, 1)


def unwarp(img, flow):
    """flow: normalized displacement Bx2xHxW, not pixel displacement."""
    if flow.shape[-2:] != img.shape[-2:]:
        flow = F.interpolate(flow, size=img.shape[-2:], mode="bilinear", align_corners=True)
    # White padding matches synthetic generation and document backgrounds.
    return F.grid_sample(img-1, flow_to_grid(flow), align_corners=True, padding_mode="zeros") + 1


class DewarpNet(nn.Module):
    def __init__(self, backbone="resnet18", pretrained_encoder=False, max_displacement=.25):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError("Only resnet18 is supported")
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained_encoder else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layers = nn.ModuleList([encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4])
        self.decoders = nn.ModuleList([ConvBlock(512+256, 256), ConvBlock(256+128, 128),
                                      ConvBlock(128+64, 64), ConvBlock(64+64, 32)])
        self.output = nn.Conv2d(32, 2, 1)
        self.max_displacement = max_displacement
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, img):
        x = self.stem((img-self.mean)/self.std)
        skips = [x]
        x = self.pool(x)
        for layer in self.layers:
            x = layer(x)
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=True)
            x = decoder(torch.cat([x, skip], 1))
        x = F.interpolate(self.output(x), img.shape[-2:], mode="bilinear", align_corners=True)
        return torch.tanh(x) * self.max_displacement
