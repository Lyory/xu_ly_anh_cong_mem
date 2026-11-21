import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        groups = next(g for g in (8, 4, 2, 1) if out_ch % g == 0)
        super().__init__(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(groups, out_ch),
                         nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1),
                         nn.GroupNorm(groups, out_ch), nn.SiLU())


class AttentionGate(nn.Module):
    def __init__(self, skip_ch, gate_ch):
        super().__init__()
        hidden = max(1, skip_ch // 2)
        self.skip = nn.Conv2d(skip_ch, hidden, 1)
        self.gate = nn.Conv2d(gate_ch, hidden, 1)
        self.score = nn.Sequential(nn.SiLU(), nn.Conv2d(hidden, 1, 1), nn.Sigmoid())

    def forward(self, skip, gate):
        return skip * self.score(self.skip(skip) + self.gate(gate))


class UNetAttention(nn.Module):
    """Four downsampling levels, gated skips, additive RGB residual output."""
    def __init__(self, in_ch=3, out_ch=3, base=32):
        super().__init__()
        if in_ch not in (3, 4) or out_ch != 3 or base < 2:
            raise ValueError("Supported: in_ch=3/4, out_ch=3, base>=2")
        channels = [base * 2**i for i in range(5)]
        self.encoders = nn.ModuleList([ConvBlock(in_ch, base)] +
                                     [ConvBlock(a, b) for a, b in zip(channels[:-1], channels[1:])])
        self.gates = nn.ModuleList([AttentionGate(channels[i], channels[i+1]) for i in range(3, -1, -1)])
        self.decoders = nn.ModuleList([ConvBlock(channels[i]+channels[i+1], channels[i]) for i in range(3, -1, -1)])
        self.output = nn.Conv2d(base, out_ch, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image):
        skips = []
        x = image
        for i, encoder in enumerate(self.encoders):
            x = encoder(F.avg_pool2d(x, 2) if i else x)
            skips.append(x)
        for gate, decoder, skip in zip(self.gates, self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, gate(skip, x)], dim=1))
        return image[:, :3] + self.output(x)
