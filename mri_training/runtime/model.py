"""MobileNetV3-UNet for cardiac short-axis MRI (export approved under the CardioAI MCP agreement).

Encoder : MobileNetV3-Large (Howard et al., 2019) in 2-D, input 2 channels (image, sequence flag)
Decoder : U-Net style, skip connections at 1/2, 1/4, 1/8, 1/16 -> per-pixel logits for
          CLASSES (background, LV blood pool, viable myocardium, scar, MVO, RV blood pool)
Study   : deepest encoder features pooled over each slice, then mean+max over the slices of a
          study -> one logit per study-level finding (STUDY_HEADS)

Normalisation: GroupNorm instead of MobileNetV3's BatchNorm. A training batch is the slices of ONE
study and ONE sequence, so BatchNorm would learn per-study statistics in training and use pooled
running statistics in serving (and erase the constant sequence channel); GroupNorm behaves the same
in both. Layer layout, widths, SE and activations are MobileNetV3-Large.

Weights are saved as a plain state_dict and loaded strict=True.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["background", "lv_pool", "myocardium", "scar", "mvo", "rv_pool"]
STUDY_HEADS = ["infarct", "nonischemic_lge", "lv_thrombus", "lv_aneurysm"]

# (kernel, expansion, out, squeeze-excite, activation, stride): MobileNetV3-Large
LARGE: List[Tuple[int, int, int, bool, str, int]] = [
    (3, 16, 16, False, "RE", 1), (3, 64, 24, False, "RE", 2), (3, 72, 24, False, "RE", 1),
    (5, 72, 40, True, "RE", 2), (5, 120, 40, True, "RE", 1), (5, 120, 40, True, "RE", 1),
    (3, 240, 80, False, "HS", 2), (3, 200, 80, False, "HS", 1), (3, 184, 80, False, "HS", 1),
    (3, 184, 80, False, "HS", 1), (3, 480, 112, True, "HS", 1), (3, 672, 112, True, "HS", 1),
    (5, 672, 160, True, "HS", 2), (5, 960, 160, True, "HS", 1), (5, 960, 160, True, "HS", 1),
]


def _div(v: float, d: int = 8) -> int:
    n = max(d, int(v + d / 2) // d * d)
    return n + d if n < 0.9 * v else n


def _norm(ch: int) -> nn.Module:
    return nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)


def _act(kind: str) -> nn.Module:
    return nn.Hardswish(inplace=True) if kind == "HS" else nn.ReLU(inplace=True)


class SE(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        mid = _div(ch // 4)
        self.f = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, mid, 1), nn.ReLU(inplace=True),
                               nn.Conv2d(mid, ch, 1), nn.Hardsigmoid(inplace=True))

    def forward(self, x):
        return x * self.f(x)


class IR(nn.Module):
    def __init__(self, cin, k, exp, cout, se, act, stride):
        super().__init__()
        self.res = stride == 1 and cin == cout
        layers: List[nn.Module] = []
        if exp != cin:
            layers += [nn.Conv2d(cin, exp, 1, bias=False), _norm(exp), _act(act)]
        layers += [nn.Conv2d(exp, exp, k, stride, k // 2, groups=exp, bias=False), _norm(exp), _act(act)]
        if se:
            layers.append(SE(exp))
        layers += [nn.Conv2d(exp, cout, 1, bias=False), _norm(cout)]
        self.b = nn.Sequential(*layers)

    def forward(self, x):
        y = self.b(x)
        return x + y if self.res else y


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), _norm(cout), nn.ReLU(inplace=True),
                               nn.Conv2d(cout, cout, 3, padding=1, bias=False), _norm(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.c(torch.cat([x, skip], 1))


class MobileNetV3UNet(nn.Module):
    def __init__(self, width: float = 1.0, in_ch: int = 2, n_classes: int = len(CLASSES),
                 study_heads: Sequence[str] = STUDY_HEADS, dropout: float = 0.2):
        super().__init__()
        self.study_heads = list(study_heads)
        c0 = _div(16 * width)
        self.stem = nn.Sequential(nn.Conv2d(in_ch, c0, 3, 2, 1, bias=False), _norm(c0), nn.Hardswish(inplace=True))
        blocks, cin, self.skip_at, skips = [], c0, [], []
        for i, (k, e, o, se, a, s) in enumerate(LARGE):
            e, o = _div(e * width), _div(o * width)
            if s == 2 and i > 0:
                self.skip_at.append(len(blocks) - 1)
                skips.append(cin)
            blocks.append(IR(cin, k, e, o, se, a, s))
            cin = o
        self.blocks = nn.ModuleList(blocks)
        last = _div(960 * width)
        self.head_conv = nn.Sequential(nn.Conv2d(cin, last, 1, bias=False), _norm(last), nn.Hardswish(inplace=True))
        # skips: [1/2, 1/4, 1/8, 1/16] (outputs just before each later stride-2 block); bottleneck 1/32
        d = [_div(c * width) for c in (256, 128, 64, 32)]
        self.up4 = Up(last, skips[3], d[0])
        self.up3 = Up(d[0], skips[2], d[1])
        self.up2 = Up(d[1], skips[1], d[2])
        self.up1 = Up(d[2], skips[0], d[3])
        self.seg = nn.Conv2d(d[3], n_classes, 1)
        self.study = nn.Sequential(nn.Linear(2 * last, 256), nn.Hardswish(inplace=True), nn.Dropout(dropout),
                                   nn.Linear(256, len(self.study_heads)))

    def encode(self, x):
        x = self.stem(x)
        skips = []
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i in self.skip_at:
                skips.append(x)
        return self.head_conv(x), skips

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (S, 2, H, W) slices of ONE study -> (seg logits (S, C, H, W), study logits (n_heads,))."""
        z, skips = self.encode(x)
        y = self.up4(z, skips[3])
        y = self.up3(y, skips[2])
        y = self.up2(y, skips[1])
        y = self.up1(y, skips[0])
        seg = F.interpolate(self.seg(y), size=x.shape[-2:], mode="bilinear", align_corners=False)
        pooled = F.adaptive_avg_pool2d(z, 1).flatten(1)                    # (S, last)
        study = self.study(torch.cat([pooled.mean(0), pooled.max(0).values]))
        return seg, study


def build_model(arch: Dict) -> MobileNetV3UNet:
    return MobileNetV3UNet(width=float(arch.get("width", 1.0)), study_heads=arch.get("study_heads", STUDY_HEADS),
                           dropout=float(arch.get("dropout", 0.2)))


def load_model(weights_path: str, arch: Dict, device: str = "cpu") -> MobileNetV3UNet:
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError("expected a plain state_dict of tensors (not a training checkpoint wrapper)")
    m = build_model(arch)
    m.load_state_dict(state, strict=True)
    return m.to(device).eval()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
