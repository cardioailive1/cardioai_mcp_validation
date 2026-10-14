"""MobileNetV3 adapted to 1-D, 12-lead ECG (input 12 x 5000 at 500 Hz, 10 s).

MobileNetV3-Small block layout (Howard et al., 2019): inverted residuals with
depthwise convolutions, squeeze-and-excitation and hard-swish, with kernels
widened for 500 Hz signals (7 and 11 samples = 14 / 22 ms per layer, growing with
downsampling). Five stride-2 stages -> 157 time steps, global average pool, then a
small head with one sigmoid output per finding (multi-label).

Export: weights are saved as a plain state_dict and always loaded with
strict=True (load_model), so a mismatched or wrapped checkpoint fails loudly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

HEADS = ["acute_mi", "stemi", "af"]

# (kernel, expansion channels, out channels, squeeze-excite, activation, stride)
SMALL_1D: List[Tuple[int, int, int, bool, str, int]] = [
    (7, 16, 16, True, "RE", 2),
    (7, 72, 24, False, "RE", 2),
    (7, 88, 24, False, "RE", 1),
    (11, 96, 40, True, "HS", 2),
    (11, 240, 40, True, "HS", 1),
    (11, 240, 40, True, "HS", 1),
    (11, 120, 48, True, "HS", 1),
    (11, 144, 48, True, "HS", 1),
    (11, 288, 96, True, "HS", 2),
    (11, 576, 96, True, "HS", 1),
    (11, 576, 96, True, "HS", 1),
]


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


def _act(kind: str) -> nn.Module:
    return nn.Hardswish(inplace=True) if kind == "HS" else nn.ReLU(inplace=True)


class SqueezeExcite1d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        mid = _make_divisible(ch // 4)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(ch, mid, 1), nn.ReLU(inplace=True),
                                nn.Conv1d(mid, ch, 1), nn.Hardsigmoid(inplace=True))

    def forward(self, x):
        return x * self.fc(x)


class InvertedResidual1d(nn.Module):
    def __init__(self, cin: int, k: int, exp: int, cout: int, se: bool, act: str, stride: int):
        super().__init__()
        self.use_res = stride == 1 and cin == cout
        layers: List[nn.Module] = []
        if exp != cin:
            layers += [nn.Conv1d(cin, exp, 1, bias=False), nn.BatchNorm1d(exp), _act(act)]
        layers += [nn.Conv1d(exp, exp, k, stride, padding=k // 2, groups=exp, bias=False), nn.BatchNorm1d(exp), _act(act)]
        if se:
            layers.append(SqueezeExcite1d(exp))
        layers += [nn.Conv1d(exp, cout, 1, bias=False), nn.BatchNorm1d(cout)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class MobileNetV3ECG(nn.Module):
    def __init__(self, heads: Sequence[str] = HEADS, in_leads: int = 12, width: float = 1.0,
                 cfg: Optional[List[Tuple[int, int, int, bool, str, int]]] = None, dropout: float = 0.2):
        super().__init__()
        self.heads = list(heads)
        cfg = cfg or SMALL_1D
        c0 = _make_divisible(16 * width)
        layers: List[nn.Module] = [nn.Conv1d(in_leads, c0, 15, 2, padding=7, bias=False), nn.BatchNorm1d(c0), nn.Hardswish(inplace=True)]
        cin = c0
        for k, exp, cout, se, act, s in cfg:
            exp, cout = _make_divisible(exp * width), _make_divisible(cout * width)
            layers.append(InvertedResidual1d(cin, k, exp, cout, se, act, s))
            cin = cout
        last = _make_divisible(576 * width)
        layers += [nn.Conv1d(cin, last, 1, bias=False), nn.BatchNorm1d(last), nn.Hardswish(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Linear(last, 256), nn.Hardswish(inplace=True), nn.Dropout(dropout),
                                        nn.Linear(256, len(self.heads)))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 12, 5000) -> logits (B, n_heads)."""
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))


def build_model(arch: Dict) -> MobileNetV3ECG:
    return MobileNetV3ECG(heads=arch.get("heads", HEADS), width=float(arch.get("width", 1.0)),
                          dropout=float(arch.get("dropout", 0.2)))


def load_model(weights_path: str, arch: Dict, device: str = "cpu") -> MobileNetV3ECG:
    """Plain state_dict only, strict=True: a wrapped or mismatched checkpoint raises."""
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError("expected a plain state_dict of tensors (not a training checkpoint wrapper)")
    model = build_model(arch)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
