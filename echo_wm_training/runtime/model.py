"""MobileNetV3-WM: wall-motion model for apical echo cines (export approved: MobileNetV3 family).

Frame encoder : MobileNetV3-Small (Howard et al., 2019) in 2-D, 1 greyscale channel, GroupNorm instead of
                BatchNorm (a batch is a few clips of ONE study; GroupNorm behaves the same in training and serving).
Motion        : features at 1/16 (7 x 7 for 112 px) fused with the 1/32 features, then two residual temporal
                convolutions per location over the clip's frames; the mean and the standard deviation over time
                (how much each location moves) form the motion map.
Segments      : for each apical view, 7 slots (basal-left .. apex .. basal-right, segments.VIEW_SEGMENTS). Each
                slot attends to the motion map with a learned spatial prior (initialised along the LV outline,
                apex up) plus a content query, and a shared classifier gives 4 scores (normal, hypokinetic,
                akinetic, dyskinetic). Clips are averaged into the 17 AHA segments by `study_outputs`.
View          : A4C / A2C / A3C / other from the clip's appearance (quality check and data selection).
Global        : LVEF (%) and global hypokinesis, from all clips of the study (heart-failure assessment).

Weights are saved as a plain state_dict and loaded strict=True.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segments import N_SCORE, N_SEG, N_SLOT, VIEW_CLASSES, VIEWS, slot_index

# (kernel, expansion, out, squeeze-excite, activation, stride): MobileNetV3-Small
SMALL: List[Tuple[int, int, int, bool, str, int]] = [
    (3, 16, 16, True, "RE", 2), (3, 72, 24, False, "RE", 2), (3, 88, 24, False, "RE", 1),
    (5, 96, 40, True, "HS", 2), (5, 240, 40, True, "HS", 1), (5, 240, 40, True, "HS", 1),
    (5, 120, 48, True, "HS", 1), (5, 144, 48, True, "HS", 1), (5, 288, 96, True, "HS", 2),
    (5, 576, 96, True, "HS", 1), (5, 576, 96, True, "HS", 1),
]
GRID = 7                     # 1/16 of 112 px


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


class TemporalBlock(nn.Module):
    def __init__(self, d: int, k: int = 5):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(d, d, k, padding=k // 2, groups=d, bias=False), _norm(d), nn.Hardswish(inplace=True),
                               nn.Conv1d(d, d, 1, bias=False), _norm(d))

    def forward(self, x):                      # (N, D, T)
        return F.hardswish(x + self.f(x))


def _slot_prior(sigma: float = 1.0) -> torch.Tensor:
    """(7, 49) log-prior over the 7 x 7 grid: slots along an inverted U (apex at the top centre)."""
    pts = [(5.5, 1.5), (3.5, 1.2), (1.8, 2.0), (0.8, 3.0), (1.8, 4.0), (3.5, 4.8), (5.5, 4.5)]
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    pri = np.stack([-((yy - r) ** 2 + (xx - c) ** 2).ravel() / (2 * sigma ** 2) for r, c in pts])
    return torch.tensor(pri, dtype=torch.float32)


class MobileNetV3WM(nn.Module):
    def __init__(self, width: float = 1.0, d: int = 128, dropout: float = 0.2):
        super().__init__()
        c0 = _div(16 * width)
        self.stem = nn.Sequential(nn.Conv2d(1, c0, 3, 2, 1, bias=False), _norm(c0), nn.Hardswish(inplace=True))
        blocks, cin, self.tap = [], c0, None
        for i, (k, e, o, se, a, s) in enumerate(SMALL):
            e, o = _div(e * width), _div(o * width)
            if s == 2 and i == 8:
                self.tap = len(blocks) - 1         # last block at 1/16
                c16 = cin
            blocks.append(IR(cin, k, e, o, se, a, s))
            cin = o
        self.blocks = nn.ModuleList(blocks)
        last = _div(576 * width)
        self.head_conv = nn.Sequential(nn.Conv2d(cin, last, 1, bias=False), _norm(last), nn.Hardswish(inplace=True))
        self.d = d
        self.p16 = nn.Sequential(nn.Conv2d(c16, d // 2, 1, bias=False), _norm(d // 2), nn.Hardswish(inplace=True))
        self.p32 = nn.Sequential(nn.Conv2d(last, d, 1, bias=False), _norm(d), nn.Hardswish(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(d + d // 2, d, 1, bias=False), _norm(d), nn.Hardswish(inplace=True))
        self.temporal = nn.Sequential(TemporalBlock(d), TemporalBlock(d))
        self.motion = nn.Sequential(nn.Conv2d(2 * d, d, 1, bias=False), _norm(d), nn.Hardswish(inplace=True))
        nv = len(VIEWS)
        self.slot_prior = nn.Parameter(_slot_prior().unsqueeze(0).repeat(nv, 1, 1))     # (views, 7, 49)
        self.slot_query = nn.Parameter(torch.randn(nv, N_SLOT, d) * 0.02)
        self.slot_embed = nn.Parameter(torch.randn(nv, N_SLOT, d) * 0.02)
        self.slot_head = nn.Sequential(nn.Linear(d, d), nn.Hardswish(inplace=True), nn.Dropout(dropout), nn.Linear(d, N_SCORE))
        self.app = nn.Linear(last, d)
        self.view_head = nn.Sequential(nn.Linear(2 * d, d), nn.Hardswish(inplace=True), nn.Dropout(dropout),
                                       nn.Linear(d, len(VIEW_CLASSES)))
        self.global_head = nn.Sequential(nn.Linear(2 * d, d), nn.Hardswish(inplace=True), nn.Dropout(dropout), nn.Linear(d, 2))
        self.register_buffer("seg_of_slot", torch.from_numpy(slot_index()), persistent=False)

    # ------------------------------------------------------------------ encoder (shared with pretraining)
    def encode(self, x):
        """x (N, 1, H, W) -> (1/32 features after head_conv, 1/16 features)."""
        x = self.stem(x)
        f16 = None
        for i, b in enumerate(self.blocks):
            x = b(x)
            if i == self.tap:
                f16 = x
        return self.head_conv(x), f16

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x (N, T, H, W): N clips of ONE study -> per-clip outputs.

        slots  (N, 3, 7, 4)  slot logits for each apical view's slots (the clip's view picks one row)
        view   (N, 4)        view logits (A4C, A2C, A3C, other)
        feat   (N, 2d)       clip feature for the global head (study_outputs)
        attn   (N, 3, 7, 49) slot attention maps (for display)"""
        n, t, h, w = x.shape
        z32, z16 = self.encode(x.reshape(n * t, 1, h, w))
        g = z16.shape[-1]
        f = self.fuse(torch.cat([self.p16(z16), F.interpolate(self.p32(z32), size=z16.shape[-2:], mode="bilinear",
                                                              align_corners=False)], 1))       # (N*T, d, g, g)
        f = f.reshape(n, t, self.d, g * g).permute(0, 3, 2, 1).reshape(n * g * g, self.d, t)
        f = self.temporal(f)
        stats = torch.cat([f.mean(-1), f.std(-1)], 1).reshape(n, g * g, 2 * self.d).permute(0, 2, 1).reshape(n, 2 * self.d, g, g)
        mo = self.motion(stats).flatten(2)                                                      # (N, d, g*g)
        prior = self.slot_prior if g == GRID else F.interpolate(
            self.slot_prior.reshape(-1, 1, GRID, GRID), size=(g, g), mode="bilinear", align_corners=False).reshape(
            self.slot_prior.shape[0], N_SLOT, g * g)
        logits = prior.unsqueeze(0) + torch.einsum("vsd,ndl->nvsl", self.slot_query, mo) / math.sqrt(self.d)
        attn = logits.softmax(-1)                                                               # (N, 3, 7, L)
        pooled = torch.einsum("nvsl,ndl->nvsd", attn, mo) + self.slot_embed.unsqueeze(0)
        slots = self.slot_head(pooled)                                                          # (N, 3, 7, 4)
        app = self.app(F.adaptive_avg_pool2d(z32, 1).flatten(1).reshape(n, t, -1).mean(1))     # (N, d)
        mpool = mo.mean(-1)
        view = self.view_head(torch.cat([app, mpool], 1))
        return {"slots": slots, "view": view, "feat": torch.cat([app, mpool], 1), "attn": attn}

    def study_outputs(self, out: Dict[str, torch.Tensor], view_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Combine a study's clips: view_idx (N,) in 0..2 (A4C, A2C, A3C) or -1 (not an apical view).

        seg   (17, 4) segment logits averaged over the clips that show each segment (0 where none)
        seen  (17,)   bool
        lvef  ()      percent, from all apical clips (all clips if none is apical)
        ghypo ()      global hypokinesis logit"""
        dev = out["slots"].device
        seg = torch.zeros(N_SEG, N_SCORE, device=dev)
        cnt = torch.zeros(N_SEG, device=dev)
        apical = view_idx >= 0
        for i in torch.nonzero(apical).flatten().tolist():
            v = int(view_idx[i])
            idx = self.seg_of_slot[v]
            seg = seg.index_add(0, idx, out["slots"][i, v])
            cnt = cnt.index_add(0, idx, torch.ones(N_SLOT, device=dev))
        seen = cnt > 0
        seg = seg / cnt.clamp(min=1).unsqueeze(1)
        feats = out["feat"][apical] if apical.any() else out["feat"]
        gl = self.global_head(feats.mean(0))
        return {"seg": seg, "seen": seen, "lvef": 100.0 * torch.sigmoid(gl[0]), "ghypo": gl[1]}


def build_model(arch: Dict) -> MobileNetV3WM:
    return MobileNetV3WM(width=float(arch.get("width", 1.0)), d=int(arch.get("d", 128)), dropout=float(arch.get("dropout", 0.2)))


def load_model(weights_path: str, arch: Dict, device: str = "cpu") -> MobileNetV3WM:
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError("expected a plain state_dict of tensors (not a training checkpoint wrapper)")
    m = build_model(arch)
    m.load_state_dict(state, strict=True)
    return m.to(device).eval()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def view_index(views: Sequence[Optional[str]]) -> torch.Tensor:
    return torch.tensor([VIEWS.index(v) if v in VIEWS else -1 for v in views], dtype=torch.long)
