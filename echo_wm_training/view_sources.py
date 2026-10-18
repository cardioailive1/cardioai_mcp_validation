"""Automatic view labels for echo clips, so people label as little as possible.

Three sources, tried in this order (views.assign_views combines them; a person's label always wins):

  1. tags      DICOM text tags the scanner wrote (io.list_clips -> view_tag)
  2. sr        the echo machine's measurement report (DICOM Structured Report, SR) stored in the study: a
               measurement made on a clip references that clip's SOPInstanceUID and names its view ("Apical four
               chamber", "A2C", "Method of disks, single plane, A4C"). A clip a sonographer traced an EF on is,
               by definition, that view.
  3. cluster   clips grouped by the self-supervised encoder (pretrain.py): a group whose tagged / SR clips agree
               (>= cluster_auto_purity, at least cluster_auto_min of them) takes their view automatically; the
               other groups are labelled by a person from one picture of 8 sample clips per group (about 60
               pictures instead of 2,000 clips). A group marked "mixed" is left unlabelled.

    python -m echo_wm_training.view_sources coverage --clips clips.parquet --studies /workspace/echo/dicom --out /workspace/echo/views
        -> sr_views.csv (clip view from the measurement reports) + coverage.json (what share of clips and
           studies each source covers: decide whether clustering is needed at all)
    python -m echo_wm_training.view_sources clusters --clips clips.parquet --sr /workspace/echo/views/sr_views.csv \
           --encoder /workspace/echo/runs/ssl_v1 --k 60 --out /workspace/echo/views/clusters
        -> clusters.csv, img/C001.png ..., clusters.xlsx (only the groups no source settles)
    python -m echo_wm_training.view_sources read-clusters --dir /workspace/echo/views/clusters --out cluster_views.csv

Everything stays inside MCP.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .runtime.segments import VIEW_CLASSES, VIEWS

VIEW_WORDS = [
    ("A4C", re.compile(r"(?i)(apical\s*four|apical\s*4|four[- ]?chamber|4[- ]?ch\b|\ba4c\b|\bap4\b)")),
    ("A2C", re.compile(r"(?i)(apical\s*two|apical\s*2|two[- ]?chamber|2[- ]?ch\b|\ba2c\b|\bap2\b)")),
    ("A3C", re.compile(r"(?i)(apical\s*three|apical\s*3|three[- ]?chamber|3[- ]?ch\b|\ba3c\b|\bap3\b|apical\s*long[- ]?axis|\baplax\b)")),
]
SR_CLASS_PREFIX = "1.2.840.10008.5.1.4.1.1.88."          # DICOM Structured Report storage SOP classes


# --------------------------------------------------------------------------- 2. measurement reports (SR)
def _meanings(item) -> List[str]:
    out = []
    for tag in ("ConceptNameCodeSequence", "ConceptCodeSequence"):
        for c in getattr(item, tag, []) or []:
            if getattr(c, "CodeMeaning", None):
                out.append(str(c.CodeMeaning))
    if getattr(item, "TextValue", None):
        out.append(str(item.TextValue))
    return out


def _refs(item) -> List[str]:
    out = []
    for r in getattr(item, "ReferencedSOPSequence", []) or []:
        if getattr(r, "ReferencedSOPInstanceUID", None):
            out.append(str(r.ReferencedSOPInstanceUID))
    return out


def _walk(item, context: List[str], found: Dict[str, List[str]]) -> None:
    """Depth-first over the SR tree. An image reference takes the words of its enclosing containers and of its
    siblings in the same group (e.g. 'Image View: Apical four chamber' next to the IMAGE item)."""
    kids = list(getattr(item, "ContentSequence", []) or [])
    own = context + _meanings(item)
    for uid in _refs(item):
        found.setdefault(uid, []).extend(own)
    group = own + [m for k in kids if not getattr(k, "ContentSequence", None) for m in _meanings(k)]
    for k in kids:
        if getattr(k, "ContentSequence", None):
            _walk(k, own, found)
        else:
            for uid in _refs(k):
                found.setdefault(uid, []).extend(group)


def view_from_words(words: Iterable[str]) -> Optional[str]:
    text = " ".join(words)
    hits = {v for v, rx in VIEW_WORDS if rx.search(text)}
    return hits.pop() if len(hits) == 1 else None           # two views named together: no label


def sr_views(paths: Iterable[str]) -> pd.DataFrame:
    """SR files -> one row per referenced clip: sop_uid, view_sr (None when ambiguous), evidence."""
    import pydicom
    found: Dict[str, List[str]] = {}
    for p in paths:
        try:
            ds = pydicom.dcmread(str(p), force=True)
        except Exception:  # noqa: BLE001
            continue
        if not str(getattr(ds, "SOPClassUID", "")).startswith(SR_CLASS_PREFIX) and getattr(ds, "Modality", "") != "SR":
            continue
        _walk(ds, [], found)
    rows = [{"sop_uid": uid, "view_sr": view_from_words(w), "sr_evidence": "; ".join(dict.fromkeys(w))[:300]}
            for uid, w in found.items()]
    return pd.DataFrame(rows, columns=["sop_uid", "view_sr", "sr_evidence"])


def study_sr_files(study_dirs: Iterable[str]) -> List[str]:
    import pydicom
    out = []
    for d in study_dirs:
        for f in sorted(Path(d).rglob("*")):
            if not f.is_file():
                continue
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True, specific_tags=["SOPClassUID", "Modality"])
            except Exception:  # noqa: BLE001
                continue
            if str(getattr(ds, "SOPClassUID", "")).startswith(SR_CLASS_PREFIX) or getattr(ds, "Modality", "") == "SR":
                out.append(str(f))
    return out


def coverage(clips: pd.DataFrame, sr: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """How much each automatic source covers: decides whether clustering / hand labelling is needed."""
    c = clips.copy()
    srv = sr.dropna(subset=["view_sr"]).set_index("sop_uid").view_sr if sr is not None and len(sr) else pd.Series(dtype=object)
    c["view_sr"] = c.sop_uid.map(srv) if "sop_uid" in c else None
    c["view_tag"] = c["view_tag"] if "view_tag" in c else None
    tag, s = c.view_tag.isin(VIEWS), c.view_sr.isin(VIEWS)
    both, any_ = tag & s, tag | s
    lab = c.view_tag.where(tag).fillna(c.view_sr.where(s))
    return {"clips": int(len(c)), "studies": int(c.study_id.nunique()),
            "clips_with_tag": int(tag.sum()), "clips_with_sr": int(s.sum()), "clips_with_either": int(any_.sum()),
            "share_with_either": round(float(any_.mean()), 3) if len(c) else 0.0,
            "tag_sr_agree": int((both & (c.view_tag == c.view_sr)).sum()), "tag_sr_disagree": int((both & (c.view_tag != c.view_sr)).sum()),
            "by_view": {v: int((lab == v).sum()) for v in VIEWS},
            "studies_with_view": {v: int(c[lab == v].study_id.nunique()) for v in VIEWS}}


# --------------------------------------------------------------------------- 3. clusters
def embed_clips(model, clips: pd.DataFrame, cache, device: str = "cpu", frames: int = 4) -> np.ndarray:
    """Encoder embedding per clip: pooled 1/32 features of a few evenly spaced frames, averaged, L2-normalised."""
    import torch
    import torch.nn.functional as F
    model.eval()
    out = np.zeros((len(clips), model.head_conv[0].out_channels), np.float32)
    with torch.no_grad():
        for i, uri in enumerate(clips.clip_uri):
            fr = cache(uri)
            if fr is None:
                out[i] = np.nan
                continue
            idx = np.linspace(0, len(fr) - 1, frames).round().astype(int)
            x = torch.from_numpy(((fr[idx] - fr.mean()) / (fr.std() + 1e-6)).astype(np.float32)[:, None]).to(device)
            z, _ = model.encode(x)
            out[i] = F.normalize(F.adaptive_avg_pool2d(z, 1).flatten(1).mean(0), dim=0).cpu().numpy()
    return out


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    """k-means++ / Lloyd on rows of X (deterministic for a seed). Rows with NaN get -1."""
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(X).any(1)
    Y = X[ok]
    k = max(1, min(k, len(Y)))
    cent = [Y[rng.integers(len(Y))]]
    for _ in range(1, k):
        d = np.min(((Y[:, None] - np.array(cent)[None]) ** 2).sum(-1), axis=1)
        cent.append(Y[rng.choice(len(Y), p=d / d.sum())] if d.sum() > 0 else Y[rng.integers(len(Y))])
    C = np.array(cent)
    lab = np.zeros(len(Y), int)
    for _ in range(iters):
        new = ((Y[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        if (new == lab).all() and _ > 0:
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                C[j] = Y[lab == j].mean(0)
    out = np.full(len(X), -1)
    out[ok] = lab
    return out


def cluster_table(clips: pd.DataFrame, cluster: np.ndarray, auto_purity: float = 0.9, auto_min: int = 10) -> pd.DataFrame:
    """Per cluster: size, known views (tag / SR), and an automatic view when the known ones agree."""
    c = clips.assign(cluster=cluster)
    known = c.view_tag.where(c.view_tag.isin(VIEWS)) if "view_tag" in c else pd.Series(None, index=c.index)
    if "view_sr" in c:
        known = known.fillna(c.view_sr.where(c.view_sr.isin(VIEWS)))
    c["known"] = known
    rows = []
    for j, g in c[c.cluster >= 0].groupby("cluster"):
        k = g.known.dropna()
        top = k.value_counts()
        view, share = (top.index[0], float(top.iloc[0] / len(k))) if len(k) else (None, 0.0)
        auto = view if (len(k) >= auto_min and share >= auto_purity) else None
        rows.append({"cluster": int(j), "cluster_id": f"C{int(j) + 1:03d}", "clips": len(g), "known": len(k),
                     "known_view": view, "known_share": round(share, 3), "auto_view": auto})
    return pd.DataFrame(rows)


def export_clusters(clips: pd.DataFrame, cluster: np.ndarray, out: str, cache, labellers: List[str],
                    auto_purity: float = 0.9, auto_min: int = 10, per_cluster: int = 8, seed: int = 20261017) -> pd.DataFrame:
    """clusters.csv for all groups; img/<id>.png (8 sample clips, 2 x 4, middle frame each) and clusters.xlsx
    for the groups that need a person."""
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation

    from .views import write_png
    outp = Path(out)
    (outp / "img").mkdir(parents=True, exist_ok=True)
    t = cluster_table(clips, cluster, auto_purity, auto_min)
    rng = np.random.default_rng(seed)
    c = clips.assign(cluster=cluster)
    for _, r in t.iterrows():
        g = c[c.cluster == r.cluster]
        pick = g.iloc[rng.choice(len(g), min(per_cluster, len(g)), replace=False)]
        tiles = []
        for uri in pick.clip_uri:
            fr = cache(uri)
            tiles.append(np.zeros((112, 112), np.float32) if fr is None else fr[len(fr) // 2])
        while len(tiles) < per_cluster:
            tiles.append(np.zeros((112, 112), np.float32))
        img = np.concatenate([np.concatenate(tiles[i * 4:(i + 1) * 4], axis=1) for i in range(per_cluster // 4)], axis=0)
        write_png(outp / "img" / f"{r.cluster_id}.png", (img * 255).clip(0, 255).astype(np.uint8))
    c[["clip_uri", "study_id", "cluster"]].to_csv(outp / "clip_clusters.csv", index=False)
    t.to_csv(outp / "clusters.csv", index=False)
    todo = t[t.auto_view.isna()].reset_index(drop=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "clusters"
    ws.append(["cluster_id", "image", "clips", "hint (tagged / SR clips)", "view", "labeller", "note"])
    for i, r in todo.iterrows():
        hint = f"{r.known_view} {r.known_share:.0%} of {r.known}" if r.known else ""
        ws.append([r.cluster_id, f"img/{r.cluster_id}.png", int(r.clips), hint, None, labellers[i % len(labellers)], None])
    if len(todo):
        dv = DataValidation(type="list", formula1='"' + ",".join(VIEW_CLASSES + ["mixed"]) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"E2:E{ws.max_row}")
    hs = wb.create_sheet("how")
    for line in ("Open img/<cluster_id>.png: the middle frame of 8 clips of the group, apex at the top.",
                 "Pick the view if (nearly) all 8 are the same view: A4C, A2C, A3C or other (parasternal, subcostal, ...).",
                 "Pick 'mixed' when the 8 are not one view: those clips stay unlabelled (or go to the per-clip sheet).",
                 "The hint is what the scanner tags / measurement reports say about some clips of the group."):
        hs.append([line])
    wb.save(outp / "clusters.xlsx")
    return t


def read_clusters(d: str) -> pd.DataFrame:
    """clip_uri -> view from the clusters (automatic, or the person's pick; 'mixed' groups are left out)."""
    dp = Path(d)
    t = pd.read_csv(dp / "clusters.csv")
    cc = pd.read_csv(dp / "clip_clusters.csv")
    view = dict(zip(t.cluster, t.auto_view.where(t.auto_view.notna(), None)))
    src = {j: "auto" for j, v in view.items() if isinstance(v, str)}
    xl = dp / "clusters.xlsx"
    if xl.exists():
        s = pd.read_excel(xl, sheet_name="clusters", dtype=str)
        ids = dict(zip(t.cluster_id, t.cluster))
        for _, r in s.iterrows():
            v = str(r.get("view") or "").strip()
            if v in VIEW_CLASSES:
                view[ids[r.cluster_id]] = v
                src[ids[r.cluster_id]] = "person"
            elif v and v != "nan" and v != "mixed":
                raise ValueError(f"{r.cluster_id}: view {v!r} not in {VIEW_CLASSES + ['mixed']}")
    cc["view"] = cc.cluster.map(view)
    cc["cluster_source"] = cc.cluster.map(src)
    return cc[cc["view"].isin(VIEW_CLASSES)][["clip_uri", "study_id", "view", "cluster", "cluster_source"]]


# --------------------------------------------------------------------------- CLI
def _read(p: str) -> pd.DataFrame:
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--clips", required=True)
    c.add_argument("--studies", required=True, help="root folder of the delivered studies (SR files are found inside)")
    c.add_argument("--out", required=True)
    k = sub.add_parser("clusters")
    k.add_argument("--clips", required=True)
    k.add_argument("--sr", default=None, help="sr_views.csv from 'coverage'")
    k.add_argument("--encoder", required=True, help="pretrain.py output folder (encoder.pt)")
    k.add_argument("--k", type=int, default=60)
    k.add_argument("--labellers", default="ds1,ds2")
    k.add_argument("--cache-dir", default=None)
    k.add_argument("--device", default=None)
    k.add_argument("--out", required=True)
    r = sub.add_parser("read-clusters")
    r.add_argument("--dir", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "coverage":
        clips = _read(a.clips)
        sr = sr_views(study_sr_files([a.studies]))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        sr.to_csv(Path(a.out) / "sr_views.csv", index=False)
        rep = coverage(clips, sr)
        (Path(a.out) / "coverage.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2))
    elif a.cmd == "clusters":
        from .dataset import load_config
        from .runtime.model import build_model
        from .train import ClipCache, load_init
        cfg = load_config()
        clips = _read(a.clips)
        if a.sr:
            sr = pd.read_csv(a.sr).dropna(subset=["view_sr"])
            clips["view_sr"] = clips.sop_uid.map(sr.set_index("sop_uid").view_sr)
        model = build_model(cfg["arch"])
        load_init(model, a.encoder, print)
        device = a.device or "cpu"
        cache = ClipCache(cache_dir=a.cache_dir)
        X = embed_clips(model.to(device), clips, cache, device)
        lab = kmeans(X, a.k, seed=int(cfg["splits"]["seed"]))
        vc = cfg["views"]
        t = export_clusters(clips, lab, a.out, cache, [x.strip() for x in a.labellers.split(",") if x.strip()],
                            float(vc.get("cluster_auto_purity", 0.9)), int(vc.get("cluster_auto_min", 10)))
        print(f"{len(t)} clusters; {int(t.auto_view.notna().sum())} labelled automatically "
              f"({int(t[t.auto_view.notna()].clips.sum())} clips); {int(t.auto_view.isna().sum())} for people in clusters.xlsx")
    else:
        v = read_clusters(a.dir)
        v.to_csv(a.out, index=False)
        print(v["view"].value_counts().to_string())


if __name__ == "__main__":
    main()
