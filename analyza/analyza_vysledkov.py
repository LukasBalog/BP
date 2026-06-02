#!/usr/bin/env python3
"""
Analyza vysledkov segmentacie 
===================================================

Predpokladana struktura adresarov:
  RESULTS_DIR/
    fold_0/
      validation/
        {case}.nii.gz          
        summary.json           
    fold_1/validation/...
    ...
    fold_4/validation/...

  LABELS_DIR/
    {case}.nii.gz              

Vystupy:
  - results_by_lesion.csv            per-lezia analyza
  - summary_by_size.csv              agregat podla velkostnej kategorie
  - summary_by_fold_and_size.csv     fold x kategoria
  - nnunet_case_summary.csv          case-level metriky
  - figures/                         grafy
"""

# ============================================================
# KONFIGURACIA -- UPRAVUJ LEN TIETO CESTY
# ============================================================

RESULTS_DIR = r"C:\Bakalarka\NEW\results\nnUNetTrainerMedNeXt_B_k3__nnUNetPlans__3d_fullres"
LABELS_DIR  = r"C:\Bakalarka\NEW\data\Dataset001_SpineLesions\labelsTr"
OUTPUT_DIR  = r"C:\Bakalarka\NEW\results\analyzq"

# Hranice velkosti lezii v mm3
SIZE_BINS = {
    "small":  (0,    100),
    "medium": (100,  500),
    "large":  (500,  float("inf")),
}

# Minimalna velkost connected component (filter sumu)
MIN_LESION_VOXELS = 5

LABEL_FOREGROUND = 1
N_FOLDS = 5

# ============================================================

import json
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# Pomocne funkcie -- NIfTI
# ------------------------------------------------------------

def load_nii(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    zooms = np.array(img.header.get_zooms()[:3])
    return (data > 0.5).astype(np.uint8), zooms


def voxel_to_mm3(n_voxels: int, voxel_size: np.ndarray) -> float:
    return float(n_voxels * np.prod(voxel_size))


def get_connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    struct = ndimage.generate_binary_structure(3, 2)
    return ndimage.label(mask, structure=struct)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if inter == 0 else 0.0
    return float(2 * inter / denom)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt.astype(bool)).sum()
    fn = np.logical_and(~pred.astype(bool), gt).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(prec), float(rec)


def f1_score(prec: float, rec: float) -> float:
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def assign_size_category(vol_mm3: float) -> str:
    for cat, (lo, hi) in SIZE_BINS.items():
        if lo <= vol_mm3 < hi:
            return cat
    return list(SIZE_BINS.keys())[-1]


# ------------------------------------------------------------
# Nacitanie summary.json
# ------------------------------------------------------------

def load_summary_json(fold: int, results_dir: Path) -> Optional[dict]:
    json_path = results_dir / f"fold_{fold}" / "validation" / "summary.json"
    if not json_path.exists():
        print(f"  [!] summary.json not found: {json_path}")
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_summary_json(data: dict, fold: int) -> List[dict]:
    records = []
    for entry in data.get("metric_per_case", []):
        pred_file = entry.get("prediction_file", "")
        case_id = Path(pred_file).name.replace(".nii.gz", "")
        metrics_raw = entry.get("metrics", {})

        if "1" in metrics_raw:
            m = metrics_raw["1"]
        elif str(LABEL_FOREGROUND) in metrics_raw:
            m = metrics_raw[str(LABEL_FOREGROUND)]
        else:
            m = next(
                (v for k, v in metrics_raw.items() if k != "0"),
                {}
            )

        records.append({
            "fold":       fold,
            "case_id":    case_id,
            "dsc_nnunet": round(m.get("Dice", float("nan")), 4),
            "iou_nnunet": round(m.get("IoU",  float("nan")), 4),
            "tp":         m.get("TP",     None),
            "fp":         m.get("FP",     None),
            "fn":         m.get("FN",     None),
            "n_pred":     m.get("n_pred", None),
            "n_ref":      m.get("n_ref",  None),
        })

    return records


def load_all_summaries(results_dir: Path) -> pd.DataFrame:
    all_records = []
    for fold in range(N_FOLDS):
        data = load_summary_json(fold, results_dir)
        if data is None:
            continue
        records = parse_summary_json(data, fold)
        all_records.extend(records)

        fg = data.get("foreground_mean", {})
        dsc_mean = fg.get("Dice", float("nan"))
        print(f"   Fold {fold} -- DSC (foreground_mean): {dsc_mean:.4f}  "
              f"({len(records)} cases)")

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


# ------------------------------------------------------------
# Per-lezia analyza z NIfTI
# ------------------------------------------------------------

def find_predictions(results_dir: Path, fold: int) -> Dict[str, Path]:
    val_dir = results_dir / f"fold_{fold}" / "validation"
    if not val_dir.exists():
        print(f"  [!] fold_{fold}/validation not found: {val_dir}")
        return {}
    return {
        f.name.replace(".nii.gz", ""): f
        for f in val_dir.glob("*.nii.gz")
    }


def analyze_case(
    case_id: str, pred_path: Path, gt_path: Path, fold: int,
    case_dsc_nnunet: Optional[float] = None,
) -> List[dict]:
    if not gt_path.exists():
        print(f"  [!] GT not found: {gt_path}")
        return []

    pred, _    = load_nii(pred_path)
    gt,   vox  = load_nii(gt_path)

    if pred.shape != gt.shape:
        print(f"  [!] Shape mismatch {case_id}: pred={pred.shape} gt={gt.shape}")
        return []

    labeled_gt, n_gt = get_connected_components(gt)
    records = []

    for comp_id in range(1, n_gt + 1):
        gt_comp  = (labeled_gt == comp_id).astype(np.uint8)
        n_vox_gt = int(gt_comp.sum())

        if n_vox_gt < MIN_LESION_VOXELS:
            continue

        vol_mm3  = voxel_to_mm3(n_vox_gt, vox)
        size_cat = assign_size_category(vol_mm3)

        slices = ndimage.find_objects(gt_comp)[0]
        pad    = 3
        slices_p = tuple(
            slice(max(0, s.start - pad), min(gt.shape[i], s.stop + pad))
            for i, s in enumerate(slices)
        )
        pred_roi = pred[slices_p]
        gt_roi   = gt_comp[slices_p]

        dsc_val       = dice(pred_roi, gt_roi)
        prec, rec     = precision_recall(pred_roi, gt_roi)
        f1_val        = f1_score(prec, rec)
        detected      = int(np.logical_and(pred_roi, gt_roi).sum() > 0)

        records.append({
            "fold":            fold,
            "case_id":         case_id,
            "lesion_id":       f"{case_id}_L{comp_id:02d}",
            "vol_mm3":         round(vol_mm3, 2),
            "vol_vox":         n_vox_gt,
            "size_cat":        size_cat,
            "dsc":             round(dsc_val, 4),
            "precision":       round(prec, 4),
            "recall":          round(rec, 4),
            "f1":              round(f1_val, 4),
            "detected":        detected,
            "dsc_nnunet_case": round(case_dsc_nnunet, 4) if case_dsc_nnunet is not None else float("nan"),
        })

    return records


def run_lesion_analysis(
    results_dir: Path,
    labels_dir: Path,
    case_summaries: pd.DataFrame,
) -> pd.DataFrame:
    all_records = []

    for fold in range(N_FOLDS):
        print(f"\n-- Fold {fold} --")
        preds = find_predictions(results_dir, fold)
        if not preds:
            continue
        print(f"   Predictions found: {len(preds)}")

        for case_id, pred_path in sorted(preds.items()):
            gt_path = labels_dir / f"{case_id}.nii.gz"

            nnunet_dsc = None
            if not case_summaries.empty:
                row = case_summaries[
                    (case_summaries["fold"] == fold) &
                    (case_summaries["case_id"] == case_id)
                ]
                if not row.empty:
                    nnunet_dsc = row.iloc[0]["dsc_nnunet"]

            records = analyze_case(case_id, pred_path, gt_path, fold, nnunet_dsc)
            all_records.extend(records)

            if records:
                avg_dsc = np.mean([r["dsc"] for r in records])
                n_les   = len(records)
                nn_str  = f"  nnUNet DSC={nnunet_dsc:.3f}" if nnunet_dsc is not None else ""
                print(f"   {case_id}: {n_les} lesions | avg DSC={avg_dsc:.3f}{nn_str}")

    return pd.DataFrame(all_records)


# ------------------------------------------------------------
# Agregacia
# ------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat in list(SIZE_BINS.keys()) + ["total"]:
        sub = df if cat == "total" else df[df["size_cat"] == cat]
        if sub.empty:
            continue
        rows.append({
            "category":       cat,
            "n_lesions":      len(sub),
            "n_cases":        sub["case_id"].nunique(),
            "vol_median_mm3": round(sub["vol_mm3"].median(), 1),
            "dsc_mean":       round(sub["dsc"].mean(), 3),
            "dsc_median":     round(sub["dsc"].median(), 3),
            "dsc_std":        round(sub["dsc"].std(), 3),
            "precision":      round(sub["precision"].mean(), 3),
            "recall":         round(sub["recall"].mean(), 3),
            "f1":             round(sub["f1"].mean(), 3),
            "det_rate":       round(sub["detected"].mean(), 3),
        })
    return pd.DataFrame(rows)


def summarize_per_fold(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in range(N_FOLDS):
        for cat in list(SIZE_BINS.keys()) + ["total"]:
            sub = (df[df["fold"] == fold] if cat == "total"
                   else df[(df["fold"] == fold) & (df["size_cat"] == cat)])
            if sub.empty:
                continue
            rows.append({
                "fold":      fold,
                "size_cat":  cat,
                "n":         len(sub),
                "dsc":       round(sub["dsc"].mean(), 4),
                "precision": round(sub["precision"].mean(), 4),
                "recall":    round(sub["recall"].mean(), 4),
                "f1":        round(sub["f1"].mean(), 4),
                "det_rate":  round(sub["detected"].mean(), 4),
            })
    return pd.DataFrame(rows)


def summarize_nnunet_by_fold(case_df: pd.DataFrame) -> pd.DataFrame:
    if case_df.empty:
        return pd.DataFrame()
    rows = []
    for fold in range(N_FOLDS):
        sub = case_df[case_df["fold"] == fold]
        if sub.empty:
            continue
        rows.append({
            "fold":       fold,
            "n_cases":    len(sub),
            "dsc_mean":   round(sub["dsc_nnunet"].mean(), 4),
            "dsc_median": round(sub["dsc_nnunet"].median(), 4),
            "dsc_std":    round(sub["dsc_nnunet"].std(), 4),
            "iou_mean":   round(sub["iou_nnunet"].mean(), 4),
        })
    rows.append({
        "fold":       "total",
        "n_cases":    len(case_df),
        "dsc_mean":   round(case_df["dsc_nnunet"].mean(), 4),
        "dsc_median": round(case_df["dsc_nnunet"].median(), 4),
        "dsc_std":    round(case_df["dsc_nnunet"].std(), 4),
        "iou_mean":   round(case_df["iou_nnunet"].mean(), 4),
    })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Vizualizacia 
# ------------------------------------------------------------

COLORS = {
    "small":  "#4C9BE8",
    "medium": "#F5A623",
    "large":  "#7ED321",
    "total":  "#888888",
}


def plot_dsc_boxplot(df: pd.DataFrame, out_dir: Path):
    cats = [c for c in SIZE_BINS.keys() if c in df["size_cat"].unique()]
    data = [df[df["size_cat"] == c]["dsc"].values for c in cats]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, cat in zip(bp["boxes"], cats):
        patch.set_facecolor(COLORS[cat])
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(cats) + 1))
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylabel("Dice Similarity Coefficient (DSC)", fontsize=11)
    ax.set_title("DSC by lesion size (all folds)", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="DSC=0.5")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "dsc_boxplot_by_size.png", dpi=150)
    plt.close(fig)
    print("  [+] dsc_boxplot_by_size.png")


def plot_metrics_bar(summary: pd.DataFrame, out_dir: Path):
    rows = summary[summary["category"] != "total"]
    metrics = ["dsc_mean", "precision", "recall", "f1"]
    metric_labels = ["DSC (mean)", "Precision", "Recall", "F1"]
    metric_colors = ["#4C9BE8", "#F5A623", "#7ED321", "#BD10E0"]
    x = np.arange(len(rows))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
        vals = rows[metric].values
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(rows["category"], fontsize=10)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title("Metrics by lesion size", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_bar_by_size.png", dpi=150)
    plt.close(fig)
    print("  [+] metrics_bar_by_size.png")


def plot_dsc_per_fold(fold_summary: pd.DataFrame, out_dir: Path):
    cats = [c for c in SIZE_BINS.keys()]
    fig, ax = plt.subplots(figsize=(9, 5))
    for cat in cats:
        sub = fold_summary[fold_summary["size_cat"] == cat].sort_values("fold")
        if sub.empty:
            continue
        ax.plot(sub["fold"], sub["dsc"], marker="o", label=cat,
                color=COLORS[cat], linewidth=2, markersize=7)

    ax.set_xticks(range(N_FOLDS))
    ax.set_xticklabels([f"Fold {i}" for i in range(N_FOLDS)], fontsize=10)
    ax.set_ylabel("DSC (mean per lesion)", fontsize=11)
    ax.set_title("DSC by lesion size and fold", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "dsc_per_fold.png", dpi=150)
    plt.close(fig)
    print("  [+] dsc_per_fold.png")


def plot_nnunet_dsc_per_fold(case_df: pd.DataFrame, out_dir: Path):
    if case_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    fold_data = [
        case_df[case_df["fold"] == f]["dsc_nnunet"].dropna().values
        for f in range(N_FOLDS)
    ]
    bp = ax.boxplot(fold_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    fold_colors = ["#4C9BE8", "#F5A623", "#7ED321", "#BD10E0", "#E84C4C"]
    for patch, color in zip(bp["boxes"], fold_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([f"Fold {i}" for i in range(N_FOLDS)], fontsize=10)
    ax.set_ylabel("DSC (case-level)", fontsize=11)
    ax.set_title("Case-level DSC by fold", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)

    overall_mean = case_df["dsc_nnunet"].mean()
    ax.axhline(overall_mean, color="red", linestyle="--", linewidth=1.2,
               label=f"Mean = {overall_mean:.3f}")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "dsc_per_fold_caselevel.png", dpi=150)
    plt.close(fig)
    print("  [+] dsc_per_fold_caselevel.png")


def plot_nnunet_vs_lesion_dsc(
    df_lesion: pd.DataFrame,
    case_df: pd.DataFrame,
    out_dir: Path,
):
    if case_df.empty or df_lesion.empty:
        return

    avg_lesion = (
        df_lesion.groupby(["fold", "case_id"])["dsc"]
        .mean()
        .reset_index()
        .rename(columns={"dsc": "dsc_lesion_avg"})
    )
    merged = pd.merge(case_df, avg_lesion, on=["fold", "case_id"], how="inner")
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    fold_colors = ["#4C9BE8", "#F5A623", "#7ED321", "#BD10E0", "#E84C4C"]
    for fold in range(N_FOLDS):
        sub = merged[merged["fold"] == fold]
        ax.scatter(sub["dsc_nnunet"], sub["dsc_lesion_avg"],
                   color=fold_colors[fold], alpha=0.75, s=45,
                   label=f"Fold {fold}", edgecolors="none")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="y = x")
    ax.set_xlabel("DSC case-level", fontsize=11)
    ax.set_ylabel("DSC per-lesion (mean)", fontsize=11)
    ax.set_title("Case-level vs per-lesion DSC", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "caselevel_vs_lesion_dsc_scatter.png", dpi=150)
    plt.close(fig)
    print("  [+] caselevel_vs_lesion_dsc_scatter.png")


def plot_detection_rate(summary: pd.DataFrame, out_dir: Path):
    rows = summary[summary["category"] != "total"]
    fig, ax = plt.subplots(figsize=(7, 4))
    cats   = rows["category"].tolist()
    rates  = rows["det_rate"].tolist()
    colors = [COLORS[c] for c in cats]
    bars   = ax.bar(cats, rates, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Detection rate", fontsize=11)
    ax.set_title("Detection rate by lesion size", fontsize=13)
    ax.set_ylim(0, 1.2)
    ax.axhline(1.0, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "detection_rate_by_size.png", dpi=150)
    plt.close(fig)
    print("  [+] detection_rate_by_size.png")


def plot_volume_histogram(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    vmin = max(df["vol_mm3"].min(), 1)
    vmax = df["vol_mm3"].max() + 1
    bins = np.logspace(np.log10(vmin), np.log10(vmax), 40)
    for cat, (lo, hi) in SIZE_BINS.items():
        sub = df[(df["vol_mm3"] >= lo) & (df["vol_mm3"] < hi)]["vol_mm3"]
        if sub.empty:
            continue
        ax.hist(sub, bins=bins, color=COLORS[cat], alpha=0.75,
                label=cat, edgecolor="white")
    for _, (lo, hi) in list(SIZE_BINS.items())[:-1]:
        if hi < float("inf"):
            ax.axvline(hi, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Lesion volume (mm3, log scale)", fontsize=11)
    ax.set_ylabel("Number of lesions", fontsize=11)
    ax.set_title("Lesion volume distribution (ground truth)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "volume_histogram.png", dpi=150)
    plt.close(fig)
    print("  [+] volume_histogram.png")


def plot_scatter_vol_dsc(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cat in SIZE_BINS.keys():
        sub = df[df["size_cat"] == cat]
        if sub.empty:
            continue
        ax.scatter(sub["vol_mm3"], sub["dsc"], color=COLORS[cat],
                   alpha=0.5, s=25, label=cat, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("Lesion volume (mm3, log scale)", fontsize=11)
    ax.set_ylabel("DSC", fontsize=11)
    ax.set_title("DSC vs lesion volume", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    try:
        log_vol = np.log10(df["vol_mm3"].clip(lower=1))
        z = np.polyfit(log_vol, df["dsc"], 1)
        x_line = np.logspace(log_vol.min(), log_vol.max(), 100)
        ax.plot(x_line, np.poly1d(z)(np.log10(x_line)),
                "k--", linewidth=1.5, alpha=0.6, label="trend")
        ax.legend(fontsize=9)
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_vol_dsc.png", dpi=150)
    plt.close(fig)
    print("  [+] scatter_vol_dsc.png")


# ------------------------------------------------------------
# Vypis
# ------------------------------------------------------------

def print_table(title: str, df: pd.DataFrame):
    sep = "-" * 110
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(df.to_string(index=False))
    print(sep)


# ------------------------------------------------------------
# Hlavny tok
# ------------------------------------------------------------

def main():
    results_dir = Path(RESULTS_DIR)
    labels_dir  = Path(LABELS_DIR)
    output_dir  = Path(OUTPUT_DIR)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Lesion segmentation analysis")
    print("=" * 65)
    print(f"  Results: {results_dir}")
    print(f"  GT:      {labels_dir}")
    print(f"  Output:  {output_dir}")

    # 1. Nacitaj summary.json
    print("\n[1/3] Loading summary.json...")
    case_df = load_all_summaries(results_dir)

    if not case_df.empty:
        nnunet_fold_summary = summarize_nnunet_by_fold(case_df)
        print_table("Case-level DSC by fold", nnunet_fold_summary)
        case_df.to_csv(output_dir / "nnunet_case_summary.csv", index=False)
        nnunet_fold_summary.to_csv(output_dir / "nnunet_fold_summary.csv", index=False)
        print(f"\n  [+] nnunet_case_summary.csv")
        print(f"  [+] nnunet_fold_summary.csv")
    else:
        print("  [!] summary.json not found in any fold.")

    # 2. Per-lezia analyza
    print("\n[2/3] Per-lesion analysis (connected components)...")
    df = run_lesion_analysis(results_dir, labels_dir, case_df)

    if df.empty:
        print("\n[!] No data. Check RESULTS_DIR and LABELS_DIR.")
        return

    print(f"\n  Total lesions: {len(df)}")
    for cat, cnt in df["size_cat"].value_counts().items():
        print(f"    {cat}: {cnt}")

    df.to_csv(output_dir / "results_by_lesion.csv", index=False)
    print(f"  [+] results_by_lesion.csv")

    summary   = summarize(df)
    fold_summ = summarize_per_fold(df)
    summary.to_csv(output_dir / "summary_by_size.csv", index=False)
    fold_summ.to_csv(output_dir / "summary_by_fold_and_size.csv", index=False)
    print(f"  [+] summary_by_size.csv")
    print(f"  [+] summary_by_fold_and_size.csv")

    print_table("Per-lesion metrics by size category", summary)

    # 3. Grafy
    print("\n[3/3] Generating figures...")
    plot_dsc_boxplot(df, figures_dir)
    plot_metrics_bar(summary, figures_dir)
    plot_dsc_per_fold(fold_summ, figures_dir)
    plot_detection_rate(summary, figures_dir)
    plot_volume_histogram(df, figures_dir)
    plot_scatter_vol_dsc(df, figures_dir)

    if not case_df.empty:
        plot_nnunet_dsc_per_fold(case_df, figures_dir)
        plot_nnunet_vs_lesion_dsc(df, case_df, figures_dir)

    print("\n" + "=" * 65)
    print(f"  DONE!  Output in: {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
