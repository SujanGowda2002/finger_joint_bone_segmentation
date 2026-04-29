import os
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# USER CONFIG
# =========================================================
PACKAGE_DIR = "SeedSet27_Training_Only_Package"
OUTPUT_DIR = "sanity_check_outputs"

SMALL_MASK_RATIO = 0.01
LARGE_MASK_RATIO = 0.60
MAX_CONNECTED_COMPONENTS = 3
MAX_HOLES = 2
NUM_GOOD_EXAMPLES = 6
NUM_BAD_EXAMPLES = 12

MASK_BIN_THRESHOLD = 20


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_manifest_csv(package_dir: Path) -> Path:
    candidates = list(package_dir.rglob("manifest.csv"))
    if not candidates:
        raise FileNotFoundError(f"Could not find manifest.csv under: {package_dir}")
    return candidates[0]


def guess_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    cols = list(df.columns)

    for c in cols:
        c_low = c.lower()
        if all(k in c_low for k in keywords):
            return c

    for c in cols:
        c_low = c.lower()
        if any(k in c_low for k in keywords):
            return c

    return None


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Automatically detect:
    - image column
    - mask column
    - joint column (if available)
    - status column (if available)
    """
    image_col = None
    mask_col = None
    joint_col = None
    status_col = None

    image_candidates = [["image"], ["img"], ["roi"], ["input"]]
    for ks in image_candidates:
        image_col = guess_column(df, ks)
        if image_col:
            break

    mask_candidates = [["mask"], ["pseudo", "mask"], ["label"], ["annotation"]]
    for ks in mask_candidates:
        mask_col = guess_column(df, ks)
        if mask_col and mask_col != image_col:
            break

    joint_candidates = [["joint"], ["joint_type"], ["type"], ["category"]]
    for ks in joint_candidates:
        joint_col = guess_column(df, ks)
        if joint_col:
            break

    status_candidates = [["status"], ["review"], ["accept"], ["decision"]]
    for ks in status_candidates:
        status_col = guess_column(df, ks)
        if status_col:
            break

    if image_col is None or mask_col is None:
        raise ValueError(
            f"Could not automatically detect image/mask columns.\n"
            f"Columns found: {list(df.columns)}\n"
            f"Please rename your manifest columns or hardcode them in the script."
        )

    return image_col, mask_col, joint_col, status_col


def resolve_path(base_dir: Path, p: str) -> Path:
    """
    Resolve a path from the manifest:
    1. absolute path
    2. relative path under package directory
    3. filename-only search fallback
    """
    p = str(p).strip()

    candidate = Path(p)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    candidate = base_dir / p
    if candidate.exists():
        return candidate

    name_only = Path(p).name
    matches = list(base_dir.rglob(name_only))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not resolve path: {p}")


def read_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def binarize_mask(mask: np.ndarray, threshold: int = MASK_BIN_THRESHOLD) -> np.ndarray:
    """
    Convert a mask into a binary 0/255 mask
    """
    return np.where(mask > threshold, 255, 0).astype(np.uint8)


def compute_mask_area(binary_mask: np.ndarray) -> int:
    """
    Compute the foreground area of a binary mask
    """
    return int(np.sum(binary_mask > 0))


def touching_border(binary_mask: np.ndarray) -> bool:
    """
    Check whether the mask touches any image border
    """
    return bool(
        np.any(binary_mask[0, :] > 0) or
        np.any(binary_mask[-1, :] > 0) or
        np.any(binary_mask[:, 0] > 0) or
        np.any(binary_mask[:, -1] > 0)
    )


def count_connected_components(binary_mask: np.ndarray) -> int:
    """
    Count connected foreground components, excluding background
    """
    num_labels, _ = cv2.connectedComponents((binary_mask > 0).astype(np.uint8))
    return max(0, num_labels - 1)


def count_holes(binary_mask: np.ndarray) -> int:
    """
    Estimate the number of internal holes in the mask
    """
    mask_bool = (binary_mask > 0).astype(np.uint8)
    h, w = mask_bool.shape

    inv = 1 - mask_bool
    flood = inv.copy()

    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 2)

    holes = np.where(flood == 1, 1, 0).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(holes)
    return max(0, num_labels - 1)


def make_overlay(image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """
    Create a color overlay of the mask on top of the grayscale image
    """
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    red_mask = np.zeros_like(base)
    red_mask[:, :, 2] = 255

    alpha = 0.35
    mask_region = binary_mask > 0

    overlay = base.copy()
    overlay[mask_region] = cv2.addWeighted(
        base[mask_region], 1 - alpha, red_mask[mask_region], alpha, 0
    )
    return overlay


def save_triplet_figure(
    image: np.ndarray,
    binary_mask: np.ndarray,
    overlay: np.ndarray,
    save_path: Path,
    title: str = ""
):
    """
    Save a 3-panel figure:
    1. original ROI image
    2. pseudo mask
    3. overlay visualization
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("ROI Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(binary_mask, cmap="gray")
    plt.title("Pseudo Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay")
    plt.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def judge_suspicious(
    area_ratio: float,
    cc_count: int,
    holes: int,
    border_touch: bool
) -> Tuple[bool, List[str]]:
    reasons = []

    if area_ratio == 0:
        reasons.append("empty_mask")
    if 0 < area_ratio < SMALL_MASK_RATIO:
        reasons.append("too_small")
    if area_ratio > LARGE_MASK_RATIO:
        reasons.append("too_large")
    if cc_count > MAX_CONNECTED_COMPONENTS:
        reasons.append("fragmented")
    if holes > MAX_HOLES:
        reasons.append("many_holes")
    if border_touch:
        reasons.append("touching_border")

    return len(reasons) > 0, reasons


def acceptance_judgement(summary: Dict) -> str:
    total = summary["total_samples"]
    suspicious = summary["suspicious_masks"]
    empty_masks = summary["empty_masks"]

    suspicious_ratio = suspicious / total if total > 0 else 1.0
    empty_ratio = empty_masks / total if total > 0 else 1.0

    if total == 0:
        return "not_usable"

    if empty_ratio > 0.10:
        return "not_usable"
    elif suspicious_ratio > 0.35:
        return "usable_after_filtering_bad_samples"
    else:
        return "usable_for_sanity_check"


def write_summary_txt(summary: Dict, save_path: Path):
    lines = []
    lines.append("Batch Validation Summary")
    lines.append("=" * 40)
    lines.append(f"Total samples: {summary['total_samples']}")
    lines.append(f"Empty masks: {summary['empty_masks']}")
    lines.append(f"Suspicious masks: {summary['suspicious_masks']}")
    lines.append(f"Good masks: {summary['good_masks']}")
    lines.append("")
    lines.append("Issue counts:")
    for k, v in summary["issue_counts"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Per joint counts:")
    for k, v in summary["per_joint_counts"].items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append(f"Acceptance judgement: {summary['acceptance_judgement']}")
    lines.append("")
    lines.append("Suggested note:")
    lines.append(
        "These pseudo masks should NOT be treated as reliable ground-truth labels. "
        "This report is only for sanity checking dataset consistency and training readiness."
    )

    save_path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================
def main():
    package_dir = Path(PACKAGE_DIR)
    if not package_dir.exists():
        raise FileNotFoundError(f"PACKAGE_DIR does not exist: {package_dir}")

    output_dir = Path(OUTPUT_DIR)
    overlays_dir = output_dir / "overlays"
    good_dir = output_dir / "good_examples"
    bad_dir = output_dir / "bad_examples"

    ensure_dir(output_dir)
    ensure_dir(overlays_dir)
    ensure_dir(good_dir)
    ensure_dir(bad_dir)

    manifest_path = find_manifest_csv(package_dir)
    print(f"[INFO] Found manifest: {manifest_path}")

    df = pd.read_csv(manifest_path)
    print(f"[INFO] Loaded manifest with {len(df)} rows")

    image_col, mask_col, joint_col, status_col = detect_columns(df)
    print(f"[INFO] Detected columns:")
    print(f"       image_col = {image_col}")
    print(f"       mask_col  = {mask_col}")
    print(f"       joint_col = {joint_col}")
    print(f"       status_col = {status_col}")

    results = []
    issue_counter = {
        "empty_mask": 0,
        "too_small": 0,
        "too_large": 0,
        "fragmented": 0,
        "many_holes": 0,
        "touching_border": 0,
    }

    for idx, row in df.iterrows():
        try:
            image_path = resolve_path(package_dir, row[image_col])
            mask_path = resolve_path(package_dir, row[mask_col])

            image = read_grayscale(image_path)
            raw_mask = read_grayscale(mask_path)
            binary_mask = binarize_mask(raw_mask)

            if image.shape != binary_mask.shape:
                binary_mask = cv2.resize(
                    binary_mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            h, w = image.shape
            img_area = h * w
            mask_area = compute_mask_area(binary_mask)
            area_ratio = mask_area / img_area if img_area > 0 else 0.0

            cc_count = count_connected_components(binary_mask)
            holes = count_holes(binary_mask)
            border_touch = touching_border(binary_mask)

            suspicious, reasons = judge_suspicious(
                area_ratio=area_ratio,
                cc_count=cc_count,
                holes=holes,
                border_touch=border_touch,
            )

            for r in reasons:
                if r in issue_counter:
                    issue_counter[r] += 1

            overlay = make_overlay(image, binary_mask)

            joint_value = str(row[joint_col]) if joint_col and pd.notna(row[joint_col]) else "unknown"
            status_value = str(row[status_col]) if status_col and pd.notna(row[status_col]) else ""

            file_stem = f"{idx:03d}_{Path(image_path).stem}"
            overlay_path = overlays_dir / f"{file_stem}_overlay.png"

            save_triplet_figure(
                image,
                binary_mask,
                overlay,
                overlay_path,
                title=f"{file_stem} | joint={joint_value}"
            )

            results.append({
                "index": idx,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "joint": joint_value,
                "status": status_value,
                "height": h,
                "width": w,
                "mask_area": mask_area,
                "area_ratio": area_ratio,
                "connected_components": cc_count,
                "holes": holes,
                "touching_border": border_touch,
                "suspicious": suspicious,
                "reasons": ";".join(reasons) if reasons else "",
                "overlay_path": str(overlay_path),
            })

        except Exception as e:
            print(f"[WARNING] Failed on row {idx}: {e}")
            results.append({
                "index": idx,
                "image_path": str(row.get(image_col, "")),
                "mask_path": str(row.get(mask_col, "")),
                "joint": str(row.get(joint_col, "unknown")) if joint_col else "unknown",
                "status": str(row.get(status_col, "")) if status_col else "",
                "height": np.nan,
                "width": np.nan,
                "mask_area": np.nan,
                "area_ratio": np.nan,
                "connected_components": np.nan,
                "holes": np.nan,
                "touching_border": np.nan,
                "suspicious": True,
                "reasons": f"read_error:{str(e)}",
                "overlay_path": "",
            })

    report_df = pd.DataFrame(results)
    report_csv = output_dir / "mask_quality_report.csv"
    report_df.to_csv(report_csv, index=False)
    print(f"[INFO] Saved report CSV: {report_csv}")

    suspicious_df = report_df[report_df["suspicious"] == True].copy()
    good_df = report_df[report_df["suspicious"] == False].copy()

    suspicious_df = suspicious_df.sort_values(
        by=["area_ratio", "connected_components", "holes"],
        ascending=[True, False, False]
    )
    good_df = good_df.sort_values(by=["area_ratio"], ascending=False)

    for _, row in good_df.head(NUM_GOOD_EXAMPLES).iterrows():
        overlay_path_str = str(row["overlay_path"]).strip()
        if overlay_path_str:
            src = Path(overlay_path_str)
            if src.exists() and src.is_file():
                dst = good_dir / src.name
                dst.write_bytes(src.read_bytes())

    for _, row in suspicious_df.head(NUM_BAD_EXAMPLES).iterrows():
        overlay_path_str = str(row["overlay_path"]).strip()
        if overlay_path_str:
            src = Path(overlay_path_str)
            if src.exists() and src.is_file():
                dst = bad_dir / src.name
                dst.write_bytes(src.read_bytes())

    per_joint_counts = report_df["joint"].value_counts(dropna=False).to_dict()

    total_samples = len(report_df)
    empty_masks = int((report_df["reasons"].fillna("").str.contains("empty_mask")).sum())
    suspicious_masks = int((report_df["suspicious"] == True).sum())
    good_masks = total_samples - suspicious_masks

    summary = {
        "total_samples": total_samples,
        "empty_masks": empty_masks,
        "suspicious_masks": suspicious_masks,
        "good_masks": good_masks,
        "issue_counts": issue_counter,
        "per_joint_counts": per_joint_counts,
        "acceptance_judgement": "",
        "notes": (
            "Pseudo masks are not verified ground-truth labels. "
            "This analysis is only for dataset sanity check and training readiness."
        )
    }

    summary["acceptance_judgement"] = acceptance_judgement(summary)

    summary_json = output_dir / "batch_validation_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_txt = output_dir / "batch_validation_summary.txt"
    write_summary_txt(summary, summary_txt)

    print(f"[INFO] Saved summary JSON: {summary_json}")
    print(f"[INFO] Saved summary TXT : {summary_txt}")

    print("\n===== FINAL SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()