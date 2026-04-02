import os
import sys
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_grayscale_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def blur_image(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)


def otsu_threshold(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def inverse_otsu_threshold(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def canny_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)


def morphological_cleanup(mask, close_kernel=(5, 5), open_kernel=(3, 3), close_iter=2, open_iter=1):
    close_k = np.ones(close_kernel, np.uint8)
    open_k = np.ones(open_kernel, np.uint8)

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=close_iter)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_k, iterations=open_iter)
    return cleaned


def fill_edges(edges, dilate_kernel=(3, 3), dilate_iter=1):
    kernel = np.ones(dilate_kernel, np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=dilate_iter)
    filled = morphological_cleanup(dilated, close_kernel=(5, 5), open_kernel=(3, 3), close_iter=2, open_iter=1)
    return filled


def combine_masks(mask_a, mask_b):
    combined = cv2.bitwise_and(mask_a, mask_b)
    return combined


def touches_border(component_mask, border_width=5):
    h, w = component_mask.shape
    border = np.zeros_like(component_mask)

    border[:border_width, :] = 255
    border[-border_width:, :] = 255
    border[:, :border_width] = 255
    border[:, -border_width:] = 255

    overlap = cv2.bitwise_and(component_mask, border)
    return np.count_nonzero(overlap) > 0


def keep_best_component(mask, min_area_ratio=0.01, max_area_ratio=0.75):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(mask)

    h, w = mask.shape
    image_area = h * w
    center_x = w / 2.0
    center_y = h / 2.0

    best_label = None
    best_score = float("-inf")

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        comp_w = stats[label, cv2.CC_STAT_WIDTH]
        comp_h = stats[label, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[label]

        area_ratio = area / image_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        component_mask = np.zeros_like(mask)
        component_mask[labels == label] = 255

        dist2 = (cx - center_x) ** 2 + (cy - center_y) ** 2
        dist_score = -dist2 / image_area

        aspect_ratio = comp_h / max(comp_w, 1)
        aspect_bonus = 0.0
        if 1.0 <= aspect_ratio <= 6.0:
            aspect_bonus = 0.15

        border_penalty = -0.5 if touches_border(component_mask, border_width=6) else 0.0

        area_score = 0.3 * area_ratio

        score = dist_score + aspect_bonus + border_penalty + area_score

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return np.zeros_like(mask)

    best_mask = np.zeros_like(mask)
    best_mask[labels == best_label] = 255
    return best_mask


def create_overlay(original_gray, binary_mask, alpha=0.35):
    original_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    red_mask = np.zeros_like(original_bgr)
    red_mask[:, :, 2] = binary_mask
    overlay = cv2.addWeighted(original_bgr, 1.0, red_mask, alpha, 0)
    return overlay


def to_bgr(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def create_summary_canvas(original, enhanced, otsu_mask, inv_otsu_mask, edges, combined, best_mask, overlay):
    original_bgr = to_bgr(original)
    enhanced_bgr = to_bgr(enhanced)
    otsu_bgr = to_bgr(otsu_mask)
    inv_otsu_bgr = to_bgr(inv_otsu_mask)
    edges_bgr = to_bgr(edges)
    combined_bgr = to_bgr(combined)
    best_bgr = to_bgr(best_mask)

    row1 = np.hstack([original_bgr, enhanced_bgr, otsu_bgr, inv_otsu_bgr])
    row2 = np.hstack([edges_bgr, combined_bgr, best_bgr, overlay])

    canvas = np.vstack([row1, row2])
    return canvas


def add_canvas_labels(canvas, tile_width, tile_height):
    labeled = canvas.copy()

    labels = [
        ("Original", 10, 25),
        ("CLAHE", tile_width + 10, 25),
        ("Otsu", 2 * tile_width + 10, 25),
        ("Inv Otsu", 3 * tile_width + 10, 25),
        ("Canny", 10, tile_height + 25),
        ("Combined", tile_width + 10, tile_height + 25),
        ("Best Mask", 2 * tile_width + 10, tile_height + 25),
        ("Overlay", 3 * tile_width + 10, tile_height + 25),
    ]

    for text, x, y in labels:
        cv2.putText(
            labeled,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return labeled


def process_single_image(image_path, output_dir):
    original = load_grayscale_image(image_path)
    enhanced = apply_clahe(original)
    blurred = blur_image(enhanced)

    otsu = otsu_threshold(blurred)
    inv_otsu = inverse_otsu_threshold(blurred)

    edges = canny_edges(blurred, low_threshold=40, high_threshold=120)
    edge_fill = fill_edges(edges, dilate_kernel=(3, 3), dilate_iter=1)

    otsu_clean = morphological_cleanup(otsu, close_kernel=(5, 5), open_kernel=(3, 3), close_iter=2, open_iter=1)
    inv_otsu_clean = morphological_cleanup(inv_otsu, close_kernel=(5, 5), open_kernel=(3, 3), close_iter=2, open_iter=1)

    combined_otsu = combine_masks(otsu_clean, edge_fill)
    combined_inv = combine_masks(inv_otsu_clean, edge_fill)

    combined_otsu = morphological_cleanup(combined_otsu, close_kernel=(5, 5), open_kernel=(3, 3), close_iter=2, open_iter=1)
    combined_inv = morphological_cleanup(combined_inv, close_kernel=(5, 5), open_kernel=(3, 3), close_iter=2, open_iter=1)

    best_otsu = keep_best_component(combined_otsu, min_area_ratio=0.01, max_area_ratio=0.60)
    best_inv = keep_best_component(combined_inv, min_area_ratio=0.01, max_area_ratio=0.60)

    if np.count_nonzero(best_otsu) >= np.count_nonzero(best_inv):
        combined = combined_otsu
        best_mask = best_otsu
    else:
        combined = combined_inv
        best_mask = best_inv

    overlay = create_overlay(original, best_mask)

    stem = Path(image_path).stem

    cv2.imwrite(str(Path(output_dir) / f"{stem}_01_original.png"), original)
    cv2.imwrite(str(Path(output_dir) / f"{stem}_02_enhanced.png"), enhanced)
    cv2.imwrite(str(Path(output_dir) / f"{stem}_03_otsu.png"), otsu_clean)
    cv2.imwrite(str(Path(output_dir) / f"{stem}_04_inv_otsu.png"), inv_otsu_clean)
    cv2.imwrite(str(Path(output_dir) / f"{stem}_05_canny.png"), edges)
    cv2.imwrite(str(Path(output_dir) / f"{stem}_06_combined.png"), combined)
    cv2.imwrite(str(Path(output_dir) / f"{stem}_07_best_mask.png"), best_mask)
    cv2.imwrite(str(Path(output_dir) / f"{stem}_08_overlay.png"), overlay)

    canvas = create_summary_canvas(
        original, enhanced, otsu_clean, inv_otsu_clean, edges, combined, best_mask, overlay
    )
    canvas = add_canvas_labels(canvas, original.shape[1], original.shape[0])
    cv2.imwrite(str(Path(output_dir) / f"{stem}_summary.png"), canvas)


def get_sample_images(image_dir, sample_size=20, seed=42):
    image_dir = Path(image_dir)
    image_files = [
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    ]

    if len(image_files) == 0:
        raise ValueError(f"No image files found in {image_dir}")

    random.seed(seed)
    sample_size = min(sample_size, len(image_files))
    return random.sample(image_files, sample_size)


def main():
    image_dir = os.path.join(PROJECT_ROOT, "data", "raw", "images")
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "classical_segmentation_demo_v2")
    sample_size = 20

    ensure_dir(output_dir)

    print(f"Reading images from: {image_dir}")
    print(f"Saving outputs to: {output_dir}")

    sample_images = get_sample_images(image_dir, sample_size=sample_size, seed=42)

    print(f"Processing {len(sample_images)} sample images...")

    for idx, image_path in enumerate(sample_images, start=1):
        print(f"[{idx}/{len(sample_images)}] Processing: {image_path.name}")
        try:
            process_single_image(image_path, output_dir)
        except Exception as e:
            print(f"Failed on {image_path.name}: {e}")

    print("\nDone.")
    print(f"Check results in: {output_dir}")


if __name__ == "__main__":
    main()