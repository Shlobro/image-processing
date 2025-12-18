#!/usr/bin/env python3
"""
Exercise 1 - Digital Archive Document Extraction
Student: Shlomo Edelstien (ID: 318751625)

This module implements a full pipeline for detecting the straight lines that
separate multiple scanned documents, visualising those lines, and cropping each
document into a standalone image.

Constraints:
    * OpenCV is only used for loading images and converting them to grayscale.
    * All other processing (filters, Canny, Hough, drawing) is implemented
      manually or with standard Python / NumPy / Pandas / Pillow components.

Running the script (from the repository root):
    python targil-bait-1/Main_318751625.py \
        --input-dir targil-bait-1 \
        --output-dir targil-bait-1

Outputs created in --output-dir:
    lines_data.csv           -> Detected line segments for every image.
    annotated_images/        -> Input images with detected lines overlayed.
    cropped_documents/       -> Cropped regions bounded by the detected grid.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw


# =============================================================================
# Utility data structures
# =============================================================================


@dataclass
class LineSegment:
    """Represents an infinite Hough line clipped to the image boundaries."""

    rho: float
    theta: float
    votes: int
    p1: Tuple[float, float]
    p2: Tuple[float, float]

    def as_record(self, filename: str) -> dict:
        """Return a serialisable view for CSV export."""
        return {
            "filename": filename,
            "x1": int(round(self.p1[0])),
            "y1": int(round(self.p1[1])),
            "x2": int(round(self.p2[0])),
            "y2": int(round(self.p2[1])),
        }

    def angle_degrees(self) -> float:
        return math.degrees(self.theta)

    def orientation(self, tolerance_deg: float) -> str | None:
        """Classify the line as vertical/horizontal if within tolerance."""
        angle = self.angle_degrees()
        angle = angle % 180.0
        if min(abs(angle), abs(angle - 180.0)) <= tolerance_deg:
            return "vertical"
        if abs(angle - 90.0) <= tolerance_deg:
            return "horizontal"
        return None

    def representative_position(self, orientation: str) -> float:
        if orientation == "vertical":
            return 0.5 * (self.p1[0] + self.p2[0])
        if orientation == "horizontal":
            return 0.5 * (self.p1[1] + self.p2[1])
        raise ValueError(f"Unsupported orientation {orientation}")

    def length(self) -> float:
        return math.hypot(self.p1[0] - self.p2[0], self.p1[1] - self.p2[1])


# =============================================================================
# Image processing primitives (filters, gradients, Canny, Hough)
# =============================================================================


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a normalised 2D Gaussian kernel."""
    if size % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd")
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Vectorised 2D convolution with reflected padding."""
    kernel_flipped = np.flipud(np.fliplr(kernel))
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    windows = sliding_window_view(padded, kernel.shape)
    # einsum collapses the kernel dimensions to produce the convolution result
    return np.einsum("ijkl,kl->ij", windows, kernel_flipped, optimize=True)


def gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve2d(image, kernel)


def sobel_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Sobel gradients along X and Y axes."""
    sobel_x = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32
    )
    sobel_y = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32
    )
    gx = convolve2d(image, sobel_x)
    gy = convolve2d(image, sobel_y)
    return gx, gy


def non_maximum_suppression(
    magnitude: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Suppress non-maximum pixels along the gradient direction."""
    direction = direction % 180
    h, w = magnitude.shape
    output = np.zeros_like(magnitude, dtype=np.float32)
    angle = direction

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = 0.0
            r = 0.0
            # 0 degrees
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            # 45 degrees
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            # 90 degrees
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            # 135 degrees
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                output[i, j] = magnitude[i, j]

    return output


def double_threshold(
    image: np.ndarray, low_ratio: float, high_ratio: float
) -> Tuple[np.ndarray, float, float]:
    """Apply double thresholding for the Canny edge detector."""
    high_threshold = image.max() * high_ratio
    low_threshold = high_threshold * low_ratio
    strong = np.uint8(255)
    weak = np.uint8(75)

    result = np.zeros_like(image, dtype=np.uint8)
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result, weak, strong


def hysteresis(edges: np.ndarray, weak: int, strong: int) -> np.ndarray:
    """Connect weak pixels that are connected to strong pixels."""
    output = edges.copy()
    h, w = output.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if output[i, j] == weak:
                if np.any(output[i - 1 : i + 2, j - 1 : j + 2] == strong):
                    output[i, j] = strong
                else:
                    output[i, j] = 0
    output[output != strong] = 0
    return output


def canny_edges(
    image: np.ndarray,
    blur_size: int = 5,
    blur_sigma: float = 1.4,
    low_ratio: float = 0.08,
    high_ratio: float = 0.2,
) -> np.ndarray:
    """Custom Canny edge detector implementation."""
    smoothed = gaussian_blur(image, blur_size, blur_sigma)
    gx, gy = sobel_gradients(smoothed)
    magnitude = np.hypot(gx, gy)
    direction = np.rad2deg(np.arctan2(gy, gx))
    suppressed = non_maximum_suppression(magnitude, direction)
    thresholded, weak, strong = double_threshold(suppressed, low_ratio, high_ratio)
    edges = hysteresis(thresholded, weak, strong)
    return edges


def hough_transform(
    edge_map: np.ndarray,
    rho_step: float,
    theta_step: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard (rho, theta) Hough transform accumulator."""
    rows, cols = edge_map.shape
    diag_len = int(math.hypot(rows, cols))
    rhos = np.arange(-diag_len, diag_len + rho_step, rho_step)
    thetas = np.arange(0.0, math.pi, theta_step)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    y_idxs, x_idxs = np.nonzero(edge_map)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    rho_min = rhos[0]

    theta_indices = np.arange(len(thetas))

    for x, y in zip(x_idxs, y_idxs):
        rho_values = x * cos_t + y * sin_t
        rho_indices = np.round((rho_values - rho_min) / rho_step).astype(int)
        rho_indices = np.clip(rho_indices, 0, len(rhos) - 1)
        np.add.at(accumulator, (rho_indices, theta_indices), 1)

    return accumulator, rhos, thetas


def find_hough_peaks(
    accumulator: np.ndarray,
    num_peaks: int,
    threshold: int,
    neighborhood_size: Tuple[int, int],
) -> List[Tuple[int, int, int]]:
    """Locate the strongest peaks in the accumulator."""
    acc = accumulator.copy()
    peaks: List[Tuple[int, int, int]] = []
    n_rho, n_theta = neighborhood_size

    for _ in range(num_peaks):
        idx = np.argmax(acc)
        value = acc.flat[idx]
        if value < threshold:
            break
        rho_idx, theta_idx = np.unravel_index(idx, acc.shape)
        peaks.append((rho_idx, theta_idx, int(value)))

        rho_min = max(0, rho_idx - n_rho // 2)
        rho_max = min(acc.shape[0], rho_idx + n_rho // 2 + 1)
        theta_min = max(0, theta_idx - n_theta // 2)
        theta_max = min(acc.shape[1], theta_idx + n_theta // 2 + 1)
        acc[rho_min:rho_max, theta_min:theta_max] = 0

    return peaks


def line_to_segment(
    rho: float, theta: float, width: int, height: int
) -> Tuple[Tuple[float, float], Tuple[float, float]] | None:
    """Convert an infinite Hough line to segment endpoints inside the image."""
    points: List[Tuple[float, float]] = []
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)

    # Intersections with x = 0 and x = width - 1
    if abs(sin_t) > 1e-6:
        y0 = rho / sin_t
        if 0 <= y0 <= height - 1:
            points.append((0.0, y0))
        y1 = (rho - (width - 1) * cos_t) / sin_t
        if 0 <= y1 <= height - 1:
            points.append((width - 1, y1))

    # Intersections with y = 0 and y = height - 1
    if abs(cos_t) > 1e-6:
        x0 = rho / cos_t
        if 0 <= x0 <= width - 1:
            points.append((x0, 0.0))
        x1 = (rho - (height - 1) * sin_t) / cos_t
        if 0 <= x1 <= width - 1:
            points.append((x1, height - 1))

    if len(points) < 2:
        return None

    # Deduplicate very close points
    unique_points: List[Tuple[float, float]] = []
    for pt in points:
        if all(math.hypot(pt[0] - q[0], pt[1] - q[1]) > 1.0 for q in unique_points):
            unique_points.append(pt)

    if len(unique_points) < 2:
        return None
    if len(unique_points) == 2:
        return unique_points[0], unique_points[1]

    # More than two valid intersections -> keep the farthest pair
    max_pair = max(
        combinations(unique_points, 2),
        key=lambda pair: math.hypot(pair[0][0] - pair[1][0], pair[0][1] - pair[1][1]),
    )
    return max_pair


def detect_line_segments(
    edge_map: np.ndarray,
    width: int,
    height: int,
    rho_step: float,
    theta_step: float,
    threshold: int,
    max_peaks: int,
    neighborhood_size: Tuple[int, int],
    min_length: float,
) -> List[LineSegment]:
    """Detect line segments using the custom Hough transform."""
    accumulator, rhos, thetas = hough_transform(edge_map, rho_step, theta_step)
    peaks = find_hough_peaks(accumulator, max_peaks, threshold, neighborhood_size)

    segments: List[LineSegment] = []
    for rho_idx, theta_idx, votes in peaks:
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        endpoints = line_to_segment(rho, theta, width, height)
        if endpoints is None:
            continue
        segment = LineSegment(rho, theta, votes, endpoints[0], endpoints[1])
        if segment.length() < min_length:
            continue
        segments.append(segment)

    return segments


# =============================================================================
# Line post-processing, grid estimation, cropping helpers
# =============================================================================


def merge_similar_lines(
    lines: Sequence[LineSegment],
    orientation: str,
    tolerance: float,
) -> List[LineSegment]:
    """Merge duplicate lines that reside within the given tolerance."""
    if not lines:
        return []
    sorted_lines = sorted(
        lines,
        key=lambda ln: ln.representative_position(orientation),
    )
    merged: List[LineSegment] = [sorted_lines[0]]
    for line in sorted_lines[1:]:
        current_pos = line.representative_position(orientation)
        last_pos = merged[-1].representative_position(orientation)
        if abs(current_pos - last_pos) > tolerance:
            merged.append(line)
        else:
            # Keep the line with more votes (sharper detection)
            if line.votes > merged[-1].votes:
                merged[-1] = line
    return merged


def ensure_boundaries(
    positions: List[float], limit: float, margin: float = 0.0
) -> List[float]:
    """Ensure the first/last positions cover the image bounds."""
    augmented = list(positions)
    if not augmented or augmented[0] > margin:
        augmented = [margin] + augmented
    if augmented[-1] < limit - margin:
        augmented.append(limit)
    return augmented


def extract_positions(
    lines: Sequence[LineSegment],
    orientation: str,
    tolerance: float,
    limit: float,
) -> List[float]:
    merged = merge_similar_lines(lines, orientation, tolerance)
    positions = [
        ln.representative_position(orientation) for ln in merged
    ]
    positions = sorted(positions)
    positions = ensure_boundaries(positions, limit)
    return positions


def build_bounding_boxes(
    vertical_positions: Sequence[float],
    horizontal_positions: Sequence[float],
    width: int,
    height: int,
    min_size: int = 50,
) -> List[Tuple[int, int, int, int]]:
    """Create document bounding boxes from the grid intersections."""
    boxes: List[Tuple[int, int, int, int]] = []
    for col in range(len(vertical_positions) - 1):
        x1 = int(round(vertical_positions[col]))
        x2 = int(round(vertical_positions[col + 1]))
        if x2 - x1 < min_size:
            continue
        for row in range(len(horizontal_positions) - 1):
            y1 = int(round(horizontal_positions[row]))
            y2 = int(round(horizontal_positions[row + 1]))
            if y2 - y1 < min_size:
                continue
            boxes.append(
                (
                    max(0, min(x1, width - 1)),
                    max(0, min(y1, height - 1)),
                    max(0, min(x2, width)),
                    max(0, min(y2, height)),
                )
            )
    return boxes


def draw_line_overlays(
    image_rgb: np.ndarray, lines: Sequence[LineSegment]
) -> Image.Image:
    """Draw detected lines on top of the image using Pillow."""
    annotated = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(annotated)
    for line in lines:
        draw.line(
            [tuple(map(int, line.p1)), tuple(map(int, line.p2))],
            fill=(255, 0, 0),
            width=5,
        )
    return annotated


def save_crops(
    image_rgb: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    base_name: str,
    output_dir: Path,
) -> List[Path]:
    """Save cropped document regions."""
    output_paths: List[Path] = []
    if not boxes:
        return output_paths
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_image = Image.fromarray(crop)
        path = output_dir / f"{base_name}_doc_{idx}.png"
        crop_image.save(path)
        output_paths.append(path)
    return output_paths


# =============================================================================
# Orchestrator
# =============================================================================


@dataclass
class PipelineConfig:
    blur_size: int = 5
    blur_sigma: float = 1.8
    canny_low_ratio: float = 0.1
    canny_high_ratio: float = 0.3
    rho_step: float = 1.0
    theta_step: float = math.radians(1.0)
    hough_threshold: int = 320
    hough_peaks: int = 25
    hough_neighborhood: Tuple[int, int] = (35, 35)
    orientation_tolerance_deg: float = 4.0
    merge_tolerance_px: float = 60.0
    min_box_size_px: int = 120
    min_line_length_px: float = 400.0


class DocumentBoundaryDetector:
    """Runs the full detection / extraction pipeline."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: PipelineConfig | None = None,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.annotated_dir = output_dir / "annotated_images"
        self.cropped_dir = output_dir / "cropped_documents"
        self.csv_path = output_dir / "lines_data.csv"
        self.config = config or PipelineConfig()

    def run(self) -> None:
        image_paths = self._gather_images()
        line_records: List[dict] = []
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.cropped_dir.mkdir(parents=True, exist_ok=True)

        for path in image_paths:
            print(f"[INFO] Processing {path.name}")
            color_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if color_bgr is None:
                print(f"[WARN] Skipping {path.name}: unable to load.")
                continue
            gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
            image_rgb = color_bgr[:, :, ::-1]
            gray_float = gray.astype(np.float32)

            edges = canny_edges(
                gray_float,
                blur_size=self.config.blur_size,
                blur_sigma=self.config.blur_sigma,
                low_ratio=self.config.canny_low_ratio,
                high_ratio=self.config.canny_high_ratio,
            )
            height, width = edges.shape

            segments = detect_line_segments(
                edge_map=edges > 0,
                width=width,
                height=height,
                rho_step=self.config.rho_step,
                theta_step=self.config.theta_step,
                threshold=self.config.hough_threshold,
                max_peaks=self.config.hough_peaks,
                neighborhood_size=self.config.hough_neighborhood,
                min_length=self.config.min_line_length_px,
            )

            segments = [
                seg
                for seg in segments
                if seg.orientation(self.config.orientation_tolerance_deg) is not None
            ]

            vertical_segments = [
                seg
                for seg in segments
                if seg.orientation(self.config.orientation_tolerance_deg) == "vertical"
            ]
            horizontal_segments = [
                seg
                for seg in segments
                if seg.orientation(self.config.orientation_tolerance_deg) == "horizontal"
            ]

            vertical_positions = extract_positions(
                vertical_segments,
                orientation="vertical",
                tolerance=self.config.merge_tolerance_px,
                limit=float(width),
            )
            horizontal_positions = extract_positions(
                horizontal_segments,
                orientation="horizontal",
                tolerance=self.config.merge_tolerance_px,
                limit=float(height),
            )

            boxes = build_bounding_boxes(
                vertical_positions,
                horizontal_positions,
                width=width,
                height=height,
                min_size=self.config.min_box_size_px,
            )

            annotated = draw_line_overlays(
                image_rgb, vertical_segments + horizontal_segments
            )
            annotated_path = self.annotated_dir / f"{path.stem}_annotated.jpg"
            annotated.save(annotated_path, quality=95)

            saved_crops = save_crops(
                image_rgb, boxes, path.stem, self.cropped_dir
            )

            line_records.extend(
                [seg.as_record(path.name) for seg in vertical_segments + horizontal_segments]
            )

            print(
                f"       Lines: {len(vertical_segments)} vertical, "
                f"{len(horizontal_segments)} horizontal | "
                f"Crops saved: {len(saved_crops)}"
            )

        if line_records:
            df = pd.DataFrame(line_records, columns=["filename", "x1", "y1", "x2", "y2"])
            df.to_csv(self.csv_path, index=False)
            print(f"[INFO] Saved CSV -> {self.csv_path}")
        else:
            print("[WARN] No lines detected; CSV not written.")

    def _gather_images(self) -> List[Path]:
        supported_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        paths = [
            p
            for p in sorted(self.input_dir.iterdir())
            if p.suffix.lower() in supported_ext
        ]
        if not paths:
            print(f"[WARN] No image files found inside {self.input_dir}")
        return paths


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect document boundaries using custom Canny + Hough pipeline."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing the scanned input images (default: script folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory that will contain lines_data.csv and generated folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_dir = args.input_dir or script_dir
    output_dir = args.output_dir or script_dir
    detector = DocumentBoundaryDetector(input_dir=input_dir, output_dir=output_dir)
    detector.run()


if __name__ == "__main__":
    main()
