import cv2
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any


def _normalize_vector(dx: float, dy: float) -> Tuple[float, float]:
    norm = float(np.hypot(dx, dy))
    if norm <= 1e-9:
        return 0.0, 0.0
    return dx / norm, dy / norm


def _line_abc_from_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float, float]:
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return float(a), float(b), float(c)


def _line_intersection_abc(
    a1: float, b1: float, c1: float, a2: float, b2: float, c2: float, *, det_eps: float = 1e-9
) -> Optional[Tuple[float, float]]:
    det = a1 * b2 - a2 * b1
    if abs(det) <= det_eps:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return float(x), float(y)


def _line_rect_intersections_abc(
    a: float, b: float, c: float, width: int, height: int, *, tol: float = 1e-6
) -> List[Tuple[float, float]]:
    x_min, x_max = 0.0, float(width - 1)
    y_min, y_max = 0.0, float(height - 1)

    points: List[Tuple[float, float]] = []

    if abs(b) > tol:
        # x = 0
        y = (-c - a * x_min) / b
        if y_min - tol <= y <= y_max + tol:
            points.append((x_min, y))
        # x = width-1
        y = (-c - a * x_max) / b
        if y_min - tol <= y <= y_max + tol:
            points.append((x_max, y))

    if abs(a) > tol:
        # y = 0
        x = (-c - b * y_min) / a
        if x_min - tol <= x <= x_max + tol:
            points.append((x, y_min))
        # y = height-1
        x = (-c - b * y_max) / a
        if x_min - tol <= x <= x_max + tol:
            points.append((x, y_max))

    unique: List[Tuple[float, float]] = []
    for x, y in points:
        if any(abs(x - ux) <= 0.75 and abs(y - uy) <= 0.75 for ux, uy in unique):
            continue
        unique.append((float(x), float(y)))

    return unique


def _dedupe_fitted_lines(df: pd.DataFrame, *, pos_tol_px: float = 4.0) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["length"] = np.hypot(df["x2"] - df["x1"], df["y2"] - df["y1"])

    def _key(row: pd.Series) -> Tuple[str, int]:
        if row["orientation"] == "vertical":
            x_mid = 0.5 * (float(row["x1"]) + float(row["x2"]))
            return ("vertical", int(round(x_mid / pos_tol_px)))
        y_mid = 0.5 * (float(row["y1"]) + float(row["y2"]))
        return ("horizontal", int(round(y_mid / pos_tol_px)))

    df["_cluster_key"] = df.apply(_key, axis=1)
    df = df.sort_values(["votes", "length"], ascending=[False, False]).drop_duplicates("_cluster_key", keep="first")
    return df.drop(columns=["_cluster_key", "length"]).reset_index(drop=True)


def _smooth_1d(arr: np.ndarray, k: int = 31) -> np.ndarray:
    if k <= 1:
        return arr.astype(np.float32)
    kernel = np.ones(int(k), dtype=np.float32) / float(k)
    return np.convolve(arr.astype(np.float32), kernel, mode="same")


def _find_true_runs(mask: np.ndarray, *, min_len: int) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    seq = mask.tolist() + [False]  # sentinel False to flush run
    for idx, val in enumerate(seq):
        if val and start is None:
            start = idx
        elif (not val) and start is not None:
            end = idx - 1
            if end - start + 1 >= int(min_len):
                runs.append((int(start), int(end)))
            start = None
    return runs


def _gap_centers_from_doc_coverage(
    doc_mask: np.ndarray,
    *,
    axis: int,
    coverage_thresh: float,
    min_width: int,
    exclude_edges: bool = True,
) -> List[int]:
    coverage = doc_mask.mean(axis=axis).astype(np.float32)
    gap_mask = coverage <= float(coverage_thresh)
    runs = _find_true_runs(gap_mask, min_len=int(min_width))
    if exclude_edges:
        limit = int(doc_mask.shape[axis ^ 1]) - 1
        runs = [(a, b) for (a, b) in runs if a > 0 and b < limit]
    return [int(round(0.5 * (a + b))) for a, b in runs]


def _snap_positions_to_centers(positions: List[int], centers: List[int], *, max_dist: int) -> List[int]:
    if not positions or not centers:
        return []
    centers_sorted = sorted(int(c) for c in centers)
    snapped: List[int] = []
    for pos in positions:
        pos_i = int(pos)
        best = min(centers_sorted, key=lambda c: abs(c - pos_i))
        if abs(best - pos_i) <= int(max_dist):
            snapped.append(int(best))
    return sorted(set(snapped))


def _dedupe_positions(positions: List[int], *, tol_px: int) -> List[int]:
    if not positions:
        return []
    out: List[int] = []
    for p in sorted(int(x) for x in positions):
        if not out or abs(p - out[-1]) > int(tol_px):
            out.append(p)
    return out


def _refine_positions_to_background_runs(
    positions: List[int],
    *,
    bg_mask: np.ndarray,
    axis: str,
    search_px: int,
    strip_half: int,
    ratio_thresh: float,
    min_keep_run: int,
) -> List[int]:
    """
    For each initial x/y position, search locally for the position that yields the
    *longest* continuous background run along the orthogonal dimension.
    """
    if not positions:
        return []

    height, width = bg_mask.shape[:2]
    axis_norm = axis.lower().strip()
    if axis_norm not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")

    refined: List[int] = []
    for pos in positions:
        pos_i = int(pos)
        if axis_norm == "x":
            lo = max(0, pos_i - int(search_px))
            hi = min(width - 1, pos_i + int(search_px))
        else:
            lo = max(0, pos_i - int(search_px))
            hi = min(height - 1, pos_i + int(search_px))

        best_pos = pos_i
        best_len = -1
        best_dist = 10**9

        for p in range(lo, hi + 1):
            if axis_norm == "x":
                x0 = max(0, p - int(strip_half))
                x1 = min(width, p + int(strip_half) + 1)
                band = bg_mask[:, x0:x1].astype(np.float32)
                run_mask = (band.mean(axis=1) >= float(ratio_thresh))
            else:
                y0 = max(0, p - int(strip_half))
                y1 = min(height, p + int(strip_half) + 1)
                band = bg_mask[y0:y1, :].astype(np.float32)
                run_mask = (band.mean(axis=0) >= float(ratio_thresh))

            runs = _find_true_runs(run_mask, min_len=1)
            run_len = max((b - a + 1) for a, b in runs) if runs else 0
            dist = abs(p - pos_i)

            if (run_len > best_len) or (run_len == best_len and dist < best_dist):
                best_len = int(run_len)
                best_pos = int(p)
                best_dist = int(dist)

        if best_len >= int(min_keep_run):
            refined.append(int(best_pos))

    return refined


def _separator_background_mask(
    gray: np.ndarray,
    *,
    edge_image: Optional[np.ndarray] = None,
    white_thresh: int = 238,
    grad_thresh: float = 12.0,
    blur_ksize: int = 5,
) -> np.ndarray:
    """
    Returns a boolean mask for "safe separator background" (the whitespace between documents).
    This is intentionally conservative to prevent drawing on documents.
    """
    if gray.ndim != 2:
        raise ValueError("gray must be a single-channel image")

    height, width = gray.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((height, width), dtype=bool)

    k_blur = int(blur_ksize)
    if k_blur % 2 == 0:
        k_blur += 1
    if k_blur < 3:
        k_blur = 3

    blur = cv2.GaussianBlur(gray, (k_blur, k_blur), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    edge_barrier = (edge_image > 0) if (edge_image is not None and edge_image.size) else None

    seed_bg = (gray >= int(white_thresh)) & (grad <= float(grad_thresh))
    if edge_barrier is not None:
        seed_bg = seed_bg & (~edge_barrier)

    # Allow stained/background regions (often ~150-190) to stay connected, but still block on edges.
    expand_white = max(int(white_thresh) - 90, 140)
    expand_grad = float(grad_thresh) * 2.2
    allowed_bg = (gray >= int(expand_white)) & (grad <= float(expand_grad))
    if edge_barrier is not None:
        allowed_bg = allowed_bg & (~edge_barrier)

    allowed_u8 = (allowed_bg.astype(np.uint8) * 255)
    seed_u8 = (seed_bg.astype(np.uint8) * 255)
    allowed_u8 = cv2.morphologyEx(allowed_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    allowed_u8 = cv2.morphologyEx(allowed_u8, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

    num_labels, labels = cv2.connectedComponents((allowed_u8 > 0).astype(np.uint8), connectivity=4)
    if num_labels <= 1:
        bg_mask = seed_bg
    else:
        border_labels = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
        border_labels = border_labels[border_labels != 0]
        seed_labels = np.unique(labels[(seed_u8 > 0)])
        seed_labels = seed_labels[seed_labels != 0]

        good_labels = np.intersect1d(border_labels, seed_labels, assume_unique=False)
        if good_labels.size == 0:
            good_labels = border_labels
        bg_mask = np.isin(labels, good_labels)

    bg_u8 = (bg_mask.astype(np.uint8) * 255)
    bg_u8 = cv2.morphologyEx(bg_u8, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)
    return (bg_u8 > 0)


def _border_connected(mask: np.ndarray) -> np.ndarray:
    """
    Keep only mask pixels connected to the image border (removes "holes" inside documents).
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        return mask.astype(bool)
    u8 = (mask.astype(np.uint8) * 255)
    num_labels, labels = cv2.connectedComponents((u8 > 0).astype(np.uint8), connectivity=4)
    if num_labels <= 1:
        return (u8 > 0)
    border_labels = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return np.zeros((h, w), dtype=bool)
    return np.isin(labels, border_labels)


def _document_mask_estimate(
    gray: np.ndarray,
    *,
    edge_image: Optional[np.ndarray] = None,
    doc_intensity_thresh: int = 235,
    open_ksize: int = 3,
    close_ksize: int = 15,
    close_post_ksize: Optional[int] = None,
    dilate_ksize: int = 9,
    min_area_frac: float = 0.002,
) -> np.ndarray:
    """
    Conservative estimate of where the documents are.
    Any drawing is restricted to the inverse of this mask.
    """
    if gray.ndim != 2:
        raise ValueError("gray must be a single-channel image")

    doc = (gray < int(doc_intensity_thresh))
    if edge_image is not None and edge_image.size:
        doc = doc | (edge_image > 0)

    k_open = int(open_ksize)
    if k_open % 2 == 0:
        k_open += 1
    k_open = max(k_open, 1)

    k_close = int(close_ksize)
    if k_close % 2 == 0:
        k_close += 1
    k_close = max(k_close, 3)

    k_close_post = int(k_close if close_post_ksize is None else close_post_ksize)
    if k_close_post % 2 == 0:
        k_close_post += 1
    k_close_post = max(k_close_post, 3)

    k_dilate = int(dilate_ksize)
    if k_dilate % 2 == 0:
        k_dilate += 1
    k_dilate = max(k_dilate, 0)

    doc_u8 = (doc.astype(np.uint8) * 255)
    if k_open >= 3:
        doc_u8 = cv2.morphologyEx(doc_u8, cv2.MORPH_OPEN, np.ones((k_open, k_open), dtype=np.uint8), iterations=1)
    doc_u8 = cv2.morphologyEx(doc_u8, cv2.MORPH_CLOSE, np.ones((k_close, k_close), dtype=np.uint8), iterations=1)

    # Keep only sufficiently large connected components (drops text/noise in whitespace).
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((doc_u8 > 0).astype(np.uint8), connectivity=8)
    min_area = int(float(min_area_frac) * float(gray.shape[0] * gray.shape[1]))
    min_area = max(min_area, 5000)
    keep = np.zeros(num_labels, dtype=bool)
    for lab in range(1, int(num_labels)):
        if int(stats[lab, cv2.CC_STAT_AREA]) >= min_area:
            keep[lab] = True
    core = np.isin(labels, np.flatnonzero(keep))

    core_u8 = (core.astype(np.uint8) * 255)
    core_u8 = cv2.morphologyEx(
        core_u8, cv2.MORPH_CLOSE, np.ones((k_close_post, k_close_post), dtype=np.uint8), iterations=1
    )
    if k_dilate >= 3:
        core_u8 = cv2.dilate(core_u8, np.ones((k_dilate, k_dilate), dtype=np.uint8), iterations=1)
    return (core_u8 > 0)


def _gap_centers_from_bboxes(
    bboxes: List[Tuple[int, int, int, int]],
    *,
    axis: str,
    limit: int,
    min_gap: int,
) -> List[int]:
    axis_norm = axis.lower().strip()
    if axis_norm not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    if not bboxes:
        return []
    intervals: List[Tuple[int, int]] = []
    for x0, y0, x1, y1 in bboxes:
        if axis_norm == "x":
            a, b = int(x0), int(x1)
        else:
            a, b = int(y0), int(y1)
        if b <= a:
            continue
        intervals.append((a, b))
    if not intervals:
        return []
    intervals.sort()
    merged: List[Tuple[int, int]] = []
    cur_a, cur_b = intervals[0]
    for a, b in intervals[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))

    centers: List[int] = []
    for (a0, b0), (a1, b1) in zip(merged, merged[1:]):
        gap0 = int(b0)
        gap1 = int(a1)
        if gap1 - gap0 < int(min_gap):
            continue
        c = int(round(0.5 * (gap0 + gap1)))
        if 0 < c < int(limit) - 1:
            centers.append(c)
    return centers


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    items = sorted((int(a), int(b)) for a, b in intervals if int(b) > int(a))
    if not items:
        return []
    merged: List[Tuple[int, int]] = []
    cur_a, cur_b = items[0]
    for a, b in items[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))
    return merged


def _gap_centers_from_intervals(merged: List[Tuple[int, int]], *, limit: int, min_gap: int) -> List[int]:
    centers: List[int] = []
    for (a0, b0), (a1, b1) in zip(merged, merged[1:]):
        gap0 = int(b0)
        gap1 = int(a1)
        if gap1 - gap0 < int(min_gap):
            continue
        c = int(round(0.5 * (gap0 + gap1)))
        if 0 < c < int(limit) - 1:
            centers.append(c)
    return centers


def _cluster_bboxes_by_overlap(
    bboxes: List[Tuple[int, int, int, int]],
    *,
    axis: str,
    overlap_frac: float,
) -> List[List[Tuple[int, int, int, int]]]:
    axis_norm = axis.lower().strip()
    if axis_norm not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    if not bboxes:
        return []

    def _interval(bb: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x0, y0, x1, y1 = bb
        return (int(x0), int(x1)) if axis_norm == "x" else (int(y0), int(y1))

    items = sorted(bboxes, key=lambda bb: (_interval(bb)[0], _interval(bb)[1]))
    clusters: List[Dict[str, Any]] = []
    for bb in items:
        a, b = _interval(bb)
        if b <= a:
            continue
        placed = False
        for cl in clusters:
            ca, cb = cl["a"], cl["b"]
            ov = max(0, min(b, cb) - max(a, ca))
            denom = min(b - a, cb - ca)
            if denom <= 0:
                continue
            if float(ov) / float(denom) >= float(overlap_frac):
                cl["bboxes"].append(bb)
                cl["a"] = min(cl["a"], a)
                cl["b"] = max(cl["b"], b)
                placed = True
                break
        if not placed:
            clusters.append({"a": a, "b": b, "bboxes": [bb]})

    return [cl["bboxes"] for cl in clusters]


def _voted_gap_centers_from_bboxes(
    bboxes: List[Tuple[int, int, int, int]],
    *,
    axis: str,
    limit: int,
    min_gap: int,
    band_overlap_frac: float = 0.25,
    bin_width: int = 25,
    min_votes_frac: float = 0.35,
) -> List[int]:
    """
    Compute gap centers per 'band' (rows for x-gaps, cols for y-gaps) and vote them into stable separators.
    """
    axis_norm = axis.lower().strip()
    if axis_norm not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    if not bboxes:
        return []

    cluster_axis = "y" if axis_norm == "x" else "x"
    bands = _cluster_bboxes_by_overlap(bboxes, axis=cluster_axis, overlap_frac=float(band_overlap_frac))
    if not bands:
        return []

    per_band_centers: List[List[int]] = []
    for band in bands:
        intervals: List[Tuple[int, int]] = []
        for x0, y0, x1, y1 in band:
            if axis_norm == "x":
                intervals.append((int(x0), int(x1)))
            else:
                intervals.append((int(y0), int(y1)))
        merged = _merge_intervals(intervals)
        per_band_centers.append(_gap_centers_from_intervals(merged, limit=int(limit), min_gap=int(min_gap)))

    n_bands = int(len(per_band_centers))
    if n_bands <= 1:
        centers = [c for band in per_band_centers for c in band]
        return sorted(set(int(c) for c in centers))

    min_votes = max(2, int(np.ceil(float(n_bands) * float(min_votes_frac))))
    bw = max(int(bin_width), 10)

    bins: Dict[int, Dict[str, Any]] = {}
    for centers in per_band_centers:
        used: set[int] = set()
        for c in centers:
            b = int(round(float(c) / float(bw)))
            if b in used:
                continue
            used.add(b)
            rec = bins.setdefault(b, {"votes": 0, "centers": []})
            rec["votes"] += 1
            rec["centers"].append(int(c))

    out: List[int] = []
    for b, rec in bins.items():
        if int(rec["votes"]) < int(min_votes):
            continue
        out.append(int(round(float(np.median(np.asarray(rec["centers"], dtype=np.float32))))))
    return sorted(set(out))


def _valley_positions_from_edges(
    edge_image: np.ndarray,
    *,
    axis: str,
    smooth_k: int = 81,
    suppress_px: int = 120,
    max_lines: int = 12,
    pct: float = 12.0,
) -> List[int]:
    """
    Pick low-edge-density valleys as candidate separator positions.
    """
    if edge_image.ndim != 2:
        raise ValueError("edge_image must be 2D")
    axis_norm = axis.lower().strip()
    if axis_norm not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")

    edges = (edge_image > 0).astype(np.float32)
    values = edges.sum(axis=0) if axis_norm == "x" else edges.sum(axis=1)
    values = values.astype(np.float32)

    k = int(smooth_k)
    if k % 2 == 0:
        k += 1
    k = max(k, 3)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    smooth = np.convolve(values, kernel, mode="same")

    threshold = float(np.percentile(smooth, float(pct)))
    work = smooth.copy()
    picked: List[int] = []
    sup = max(int(suppress_px), 1)
    for _ in range(int(max_lines)):
        idx = int(np.argmin(work))
        if float(work[idx]) > threshold:
            break
        picked.append(int(idx))
        lo = max(0, idx - sup)
        hi = min(len(work), idx + sup + 1)
        work[lo:hi] = work.max() + 1.0
    picked.sort()
    return picked


def _axis_aligned_border_segments_from_hough(
    fitted_df: pd.DataFrame,
    *,
    original_image_path: str,
    image_shape: Tuple[int, int],
    edge_image: Optional[np.ndarray] = None,
    white_thresh: int = 238,
    grad_thresh: float = 12.0,
    blur_ksize: int = 5,
    strip_half: int = 4,
    bg_ratio_thresh: float = 0.65,
    min_gap_width_px: int = 18,
    coverage_thresh: float = 0.25,
    snap_max_px: int = 60,
    min_run_ratio: float = 0.10,
    min_segment_length_px: int = 12,
    doc_side_window_px: int = 80,
    doc_side_min_frac: float = 0.05,
    doc_side_window_far_px: int = 220,
    doc_side_min_frac_far: float = 0.025,
    doc_other_side_max_frac: float = 0.015,
    endpoint_snap_px: int = 60,
    merge_gap_px: int = 180,
    candidate_bg_ratio_thresh: float = 0.55,
    candidate_score_thresh: float = 0.22,
    candidate_min_run_px: int = 6,
    candidate_max_peaks: int = 10,
    debug: bool = False,
) -> pd.DataFrame:
    height, width = image_shape
    output_columns = ["x1", "y1", "x2", "y2"]

    if fitted_df.empty:
        return pd.DataFrame(columns=output_columns)

    gray = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return pd.DataFrame(columns=output_columns)

    doc_layout = _document_mask_estimate(
        gray,
        edge_image=edge_image,
        doc_intensity_thresh=235,
        open_ksize=3,
        close_ksize=5,
        close_post_ksize=5,
        dilate_ksize=0,
        min_area_frac=0.0015,
    )
    # Slightly expanded doc mask for background corridor checks (keeps separators out of documents).
    doc_for_bg_u8 = (doc_layout.astype(np.uint8) * 255)
    # Smaller dilation preserves narrow gaps between documents (important for thin separators).
    doc_for_bg_u8 = cv2.dilate(doc_for_bg_u8, np.ones((5, 5), dtype=np.uint8), iterations=1)
    doc_for_bg = doc_for_bg_u8 > 0
    bg_mask = _border_connected(~doc_for_bg)

    x_centers = _gap_centers_from_doc_coverage(
        doc_layout, axis=0, coverage_thresh=float(coverage_thresh), min_width=int(min_gap_width_px), exclude_edges=True
    )
    y_centers = _gap_centers_from_doc_coverage(
        doc_layout, axis=1, coverage_thresh=float(coverage_thresh), min_width=int(min_gap_width_px), exclude_edges=True
    )

    # Also derive separator positions directly from the background/doc masks.
    # This is important because Hough can miss some separators (especially vertical ones).
    def _centers_from_sep_score_1d(
        bg_mean: np.ndarray,
        doc_mean: np.ndarray,
        *,
        window: int,
        smooth: int,
        score_thresh: float,
        min_run: int,
        max_peaks: int,
    ) -> List[int]:
        n = int(bg_mean.size)
        if n <= 0:
            return []
        win = max(int(window), 1)
        sm = max(int(smooth), 1)
        if sm % 2 == 0:
            sm += 1
        kernel = np.ones(sm, dtype=np.float32) / float(sm)
        bg_s = np.convolve(bg_mean.astype(np.float32), kernel, mode="same")

        pref = np.concatenate([[0.0], np.cumsum(doc_mean.astype(np.float32))])
        left = np.zeros(n, dtype=np.float32)
        right = np.zeros(n, dtype=np.float32)
        for i in range(win, n - win):
            left[i] = float(pref[i] - pref[i - win]) / float(win)
            right[i] = float(pref[i + win] - pref[i]) / float(win)

        min_lr = np.minimum(left, right)
        score = bg_s * min_lr
        keep = (
            (bg_s >= float(candidate_bg_ratio_thresh))
            & (min_lr >= float(doc_side_min_frac))
            & (score >= float(score_thresh))
        )
        runs = _find_true_runs(keep, min_len=max(int(min_run), 1))
        centers: List[int] = []
        for a, b in runs:
            c = int(round(0.5 * (a + b)))
            if c <= win or c >= n - win - 1:
                continue
            centers.append(c)

        # Also pick strong local maxima (helps when separators are thin/fragmented so runs don't form).
        peak_candidates: List[Tuple[float, int]] = []
        for i in range(1, n - 1):
            if i <= win or i >= n - win - 1:
                continue
            if score[i] < float(score_thresh):
                continue
            if score[i] >= score[i - 1] and score[i] >= score[i + 1]:
                peak_candidates.append((float(score[i]), int(i)))
        peak_candidates.sort(reverse=True)
        spacing = max(int(win // 2), 40)
        picked: List[int] = []
        for _, idx in peak_candidates:
            if any(abs(idx - p) <= spacing for p in picked):
                continue
            picked.append(int(idx))
            if len(picked) >= int(max_peaks):
                break

        centers.extend(picked)
        return centers

    w_side = max(int(doc_side_window_px), 10)
    smooth_w = max(2 * int(strip_half) + 1, 3)
    x_from_mask = _centers_from_sep_score_1d(
        bg_mask.mean(axis=0),
        doc_layout.mean(axis=0),
        window=w_side,
        smooth=smooth_w,
        score_thresh=float(candidate_score_thresh),
        min_run=int(candidate_min_run_px),
        max_peaks=int(candidate_max_peaks),
    )
    y_from_mask = _centers_from_sep_score_1d(
        bg_mask.mean(axis=1),
        doc_layout.mean(axis=1),
        window=w_side,
        smooth=smooth_w,
        score_thresh=float(candidate_score_thresh),
        min_run=int(candidate_min_run_px),
        max_peaks=int(candidate_max_peaks),
    )

    # Derive separator positions from document bounding boxes (robust when projections are noisy).
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((doc_layout > 0).astype(np.uint8), connectivity=8)
    bboxes: List[Tuple[int, int, int, int]] = []
    for lab in range(1, int(num_labels)):
        x0 = int(stats[lab, cv2.CC_STAT_LEFT])
        y0 = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        x1 = x0 + w
        y1 = y0 + h
        if w <= 0 or h <= 0:
            continue
        bboxes.append((x0, y0, x1, y1))

    x_from_boxes = _gap_centers_from_bboxes(bboxes, axis="x", limit=width, min_gap=int(min_gap_width_px))
    y_from_boxes = _gap_centers_from_bboxes(bboxes, axis="y", limit=height, min_gap=int(min_gap_width_px))
    x_from_boxes_voted = _voted_gap_centers_from_bboxes(
        bboxes, axis="x", limit=width, min_gap=int(min_gap_width_px), band_overlap_frac=0.25, bin_width=25, min_votes_frac=0.30
    )
    y_from_boxes_voted = _voted_gap_centers_from_bboxes(
        bboxes, axis="y", limit=height, min_gap=int(min_gap_width_px), band_overlap_frac=0.25, bin_width=25, min_votes_frac=0.30
    )

    def _band_local_gap_centers_from_bboxes(
        bboxes_in: List[Tuple[int, int, int, int]],
        *,
        axis: str,
        limit: int,
        min_gap: int,
        band_overlap_frac: float = 0.25,
    ) -> List[int]:
        """
        Like voting, but returns the UNION of gap centers per band.
        This recovers separators that exist only in some rows/cols (grid-like layouts).
        """
        axis_norm = axis.lower().strip()
        if axis_norm not in {"x", "y"}:
            raise ValueError("axis must be 'x' or 'y'")
        if not bboxes_in:
            return []
        cluster_axis = "y" if axis_norm == "x" else "x"
        bands = _cluster_bboxes_by_overlap(bboxes_in, axis=cluster_axis, overlap_frac=float(band_overlap_frac))
        out: List[int] = []
        for band in bands:
            intervals: List[Tuple[int, int]] = []
            for x0, y0, x1, y1 in band:
                intervals.append((int(x0), int(x1)) if axis_norm == "x" else (int(y0), int(y1)))
            merged = _merge_intervals(intervals)
            out.extend(_gap_centers_from_intervals(merged, limit=int(limit), min_gap=int(min_gap)))
        return sorted(set(int(c) for c in out))

    x_from_boxes_band = _band_local_gap_centers_from_bboxes(
        bboxes, axis="x", limit=width, min_gap=int(min_gap_width_px), band_overlap_frac=0.25
    )
    y_from_boxes_band = _band_local_gap_centers_from_bboxes(
        bboxes, axis="y", limit=height, min_gap=int(min_gap_width_px), band_overlap_frac=0.25
    )

    x_candidates: List[int] = []
    y_candidates: List[int] = []
    for row in fitted_df.itertuples(index=False):
        orientation = getattr(row, "orientation", "")
        if orientation == "vertical":
            x_candidates.append(int(round(0.5 * (float(row.x1) + float(row.x2)))))
        elif orientation == "horizontal":
            y_candidates.append(int(round(0.5 * (float(row.y1) + float(row.y2)))))

    margin = int(strip_half)
    # Allow band-local separators (not full-image spanning) by lowering required run length.
    min_run_v = max(int(height * float(min_run_ratio)), 60)
    min_run_h = max(int(width * float(min_run_ratio)), 80)

    x_refined = _refine_positions_to_background_runs(
        x_candidates,
        bg_mask=bg_mask,
        axis="x",
        search_px=max(int(snap_max_px), 10),
        strip_half=margin,
        ratio_thresh=float(bg_ratio_thresh),
        min_keep_run=max(int(min_run_v * 0.6), 40),
    )
    y_refined = _refine_positions_to_background_runs(
        y_candidates,
        bg_mask=bg_mask,
        axis="y",
        search_px=max(int(snap_max_px), 10),
        strip_half=margin,
        ratio_thresh=float(bg_ratio_thresh),
        min_keep_run=max(int(min_run_h * 0.6), 60),
    )

    x_positions = _snap_positions_to_centers(x_refined, x_centers, max_dist=int(snap_max_px)) or _dedupe_positions(
        x_refined, tol_px=max(int(margin) * 2, 8)
    )
    y_positions = _snap_positions_to_centers(y_refined, y_centers, max_dist=int(snap_max_px)) or _dedupe_positions(
        y_refined, tol_px=max(int(margin) * 2, 10)
    )

    # If Hough misses some separators, add mask-derived candidates.
    x_positions = sorted(
        set(
            int(x)
            for x in (
                x_positions + x_centers + x_from_mask + x_from_boxes + x_from_boxes_voted + x_from_boxes_band
            )
        )
    )
    y_positions = sorted(
        set(
            int(y)
            for y in (
                y_positions + y_centers + y_from_mask + y_from_boxes + y_from_boxes_voted + y_from_boxes_band
            )
        )
    )

    # Edge-energy valleys catch separators that are hard to segment as "documents" (very bright pages).
    if edge_image is not None and edge_image.size:
        x_valleys = _valley_positions_from_edges(edge_image, axis="x", smooth_k=81, suppress_px=140, max_lines=14, pct=12.0)
        y_valleys = _valley_positions_from_edges(edge_image, axis="y", smooth_k=81, suppress_px=140, max_lines=14, pct=12.0)
        x_positions = sorted(set(x_positions + x_valleys))
        y_positions = sorted(set(y_positions + y_valleys))

    # Final refinement: snap all candidate positions (including mask-derived) onto the best local
    # background corridor so long separators don't break due to slight misalignment.
    x_positions = _refine_positions_to_background_runs(
        x_positions,
        bg_mask=bg_mask,
        axis="x",
        search_px=max(int(snap_max_px), 25),
        strip_half=margin,
        ratio_thresh=float(bg_ratio_thresh),
        min_keep_run=max(int(min_run_v * 0.3), 40),
    )
    y_positions = _refine_positions_to_background_runs(
        y_positions,
        bg_mask=bg_mask,
        axis="y",
        search_px=max(int(snap_max_px), 25),
        strip_half=margin,
        ratio_thresh=float(bg_ratio_thresh),
        min_keep_run=max(int(min_run_h * 0.3), 60),
    )

    def _max_bg_run_len_at(pos: int, *, axis: str) -> int:
        pos_i = int(pos)
        axis_norm = axis.lower().strip()
        if axis_norm == "x":
            x0 = max(0, pos_i - margin)
            x1 = min(width, pos_i + margin + 1)
            slice_bg = bg_mask[:, x0:x1].astype(np.float32)
            row_bg = slice_bg.mean(axis=1) >= float(bg_ratio_thresh)
            runs = _find_true_runs(row_bg, min_len=1)
            return int(max((b - a + 1) for a, b in runs) if runs else 0)
        if axis_norm == "y":
            y0 = max(0, pos_i - margin)
            y1 = min(height, pos_i + margin + 1)
            slice_bg = bg_mask[y0:y1, :].astype(np.float32)
            col_bg = slice_bg.mean(axis=0) >= float(bg_ratio_thresh)
            runs = _find_true_runs(col_bg, min_len=1)
            return int(max((b - a + 1) for a, b in runs) if runs else 0)
        raise ValueError("axis must be 'x' or 'y'")

    def _dedupe_positions_best(positions: List[int], *, tol_px: int, axis: str) -> List[int]:
        if not positions:
            return []
        tol = max(int(tol_px), 1)
        items = sorted(set(int(p) for p in positions))
        groups: List[List[int]] = []
        cur: List[int] = []
        for p in items:
            if not cur or abs(p - cur[-1]) <= tol:
                cur.append(p)
            else:
                groups.append(cur)
                cur = [p]
        if cur:
            groups.append(cur)

        picked: List[int] = []
        for g in groups:
            scored = [(int(_max_bg_run_len_at(p, axis=axis)), int(p)) for p in g]
            scored.sort(reverse=True)  # longest run first
            picked.append(int(scored[0][1]))
        return sorted(set(picked))

    x_positions = _dedupe_positions_best(x_positions, tol_px=max(int(margin) * 3, 14), axis="x")
    # Collapse clusters of nearby row separators into a single grid line (prevents double-lines in the same gap).
    y_positions = _dedupe_positions_best(y_positions, tol_px=max(int(margin) * 4, 40), axis="y")

    # Do not globally require docs on both sides; enforce this per-segment later.

    if debug:
        print(
            f"Border-line snapping: {len(x_candidates)} vertical candidates -> {len(x_positions)} centers; "
            f"{len(y_candidates)} horizontal candidates -> {len(y_positions)} centers"
        )

    vertical_segments: List[Dict[str, int]] = []
    horizontal_segments: List[Dict[str, int]] = []

    def _col_runs(x_pos: int) -> List[Tuple[int, int]]:
        x_pos = int(x_pos)
        if x_pos <= int(doc_side_window_px) or x_pos >= width - int(doc_side_window_px) - 1:
            return []
        x0 = max(0, x_pos - margin)
        x1 = min(width, x_pos + margin + 1)
        slice_bg = bg_mask[:, x0:x1].astype(np.float32)
        row_bg = slice_bg.mean(axis=1) >= float(bg_ratio_thresh)
        side = max(int(doc_side_window_px), 10)
        side_row_frac = max(float(doc_side_min_frac) * 0.8, 0.03)
        left_doc = doc_layout[:, x_pos - side : x_pos].astype(np.float32).mean(axis=1) >= float(side_row_frac)
        right_doc = doc_layout[:, x_pos : x_pos + side].astype(np.float32).mean(axis=1) >= float(side_row_frac)
        # Allow separators OR outer borders (doc on at least one side).
        good = row_bg & (left_doc | right_doc)
        return _find_true_runs(good, min_len=min_run_v)

    def _row_runs(y_pos: int) -> List[Tuple[int, int]]:
        y_pos = int(y_pos)
        if y_pos <= int(doc_side_window_px) or y_pos >= height - int(doc_side_window_px) - 1:
            return []
        y0 = max(0, y_pos - margin)
        y1 = min(height, y_pos + margin + 1)
        slice_bg = bg_mask[y0:y1, :].astype(np.float32)
        col_bg = slice_bg.mean(axis=0) >= float(bg_ratio_thresh)
        side = max(int(doc_side_window_px), 10)
        side_col_frac = max(float(doc_side_min_frac) * 0.8, 0.03)
        above_doc = doc_layout[y_pos - side : y_pos, :].astype(np.float32).mean(axis=0) >= float(side_col_frac)
        below_doc = doc_layout[y_pos : y_pos + side, :].astype(np.float32).mean(axis=0) >= float(side_col_frac)
        # Allow separators OR outer borders (doc on at least one side).
        good = col_bg & (above_doc | below_doc)
        return _find_true_runs(good, min_len=min_run_h)

    for x in x_positions:
        for y0, y1 in _col_runs(int(x)):
            vertical_segments.append({"x": int(x), "y0": int(y0), "y1": int(y1)})

    for y in y_positions:
        for x0, x1 in _row_runs(int(y)):
            horizontal_segments.append({"y": int(y), "x0": int(x0), "x1": int(x1)})

    if not vertical_segments and not horizontal_segments:
        return pd.DataFrame(columns=output_columns)

    # Snap segment endpoints so grid lines meet at intersections (fills small gaps without drawing on documents).
    snap_px = max(int(endpoint_snap_px), 0)
    if snap_px > 0 and (x_positions or y_positions):
        def _bg_ok_vertical_bridge(x: int, y0: int, y1: int) -> bool:
            if y1 < y0:
                y0, y1 = y1, y0
            if y1 - y0 + 1 <= 0:
                return False
            x0 = max(0, int(x) - margin)
            x1 = min(width, int(x) + margin + 1)
            band = bg_mask[y0 : y1 + 1, x0:x1].astype(np.float32)
            return float(band.mean()) >= float(bg_ratio_thresh)

        def _bg_ok_horizontal_bridge(y: int, x0: int, x1: int) -> bool:
            if x1 < x0:
                x0, x1 = x1, x0
            if x1 - x0 + 1 <= 0:
                return False
            y0 = max(0, int(y) - margin)
            y1 = min(height, int(y) + margin + 1)
            band = bg_mask[y0:y1, x0 : x1 + 1].astype(np.float32)
            return float(band.mean()) >= float(bg_ratio_thresh)

        x_sorted = sorted(set(int(x) for x in x_positions))
        y_sorted = sorted(set(int(y) for y in y_positions))

        for seg in horizontal_segments:
            y = int(seg["y"])
            x0, x1 = int(seg["x0"]), int(seg["x1"])
            if x1 < x0:
                x0, x1 = x1, x0
            left_targets = [x for x in x_sorted if x0 - x <= snap_px and x <= x0]
            if left_targets:
                tx = max(left_targets)
                if tx < x0 and _bg_ok_horizontal_bridge(y, tx, x0):
                    x0 = int(tx)
            right_targets = [x for x in x_sorted if x - x1 <= snap_px and x >= x1]
            if right_targets:
                tx = min(right_targets)
                if tx > x1 and _bg_ok_horizontal_bridge(y, x1, tx):
                    x1 = int(tx)
            seg["x0"], seg["x1"] = int(x0), int(x1)

        for seg in vertical_segments:
            x = int(seg["x"])
            y0, y1 = int(seg["y0"]), int(seg["y1"])
            if y1 < y0:
                y0, y1 = y1, y0
            up_targets = [y for y in y_sorted if y0 - y <= snap_px and y <= y0]
            if up_targets:
                ty = max(up_targets)
                if ty < y0 and _bg_ok_vertical_bridge(x, ty, y0):
                    y0 = int(ty)
            down_targets = [y for y in y_sorted if y - y1 <= snap_px and y >= y1]
            if down_targets:
                ty = min(down_targets)
                if ty > y1 and _bg_ok_vertical_bridge(x, y1, ty):
                    y1 = int(ty)
            seg["y0"], seg["y1"] = int(y0), int(y1)

    # Compute intersections between axis-aligned segments and split so endpoints land on intersections.
    v_breaks: List[List[int]] = [[] for _ in range(len(vertical_segments))]
    h_breaks: List[List[int]] = [[] for _ in range(len(horizontal_segments))]

    for vi, v in enumerate(vertical_segments):
        x = v["x"]
        y0, y1 = v["y0"], v["y1"]
        if y1 < y0:
            y0, y1 = y1, y0
        for hi, h in enumerate(horizontal_segments):
            y = h["y"]
            x0, x1 = h["x0"], h["x1"]
            if x1 < x0:
                x0, x1 = x1, x0
            if x0 <= x <= x1 and y0 <= y <= y1:
                v_breaks[vi].append(int(y))
                h_breaks[hi].append(int(x))

    def _segment_bg_ok_vertical(x: int, y0: int, y1: int) -> bool:
        if y1 < y0:
            y0, y1 = y1, y0
        if y1 - y0 + 1 <= 0:
            return False
        x0 = max(0, int(x) - margin)
        x1 = min(width, int(x) + margin + 1)
        band = bg_mask[y0 : y1 + 1, x0:x1].astype(np.float32)
        return float(band.mean()) >= float(bg_ratio_thresh)

    def _segment_bg_ok_horizontal(y: int, x0: int, x1: int) -> bool:
        if x1 < x0:
            x0, x1 = x1, x0
        if x1 - x0 + 1 <= 0:
            return False
        y0 = max(0, int(y) - margin)
        y1 = min(height, int(y) + margin + 1)
        band = bg_mask[y0:y1, x0 : x1 + 1].astype(np.float32)
        return float(band.mean()) >= float(bg_ratio_thresh)

    side_px = max(int(doc_side_window_px), 10)
    side_frac = float(doc_side_min_frac)
    far_px = max(int(doc_side_window_far_px), side_px)
    far_frac = float(doc_side_min_frac_far)
    other_max = float(doc_other_side_max_frac)

    def _segment_separates_docs_vertical(x: int, y0: int, y1: int) -> bool:
        if y1 < y0:
            y0, y1 = y1, y0
        if x <= side_px or x >= width - side_px - 1:
            return False
        left = doc_layout[y0 : y1 + 1, x - side_px : x].astype(np.float32).mean()
        right = doc_layout[y0 : y1 + 1, x : x + side_px].astype(np.float32).mean()
        if float(left) >= side_frac and float(right) >= side_frac:
            return True
        # Outer border: document on one side only.
        if (float(left) >= side_frac and float(right) <= other_max) or (float(right) >= side_frac and float(left) <= other_max):
            return True
        if x <= far_px or x >= width - far_px - 1:
            return False
        left_far = doc_layout[y0 : y1 + 1, x - far_px : x].astype(np.float32).mean()
        right_far = doc_layout[y0 : y1 + 1, x : x + far_px].astype(np.float32).mean()
        return (float(left) >= side_frac and float(right_far) >= far_frac) or (
            float(left_far) >= far_frac and float(right) >= side_frac
        )

    def _segment_separates_docs_horizontal(y: int, x0: int, x1: int) -> bool:
        if x1 < x0:
            x0, x1 = x1, x0
        if y <= side_px or y >= height - side_px - 1:
            return False
        above = doc_layout[y - side_px : y, x0 : x1 + 1].astype(np.float32).mean()
        below = doc_layout[y : y + side_px, x0 : x1 + 1].astype(np.float32).mean()
        if float(above) >= side_frac and float(below) >= side_frac:
            return True
        # Outer border: document on one side only.
        if (float(above) >= side_frac and float(below) <= other_max) or (float(below) >= side_frac and float(above) <= other_max):
            return True
        if y <= far_px or y >= height - far_px - 1:
            return False
        above_far = doc_layout[y - far_px : y, x0 : x1 + 1].astype(np.float32).mean()
        below_far = doc_layout[y : y + far_px, x0 : x1 + 1].astype(np.float32).mean()
        return (float(above) >= side_frac and float(below_far) >= far_frac) or (
            float(above_far) >= far_frac and float(below) >= side_frac
        )

    out_segments: List[Dict[str, int]] = []

    for vi, v in enumerate(vertical_segments):
        x = int(v["x"])
        y0, y1 = int(v["y0"]), int(v["y1"])
        if y1 < y0:
            y0, y1 = y1, y0
        stops = sorted(set([y0, y1] + v_breaks[vi]))
        for a, b in zip(stops, stops[1:]):
            if b - a < int(min_segment_length_px):
                continue
            if not _segment_bg_ok_vertical(x, a, b):
                continue
            if not _segment_separates_docs_vertical(x, a, b):
                continue
            out_segments.append({"x1": x, "y1": int(a), "x2": x, "y2": int(b)})

    for hi, h in enumerate(horizontal_segments):
        y = int(h["y"])
        x0, x1 = int(h["x0"]), int(h["x1"])
        if x1 < x0:
            x0, x1 = x1, x0
        stops = sorted(set([x0, x1] + h_breaks[hi]))
        for a, b in zip(stops, stops[1:]):
            if b - a < int(min_segment_length_px):
                continue
            if not _segment_bg_ok_horizontal(y, a, b):
                continue
            if not _segment_separates_docs_horizontal(y, a, b):
                continue
            out_segments.append({"x1": int(a), "y1": y, "x2": int(b), "y2": y})

    if not out_segments:
        return pd.DataFrame(columns=output_columns)

    seg_df = pd.DataFrame(out_segments, columns=output_columns)

    # Merge small gaps so separators look continuous across parallel documents.
    gap = max(int(merge_gap_px), 0)
    if gap > 0 and not seg_df.empty:
        merged: List[Dict[str, int]] = []
        seg_df = seg_df.copy()
        seg_df["_orient"] = np.where(seg_df["x1"] == seg_df["x2"], "v", "h")

        for orient, group in seg_df.groupby("_orient"):
            if orient == "h":
                for y, rows in group.groupby("y1"):
                    rows = rows.sort_values("x1")
                    cur: Optional[Dict[str, int]] = None
                    for row in rows.itertuples(index=False):
                        x1, x2 = int(row.x1), int(row.x2)
                        if x2 < x1:
                            x1, x2 = x2, x1
                        if cur is None:
                            cur = {"x1": x1, "y1": int(y), "x2": x2, "y2": int(y)}
                            continue
                        if x1 <= cur["x2"] + gap:
                            bridge_ok = _segment_bg_ok_horizontal(int(y), cur["x2"], x1) if x1 > cur["x2"] else True
                            if bridge_ok:
                                cur["x2"] = max(cur["x2"], x2)
                                continue
                        merged.append(cur)
                        cur = {"x1": x1, "y1": int(y), "x2": x2, "y2": int(y)}
                    if cur is not None:
                        merged.append(cur)
            else:
                for x, rows in group.groupby("x1"):
                    rows = rows.sort_values("y1")
                    cur = None
                    for row in rows.itertuples(index=False):
                        y1, y2 = int(row.y1), int(row.y2)
                        if y2 < y1:
                            y1, y2 = y2, y1
                        if cur is None:
                            cur = {"x1": int(x), "y1": y1, "x2": int(x), "y2": y2}
                            continue
                        if y1 <= cur["y2"] + gap:
                            bridge_ok = _segment_bg_ok_vertical(int(x), cur["y2"], y1) if y1 > cur["y2"] else True
                            if bridge_ok:
                                cur["y2"] = max(cur["y2"], y2)
                                continue
                        merged.append(cur)
                        cur = {"x1": int(x), "y1": y1, "x2": int(x), "y2": y2}
                    if cur is not None:
                        merged.append(cur)

        seg_df = pd.DataFrame(merged, columns=output_columns)

    return seg_df.reset_index(drop=True)


def _intersection_clipped_segments(
    df: pd.DataFrame, image_shape: Tuple[int, int], *, min_segment_length_px: float = 10.0
) -> pd.DataFrame:
    height, width = image_shape
    if df.empty:
        return pd.DataFrame(columns=["x1", "y1", "x2", "y2"])

    line_infos: List[Dict[str, Any]] = []
    for row in df.itertuples(index=False):
        p1 = (float(row.x1), float(row.y1))
        p2 = (float(row.x2), float(row.y2))
        a, b, c = _line_abc_from_points(p1, p2)
        dx, dy = _normalize_vector(p2[0] - p1[0], p2[1] - p1[1])
        if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
            continue
        line_infos.append({"a": a, "b": b, "c": c, "dx": dx, "dy": dy, "orientation": getattr(row, "orientation", "")})

    segments: List[Tuple[int, int, int, int]] = []
    x_min, x_max = 0.0, float(width - 1)
    y_min, y_max = 0.0, float(height - 1)

    for i, li in enumerate(line_infos):
        boundary_points = _line_rect_intersections_abc(li["a"], li["b"], li["c"], width, height)
        if len(boundary_points) < 2:
            continue

        points = list(boundary_points)
        for j, lj in enumerate(line_infos):
            if i == j:
                continue
            inter = _line_intersection_abc(li["a"], li["b"], li["c"], lj["a"], lj["b"], lj["c"])
            if inter is None:
                continue
            x, y = inter
            if not (x_min - 0.5 <= x <= x_max + 0.5 and y_min - 0.5 <= y <= y_max + 0.5):
                continue
            points.append((x, y))

        # De-dupe nearby points (pixel-space)
        unique_points: List[Tuple[float, float]] = []
        for x, y in points:
            if any(abs(x - ux) <= 1.5 and abs(y - uy) <= 1.5 for ux, uy in unique_points):
                continue
            unique_points.append((x, y))

        dx, dy = li["dx"], li["dy"]
        unique_points.sort(key=lambda p: p[0] * dx + p[1] * dy)

        for (x1, y1), (x2, y2) in zip(unique_points, unique_points[1:]):
            if float(np.hypot(x2 - x1, y2 - y1)) < min_segment_length_px:
                continue

            x1i = int(np.clip(int(round(x1)), 0, width - 1))
            y1i = int(np.clip(int(round(y1)), 0, height - 1))
            x2i = int(np.clip(int(round(x2)), 0, width - 1))
            y2i = int(np.clip(int(round(y2)), 0, height - 1))

            if x1i == x2i and y1i == y2i:
                continue
            segments.append((x1i, y1i, x2i, y2i))

    if not segments:
        return pd.DataFrame(columns=["x1", "y1", "x2", "y2"])

    # De-dupe identical segments (order-insensitive)
    unique_segments: Dict[Tuple[int, int, int, int], None] = {}
    for x1, y1, x2, y2 in segments:
        if (x2, y2, x1, y1) in unique_segments:
            continue
        unique_segments[(x1, y1, x2, y2)] = None

    seg_df = pd.DataFrame(list(unique_segments.keys()), columns=["x1", "y1", "x2", "y2"])
    return seg_df


def _intersection_clipped_spans(
    df: pd.DataFrame, image_shape: Tuple[int, int], *, min_segment_length_px: float = 10.0
) -> pd.DataFrame:
    """
    For each fitted line (x1,y1)-(x2,y2), extend it across the image, but clip
    the extension so it stops at the *nearest* intersection on each side of the
    matched-edge segment (or at the image boundary if no intersection exists).

    This uses the fitted endpoints to define the line direction, so the
    "matched edges" determine which side is "before/after" when searching for
    the closest intersections.
    """
    height, width = image_shape
    if df.empty:
        return pd.DataFrame(columns=["x1", "y1", "x2", "y2"])

    x_min, x_max = 0.0, float(width - 1)
    y_min, y_max = 0.0, float(height - 1)

    line_infos: List[Dict[str, Any]] = []
    for row in df.itertuples(index=False):
        p1 = (float(row.x1), float(row.y1))
        p2 = (float(row.x2), float(row.y2))
        a, b, c = _line_abc_from_points(p1, p2)
        dx, dy = _normalize_vector(p2[0] - p1[0], p2[1] - p1[1])
        if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
            continue
        t1 = p1[0] * dx + p1[1] * dy
        t2 = p2[0] * dx + p2[1] * dy
        t_min = min(t1, t2)
        t_max = max(t1, t2)
        line_infos.append(
            {
                "a": a,
                "b": b,
                "c": c,
                "dx": dx,
                "dy": dy,
                "t_min": t_min,
                "t_max": t_max,
            }
        )

    segments: List[Tuple[int, int, int, int]] = []

    for i, li in enumerate(line_infos):
        stop_points: List[Tuple[float, float]] = []

        boundary_points = _line_rect_intersections_abc(li["a"], li["b"], li["c"], width, height)
        stop_points.extend(boundary_points)

        for j, lj in enumerate(line_infos):
            if i == j:
                continue
            inter = _line_intersection_abc(li["a"], li["b"], li["c"], lj["a"], lj["b"], lj["c"])
            if inter is None:
                continue
            x, y = inter
            if not (x_min - 0.5 <= x <= x_max + 0.5 and y_min - 0.5 <= y <= y_max + 0.5):
                continue
            stop_points.append((x, y))

        unique_stops: List[Tuple[float, float]] = []
        for x, y in stop_points:
            if any(abs(x - ux) <= 1.5 and abs(y - uy) <= 1.5 for ux, uy in unique_stops):
                continue
            unique_stops.append((x, y))

        if len(unique_stops) < 2:
            continue

        dx, dy = li["dx"], li["dy"]
        stops_with_t = [(p[0] * dx + p[1] * dy, p) for p in unique_stops]
        stops_with_t.sort(key=lambda item: item[0])

        t_min = li["t_min"]
        t_max = li["t_max"]
        eps_t = 1e-6

        left_candidates = [item for item in stops_with_t if item[0] <= t_min - eps_t]
        right_candidates = [item for item in stops_with_t if item[0] >= t_max + eps_t]

        if left_candidates:
            left_t, left_p = max(left_candidates, key=lambda item: item[0])
        else:
            left_t, left_p = min(stops_with_t, key=lambda item: item[0])

        if right_candidates:
            right_t, right_p = min(right_candidates, key=lambda item: item[0])
        else:
            right_t, right_p = max(stops_with_t, key=lambda item: item[0])

        if right_t < left_t + eps_t:
            continue

        if float(np.hypot(right_p[0] - left_p[0], right_p[1] - left_p[1])) < min_segment_length_px:
            continue

        x1i = int(np.clip(int(round(left_p[0])), 0, width - 1))
        y1i = int(np.clip(int(round(left_p[1])), 0, height - 1))
        x2i = int(np.clip(int(round(right_p[0])), 0, width - 1))
        y2i = int(np.clip(int(round(right_p[1])), 0, height - 1))

        if x1i == x2i and y1i == y2i:
            continue

        segments.append((x1i, y1i, x2i, y2i))

    if not segments:
        return pd.DataFrame(columns=["x1", "y1", "x2", "y2"])

    unique_segments: Dict[Tuple[int, int, int, int], None] = {}
    for x1, y1, x2, y2 in segments:
        if (x2, y2, x1, y1) in unique_segments:
            continue
        unique_segments[(x1, y1, x2, y2)] = None

    seg_df = pd.DataFrame(list(unique_segments.keys()), columns=["x1", "y1", "x2", "y2"])
    return seg_df

def load_image(image_path):
    """Load an image from the specified file path."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image


def calculate_gaussian_kernel(size, sigma=1):
    """calculate a Gaussian kernel.
    Args:
        size (int): Size of the kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian distribution.

        Low Sigma (e.g., 0.5): The center pixel gets almost all the weight. The blur is barely noticeable.
        High Sigma (e.g., 2.0): The neighbor pixels get more weight. The blur becomes stronger.

        A good rule of thumb is size = 6 * sigma + 1.
    """
    # 1. Ensure the size is odd
    if size % 2 == 0:
        raise ValueError("Size of the kernel must be odd.")
    
    # 2. Prepare the kernel
    kernel = np.zeros((size, size))

    # 3. Calculate the center index
    center = size // 2

    # 4. Calculate the Gaussian values
    for i in range(size):
        for j in range(size):
            # Calculate the distance from the center. eg: 
            x = i - center
            y = j - center
            # 5. Compute the Gaussian function
            # Formula: G(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))
            # We dont need the normalization factor because it will cancel out when normalizing the kernel later
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return kernel / np.sum(kernel)

def convolve2d(image, kernel):
    """Perform a 2D convolution between an image and a kernel.
    Args:
        image (np.ndarray): Input grayscale image.
        kernel (np.ndarray): Convolution kernel.
    """
    # Get dimensions
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    # Calculate padding size (for 3x3, pad is 1. For 5x5, pad is 2)
    pad = kernel_h // 2
    
    # Create an empty output image
    output = np.zeros_like(image, dtype=np.float64)
    
    # Add padding to the original image (using Zero Padding here)
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    
    # Iterate over the image
    for i in range(image_h):
        for j in range(image_w):
            # Extract the region of interest (ROI)
            # This matches the size of the kernel
            region = padded_image[i : i + kernel_h, j : j + kernel_w]
            
            # THE MATH: Element-wise multiplication followed by sum
            new_value = np.sum(region * kernel)
            
            # Store result
            output[i, j] = new_value
            
    return output
            
def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """Perform Canny edge detection on the input image.
    Args:
        image (np.ndarray): Input grayscale image.
        low_threshold (int): Low threshold for hysteresis.
        high_threshold (int): High threshold for hysteresis.
    """
    # Step 1: Noise Reduction using Gaussian Filter
    gaussian_filter = calculate_gaussian_kernel(size=15, sigma=2.5)

    smoothed_image = convolve2d(image, gaussian_filter)

    # Step 2: Gradient Calculation
    G, D = gradient_intensity_and_direction(smoothed_image)

    # Step 3: Non-Maximum Suppression
    non_max_img = non_max_suppression(np.copy(G), D)

    # Step 4: Double Thresholding
    thresh_img, weak = threshold(np.copy(non_max_img), low_threshold, high_threshold)

    # Step 5: Edge Tracking by Hysteresis
    final_image = tracking(thresh_img, weak, strong=255)

    return final_image.astype(np.uint8), G

def gradient_intensity_and_direction(image):
    """Compute gradient intensity and direction using Sobel operators.
    Args:
        image (np.ndarray): Input grayscale image.
    """

    # Sobel Kernels
    Kx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]], 
                   np.int32)
    Ky = np.array([[1, 2, 1], 
                   [0, 0, 0], 
                   [-1, -2, -1]],
                    np.int32)

    # Convolve with Sobel kernels
    Ix = convolve2d(image, Kx)
    Iy = convolve2d(image, Ky)

    # Calculate gradient magnitude and direction
    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix)
    return (G, D)

def non_max_suppression(G, D):
    """
    Perform non-maximum suppression on the gradient magnitude image.
    Args:
        G (np.ndarray): Gradient magnitude image.
        D (np.ndarray): Gradient direction image.
    Returns:
        np.ndarray: Image after non-maximum suppression.
    """
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32)
    for i in range(1, M-1):
        for j in range(1, N-1):
            angle = np.rad2deg(D[i,j])
            angle = angle % 180

            # Determine the two neighboring pixels to compare

            # angle 0
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = G[i, j+1]
                r = G[i, j-1]

            # angle 45
            elif (22.5 <= angle < 67.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]

            # angle 90
            elif (67.5 <= angle < 112.5):
                q = G[i+1, j]
                r = G[i-1, j]

            # angle 135
            elif (112.5 <= angle < 157.5):
                q = G[i-1, j-1]
                r = G[i+1, j+1]

            # Suppress non-maximum pixels
            if (G[i,j] >= q) and (G[i,j] >= r):
                Z[i,j] = G[i,j]
            else:
                Z[i,j] = 0

    return Z

def threshold(img, lowThreshold, highThreshold):
    """
    Apply double thresholding to the image.
    Args:
        img (np.ndarray): Input image after non-maximum suppression.
        lowThreshold (int): Low threshold value.
        highThreshold (int): High threshold value.
    Returns:
        np.ndarray: Image after applying double thresholding.
        int: Value representing weak edges.
    """
    weak = np.int32(50)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    img[strong_i, strong_j] = strong
    img[weak_i, weak_j] = weak
    img[zeros_i, zeros_j] = np.int32(0)

    return (img, weak)

def tracking(img, weak, strong=255):
    """
    Perform edge tracking by hysteresis.
    
    Args:
        img (np.ndarray): Input image after double thresholding.
        weak (int): Value representing weak edges.
        strong (int): Value representing strong edges.
    Returns:
        np.ndarray: Final edge-detected image.
    """

    # Collect all strong edge pixel indices
    M, N = img.shape
    queue = deque()
    for i in range(M):
        for j in range(N):
            if img[i,j] == strong:
                queue.append((i,j))
    
    # Process queue
    while queue:
        i, j = queue.popleft()
        # Check all 8 neighbors
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < M and 0 <= nj < N and img[ni,nj] == weak:
                    img[ni,nj] = strong
                    queue.append((ni,nj))

    # Set remaining weak edges to zero
    img[img == weak] = 0

    return img

def create_hough_accumulator(image_shape, 
                              num_thetas = 180):
    """
    Create and initialize the Hough accumulator array for line detection.
    
    The accumulator is a 2D voting array where:
    - Rows correspond to different rho values
    - Columns correspond to different theta values
    
    Args:
        image_shape: (height, width) of the input image
        num_thetas: Number of theta bins (default 180 for 1-degree resolution)
    
    Returns:
        Tuple of (accumulator, rhos, thetas) where:
        - accumulator: 2D array of zeros with shape (num_rhos, num_thetas)
        - rhos: 1D array of rho values (from -diagonal to +diagonal)
        - thetas: 1D array of theta values in radians
    
    """
    height, width = image_shape
    
    # Calculate the diagonal length (maximum possible rho)
    diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))
    
    # Create array of theta values (in radians)
    thetas = np.deg2rad(np.arange(-90, 89 + 1))
    
    # -------------------------------------------------------------------------
    # The rhos array should contain integer values from -diagonal to +diagonal
    # The accumulator should be a 2D array of zeros with shape (num_rhos, num_thetas)

    # 
    # STEP 1: Create rhos array using np.arange() from -diagonal to diagonal (inclusive)
    rhos = np.arange(-diagonal, diagonal + 1)
    # STEP 2: Calculate num_rhos (length of rhos array)
    num_rhos = len(rhos)
    # STEP 3: Create accumulator array of zeros with shape (num_rhos, num_thetas)
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)
    # -------------------------------------------------------------------------    
    return accumulator, rhos, thetas

def compute_rho(x: int, y: int, theta: float) -> float:
    """
    Compute the rho value for a point (x, y) at angle theta.
    
    The Hough transform represents a line in polar coordinates:
        rho = x * cos(theta) + y * sin(theta)
    
    Where:
        - rho is the perpendicular distance from origin to the line
        - theta is the angle of the perpendicular with respect to x-axis
        - (x, y) is a point that lies on the line
    
    Args:
        x: x-coordinate (column) of the point
        y: y-coordinate (row) of the point
        theta: angle in RADIANS
    
    Returns:
        rho value (can be negative)
    
    Example:
        >>> compute_rho(2, 2, deg2rad(45))  # Should return approximately 2.83
    """
    # -------------------------------------------------------------------------
    # Use the formula: rho = x * cos(theta) + y * sin(theta)
    rho = x * np.cos(theta) + y * np.sin(theta)
    # -------------------------------------------------------------------------
        
    return rho

def weighted_vote(gradient_value: float, 
                  max_gradient: float = 255.0,
                  min_weight: float = 0.1) -> float:
    """
    Calculate weighted vote based on gradient magnitude.
    
    Instead of each edge pixel contributing equally (vote=1), pixels with
    stronger gradients should contribute more to the accumulator. This makes
    the Hough Transform more robust to noise.
    
    Args:
        gradient_value: Gradient magnitude at the edge pixel (0-255)
        max_gradient: Maximum possible gradient value (default 255)
        min_weight: Minimum vote weight to prevent zero votes (default 0.1)
    
    Returns:
        Vote weight between min_weight and 1.0
    
    Formula:
        weight = max(gradient_value / max_gradient, min_weight)
    
    Example:
        >>> weighted_vote(255)  # Returns 1.0 (strongest edge)
        >>> weighted_vote(127.5)  # Returns 0.5 (medium edge)
        >>> weighted_vote(0)  # Returns 0.1 (minimum weight)
    """
    # -------------------------------------------------------------------------
    # Normalize the gradient value to range [min_weight, 1.0]
    # Use the formula: weight = max(gradient_value / max_gradient, min_weight)
    # -------------------------------------------------------------------------
    
    weight = max(gradient_value / max_gradient, min_weight)
    
    return weight

def hough_line_transform(edge_image, use_weighted=False, gradient_magnitude=None):
    """
    Perform Hough Transform for line detection.
    
    For each edge pixel (x, y), vote for all possible lines that could pass
    through that point by iterating through all theta values and computing
    the corresponding rho.
    
    Args:
        edge_image: Binary image where edge pixels have value > 0
    
    Returns:
        Tuple of (accumulator, rhos, thetas)
    
    Algorithm:
        1. Create accumulator array
        2. Find all edge pixel coordinates
        3. For each edge pixel:
           a. For each theta value:
              - Compute rho
              - Convert rho to accumulator index
              - Increment accumulator (by 1 or by weight)
    """
    # Create accumulator
    accumulator, rhos, thetas = create_hough_accumulator(edge_image.shape)
    
    if accumulator is None:
        raise ValueError("create_hough_accumulator returned None - complete TODO 1 first!")
    
    # Get edge pixel coordinates
    # np.nonzero returns (row_indices, col_indices) = (y_coords, x_coords)
    edge_y, edge_x = np.nonzero(edge_image)
    
    # -------------------------------------------------------------------------
    # For each edge pixel at (x, y):
    #   For each theta index (theta_idx) and theta value:
    #     1. Compute rho using compute_rho(x, y, theta)
    #     2. Convert rho to array index: rho_idx = int(round(rho)) + len(rhos)//2
    #     3. Check bounds: 0 <= rho_idx < len(rhos)
    #     4. If use_weighted and gradient_magnitude is not None:
    #           vote = weighted_vote(gradient_magnitude[y, x])
    #        Else:
    #           vote = 1
    #     5. Add vote to accumulator[rho_idx, theta_idx]
    # -------------------------------------------------------------------------
    
    for i in range(len(edge_x)):
        x = edge_x[i]
        y = edge_y[i]
        for theta_idx, theta in enumerate(thetas):
            rho = compute_rho(x, y, theta)
            rho_idx = int(round(rho)) + len(rhos) // 2
            
            if 0 <= rho_idx < len(rhos):
                if use_weighted and gradient_magnitude is not None:
                    vote = weighted_vote(gradient_magnitude[y, x])
                else:
                    vote = 1
                accumulator[rho_idx, theta_idx] += vote
        
    return accumulator, rhos, thetas

def find_peaks(accumulator: np.ndarray, 
               rhos: np.ndarray, 
               thetas: np.ndarray,
                threshold = None,
               num_peaks: int = 10,
               *,
               suppress_rho_bins: int = 12,
               suppress_theta_bins: int = 12,
               use_suppression: bool = True):
    """
    Find peaks (local maxima) in the Hough accumulator.
    
    Peaks represent the most likely lines in the image.
    
    Args:
        accumulator: 2D Hough accumulator array
        rhos: Array of rho values
        thetas: Array of theta values (in radians)
        threshold: Minimum vote count to consider (default: 50% of max)
        num_peaks: Maximum number of peaks to return
    
    Returns:
        List of tuples (rho, theta_degrees, votes) sorted by votes descending
    
    Algorithm:
        1. If threshold is None, set to 50% of max accumulator value
        2. Optionally pick peaks with neighborhood suppression (NMS) to avoid
           returning many near-duplicate bins for the same physical line
        3. Return top num_peaks results
    """
    # -------------------------------------------------------------------------
    # STEP 1: Set default threshold if None (use 0.5 * np.max(accumulator))
    if threshold is None:
        threshold = 0.5 * np.max(accumulator)

    if use_suppression:
        acc = accumulator.copy()
        peaks: List[Tuple[float, float, int]] = []
        max_iters = int(num_peaks) if num_peaks is not None else 0
        for _ in range(max_iters):
            flat_idx = int(np.argmax(acc))
            votes = int(acc.flat[flat_idx])
            if votes <= threshold:
                break
            rho_idx, theta_idx = np.unravel_index(flat_idx, acc.shape)
            peaks.append((float(rhos[rho_idx]), float(np.rad2deg(thetas[theta_idx])), votes))

            r0 = max(0, int(rho_idx) - int(suppress_rho_bins))
            r1 = min(acc.shape[0], int(rho_idx) + int(suppress_rho_bins) + 1)
            t0 = max(0, int(theta_idx) - int(suppress_theta_bins))
            t1 = min(acc.shape[1], int(theta_idx) + int(suppress_theta_bins) + 1)
            acc[r0:r1, t0:t1] = 0

        return peaks

    # Fallback: return top bins above threshold (can include many near-duplicates).
    rho_idxs, theta_idxs = np.where(accumulator > threshold)
    rho = rhos[rho_idxs]
    theta_degrees = np.rad2deg(thetas[theta_idxs])
    votes = accumulator[rho_idxs, theta_idxs]
    peak_list = list(zip(rho, theta_degrees, votes))
    peak_list.sort(key=lambda x: x[2], reverse=True)
    return peak_list[:num_peaks]


def process_and_draw_lines(
    peaks: List[Tuple[float, float, int]],
    image_shape: Tuple[int, int],
    edge_image: np.ndarray,
    original_image_path: Optional[str] = None,
    output_image_path: Optional[str] = None,
    *,
    rho_tolerance: float = 3.0,
    min_fitted_length_ratio: float = 0.02,
    dedupe_pos_tol_px: float = 2.0,
    min_draw_segment_length_px: float = 8.0,
    angle_tolerance_deg: Optional[float] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Fit Hough peaks back onto edge pixels, while suppressing lines that are
    already represented nearby. Lines are drawn only if they are not too close
    to an already accepted, near-parallel line.
    """
    height, width = image_shape
    output_columns = ["x1", "y1", "x2", "y2"]
    spacing_px = max(float(dedupe_pos_tol_px), 1.0)
    parallel_angle_tol = 8.0
    overlap_tol = 2.0

    def _angular_gap_deg(a: float, b: float) -> float:
        diff = abs(a - b) % 180.0
        return diff if diff <= 90.0 else 180.0 - diff

    def _segment_meta(p1: Tuple[float, float], p2: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        length = float(np.hypot(dx, dy))
        if length <= 1e-6:
            return None
        dir_vec = (dx / length, dy / length)
        normal_vec = (-dir_vec[1], dir_vec[0])
        angle = float(np.rad2deg(np.arctan2(dy, dx)))
        return {
            "p1": (float(p1[0]), float(p1[1])),
            "p2": (float(p2[0]), float(p2[1])),
            "dir": dir_vec,
            "normal": normal_vec,
            "angle": angle,
            "length": length,
        }

    def _overlap_along_dir(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        da = a["dir"]
        proj_a = sorted(
            [
                a["p1"][0] * da[0] + a["p1"][1] * da[1],
                a["p2"][0] * da[0] + a["p2"][1] * da[1],
            ]
        )
        proj_b = sorted(
            [
                b["p1"][0] * da[0] + b["p1"][1] * da[1],
                b["p2"][0] * da[0] + b["p2"][1] * da[1],
            ]
        )
        return not (proj_a[1] < proj_b[0] - overlap_tol or proj_b[1] < proj_a[0] - overlap_tol)

    def _is_near_existing(candidate: Dict[str, Any], accepted: List[Dict[str, Any]]) -> bool:
        for other in accepted:
            if _angular_gap_deg(candidate["angle"], other["angle"]) > parallel_angle_tol:
                continue
            nx, ny = candidate["normal"]
            distance = abs((other["p1"][0] - candidate["p1"][0]) * nx + (other["p1"][1] - candidate["p1"][1]) * ny)
            if distance > spacing_px:
                continue
            if _overlap_along_dir(candidate, other):
                return True
        return False

    def _filter_segments_by_spacing(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in.empty:
            return pd.DataFrame(columns=output_columns)

        df_work = df_in.copy()
        df_work["_length"] = np.hypot(df_work["x2"] - df_work["x1"], df_work["y2"] - df_work["y1"])
        df_work = df_work[df_work["_length"] >= float(min_draw_segment_length_px)]
        df_work = df_work.sort_values("_length", ascending=False)

        kept_rows: List[Dict[str, int]] = []
        kept_meta: List[Dict[str, Any]] = []

        for row in df_work.itertuples(index=False):
            meta = _segment_meta((row.x1, row.y1), (row.x2, row.y2))
            if meta is None:
                continue
            if _is_near_existing(meta, kept_meta):
                continue
            kept_meta.append(meta)
            kept_rows.append({"x1": int(row.x1), "y1": int(row.y1), "x2": int(row.x2), "y2": int(row.y2)})

        return pd.DataFrame(kept_rows, columns=output_columns)

    edge_y, edge_x = np.nonzero(edge_image)
    if edge_x.size == 0:
        print("No edge pixels found; skipping line fitting.")
        if original_image_path is not None and output_image_path is not None:
            output_img = cv2.imread(original_image_path)
            if output_img is None:
                raise FileNotFoundError(f"Could not read original image: {original_image_path}")
            cv2.imwrite(output_image_path, output_img)
        return pd.DataFrame(columns=output_columns)

    if debug:
        print("\n--- Refitting Lines to Edge Pixels ---")
        print(f"Peaks received: {len(peaks)}")

    lines_data: List[Dict[str, Any]] = []
    accepted_meta: List[Dict[str, Any]] = []
    sorted_peaks = sorted(peaks, key=lambda p: p[2] if len(p) > 2 else 0, reverse=True)

    for i, (rho, theta_deg, votes) in enumerate(sorted_peaks):
        if angle_tolerance_deg is not None:
            theta_abs = abs(float(theta_deg))
            is_vertical = theta_abs <= float(angle_tolerance_deg)
            is_horizontal = abs(theta_abs - 90.0) <= float(angle_tolerance_deg)
            if not (is_vertical or is_horizontal):
                continue

        theta_rad = np.deg2rad(theta_deg)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        calculated_rhos = edge_x * cos_theta + edge_y * sin_theta
        matches = np.abs(calculated_rhos - rho) < rho_tolerance

        matched_x = edge_x[matches]
        matched_y = edge_y[matches]
        if matched_x.size == 0:
            continue

        projections = matched_x * cos_theta + matched_y * sin_theta
        idx_min = int(np.argmin(projections))
        idx_max = int(np.argmax(projections))
        p1 = (float(matched_x[idx_min]), float(matched_y[idx_min]))
        p2 = (float(matched_x[idx_max]), float(matched_y[idx_max]))

        segment_meta = _segment_meta(p1, p2)
        if segment_meta is None:
            continue

        if segment_meta["length"] < (min(height, width) * float(min_fitted_length_ratio)):
            continue

        if _is_near_existing(segment_meta, accepted_meta):
            if debug:
                print(f"Skipping peak {i} (theta={theta_deg:.1f}, rho={rho:.1f}) because a nearby line already exists.")
            continue

        orientation = "vertical" if abs(theta_deg) < 45 or abs(theta_deg) > 135 else "horizontal"
        lines_data.append(
            {
                "rho": rho,
                "theta": theta_deg,
                "votes": votes,
                "orientation": orientation,
                "x1": p1[0],
                "y1": p1[1],
                "x2": p2[0],
                "y2": p2[1],
            }
        )
        accepted_meta.append(segment_meta)

    df = pd.DataFrame(lines_data)

    if df.empty:
        print("No lines found after fitting to edges.")
        if original_image_path is not None and output_image_path is not None:
            output_img = cv2.imread(original_image_path)
            if output_img is None:
                raise FileNotFoundError(f"Could not read original image: {original_image_path}")
            cv2.imwrite(output_image_path, output_img)
            print(f"Saved '{output_image_path}' (no fitted segments).")
        return pd.DataFrame(columns=output_columns)

    if debug:
        print(f"Fitted segments before de-dupe: {len(df)}")

    df = _dedupe_fitted_lines(df, pos_tol_px=float(dedupe_pos_tol_px))

    if debug:
        print(f"Fitted segments after de-dupe: {len(df)}")

    # Rewritten drawing logic:
    # 1) Snap Hough-detected line positions to whitespace "gaps" between documents.
    # 2) Clip each line to background-only runs (no drawing over documents).
    # 3) Split segments at vertical/horizontal intersection points so lines stop at separators.
    if original_image_path is not None:
        draw_df = _axis_aligned_border_segments_from_hough(
            df,
            original_image_path=str(original_image_path),
            image_shape=image_shape,
            edge_image=edge_image,
            min_segment_length_px=int(min_draw_segment_length_px),
            debug=bool(debug),
        )
    else:
        draw_df = df[output_columns]

    draw_df = _filter_segments_by_spacing(draw_df)

    if original_image_path is not None:
        output_img = cv2.imread(original_image_path)
        if output_img is None:
            raise FileNotFoundError(f"Could not read original image: {original_image_path}")

        color = (0, 0, 255)
        thickness = 12
        gray = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        # Build a "separator corridor" mask (same spirit as segment generation).
        # Keep it permissive enough so lines don't look faint, but never draw on documents.
        doc_layout = _document_mask_estimate(
            gray,
            edge_image=edge_image,
            doc_intensity_thresh=235,
            open_ksize=3,
            close_ksize=5,
            close_post_ksize=5,
            dilate_ksize=0,
            min_area_frac=0.0015,
        )
        # Slight guard so anti-aliasing / thickness doesn't touch document pixels.
        doc_guard_u8 = cv2.dilate(
            (doc_layout.astype(np.uint8) * 255), np.ones((5, 5), dtype=np.uint8), iterations=1
        )
        safe_bg = _border_connected(doc_guard_u8 == 0)

        line_mask = np.zeros((height, width), dtype=np.uint8)
        for _, row in draw_df.iterrows():
            cv2.line(
                line_mask,
                (int(row["x1"]), int(row["y1"])),
                (int(row["x2"]), int(row["y2"])),
                255,
                int(thickness),
                lineType=cv2.LINE_8,
            )

        paint = (line_mask > 0) & safe_bg
        output_img[paint] = color

        if output_image_path is None:
            output_image_path = "detected_lines_fitted.png"
        cv2.imwrite(output_image_path, output_img)
        print(f"Saved '{output_image_path}' with {len(draw_df)} drawn segments.")

    return draw_df.reset_index(drop=True)

def find_line_intersection(rho1, theta1, rho2, theta2):
    """
    Finds the intersection point (x, y) of two lines given in Rho-Theta form.
    Returns None if lines are parallel.
    """
    # Solve linear system:
    # x * cos(theta1) + y * sin(theta1) = rho1
    # x * cos(theta2) + y * sin(theta2) = rho2
    
    # Pre-compute trigonometric values
    cos1, sin1 = np.cos(np.deg2rad(theta1)), np.sin(np.deg2rad(theta1))
    cos2, sin2 = np.cos(np.deg2rad(theta2)), np.sin(np.deg2rad(theta2))
    
    # Determinant of the coefficient matrix
    det = cos1 * sin2 - sin1 * cos2
    
    # If determinant is close to zero, lines are parallel
    if abs(det) < 1e-6:
        return None
    
    # Use Cramer's rule (or substitution) to solve for x and y
    x = (rho1 * sin2 - rho2 * sin1) / det
    y = (rho2 * cos1 - rho1 * cos2) / det
    
    return int(x), int(y)

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    annotated_dir = script_dir / "annotated_images"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    all_lines: List[pd.DataFrame] = []

    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    candidates: List[Path] = []
    for path in script_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in image_suffixes:
            continue
        if path.parent.name.lower() == "annotated_images":
            continue
        if path.stem.lower().endswith("_annotated") or "_annotated" in path.stem.lower():
            continue
        candidates.append(path)

    # Prefer files named like image_1.png, image_2.png, ... if they exist; otherwise process all images.
    image_paths = sorted([p for p in candidates if p.stem.lower().startswith("image_")], key=lambda p: p.name.lower())
    if not image_paths:
        image_paths = sorted(candidates, key=lambda p: p.name.lower())

    if not image_paths:
        raise SystemExit(f"No input images found in: {script_dir}")

    # ---- Tuning knobs ----
    canny_low = 30
    canny_high = 90
    peak_threshold_ratio = 0.20  # lower -> more candidate lines (can add noise)
    num_peaks = 20               # higher -> consider more candidate lines
    suppress_rho_bins = 14       # higher -> fewer near-duplicate Hough peaks
    suppress_theta_bins = 14
    use_peak_suppression = True
    rho_tolerance = 3.5          # higher -> refit matches more edge pixels
    min_fitted_length_ratio = 0.02
    dedupe_pos_tol_px = 5.0      # higher -> collapse thick/duplicate detections
    min_draw_segment_length_px = 12.0
    angle_tolerance_deg = 12.0   # set None to allow diagonal lines
    debug = False
    # ----------------------

    for image_path in image_paths:
        print(f"\n=== Processing: {image_path.name} ===")
        try:
            image = load_image(str(image_path))

            print("Performing Canny Edge Detection...")
            edges, gradient_magnitude = canny_edge_detection(
                image, low_threshold=canny_low, high_threshold=canny_high
            )
            dilated = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8), iterations=1)

            print("Performing Hough Transform...")
            accumulator, rhos, thetas = hough_line_transform(
                dilated, use_weighted=False, gradient_magnitude=gradient_magnitude
            )

            acc_max = float(np.max(accumulator)) if accumulator.size else 0.0
            print(f"Strongest line has {acc_max} votes")
            threshold_value = acc_max * float(peak_threshold_ratio)

            print("Finding Peaks in Hough Accumulator...")
            peaks = find_peaks(
                accumulator,
                rhos,
                thetas,
                threshold=threshold_value,
                num_peaks=int(num_peaks),
                suppress_rho_bins=int(suppress_rho_bins),
                suppress_theta_bins=int(suppress_theta_bins),
                use_suppression=bool(use_peak_suppression),
            )

            annotated_path = annotated_dir / f"{image_path.stem}_annotated.png"
            df_lines = process_and_draw_lines(
                peaks,
                image.shape,
                dilated,
                original_image_path=str(image_path),
                output_image_path=str(annotated_path),
                rho_tolerance=rho_tolerance,
                min_fitted_length_ratio=min_fitted_length_ratio,
                dedupe_pos_tol_px=dedupe_pos_tol_px,
                min_draw_segment_length_px=min_draw_segment_length_px,
                angle_tolerance_deg=angle_tolerance_deg,
                debug=bool(debug),
            )

            if not df_lines.empty:
                df_lines = df_lines.copy()
                df_lines.insert(0, "filename", image_path.name)
                all_lines.append(df_lines[["filename", "x1", "y1", "x2", "y2"]])

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

    output_csv = script_dir / "lines_data.csv"
    if all_lines:
        result_df = pd.concat(all_lines, ignore_index=True)
    else:
        result_df = pd.DataFrame(columns=["filename", "x1", "y1", "x2", "y2"])
    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved CSV: {output_csv}")
