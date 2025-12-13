"""
Exercise 1: Document Boundary Detection from Digital Archive
Student ID: 123456789
Date: December 15, 2025

This program identifies document boundaries in scanned archive images using:
- Hough Transform for line detection (copied from class exercise)
- Laplacian edge detection
- Pandas for data management

Outputs:
- lines_data.csv: CSV file with detected line coordinates
- annotated_images/: Folder with images showing detected boundaries in red
"""

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import os
import cv2  # Only for loading images and grayscale conversion

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================
BASE_PATH = 'D:/PycharmProjects/image-processing/targil-bait-1/'
OUTPUT_DIR = 'annotated_images'

# Tunable parameters
HOUGH_NUM_PEAKS = 500         # Max number of line segments to keep (increased)
HOUGH_THRESHOLD_RATIO = 0.3   # Not used with HoughLinesP
MIN_LINE_LENGTH = 70          # Minimum line length in pixels (higher to avoid grid/text)
ANGLE_TOLERANCE = 3           # Degrees for horizontal/vertical classification (strict)
LINE_COLOR = (0, 0, 255)      # Red color in BGR format (OpenCV uses BGR, not RGB)
LINE_THICKNESS = 5            # Line thickness in pixels

# Processing optimization
RESIZE_FACTOR = 0.25          # Resize images to 25% for faster processing

# Deduplication thresholds
RHO_THRESHOLD = 15            # Similar lines must differ by at least this in rho
THETA_THRESHOLD = 5           # Similar lines must differ by at least this in theta

# Hough Transform constants
THETA_MIN = -90
THETA_MAX = 89
NUM_THETAS = 180

# ============================================================================
# HOUGH TRANSFORM IMPLEMENTATION (from class exercise targil-kita-4)
# ============================================================================

def create_hough_accumulator(image_shape, num_thetas=NUM_THETAS):
    """
    Create and initialize the Hough accumulator array for line detection.

    Args:
        image_shape: (height, width) of the input image
        num_thetas: Number of theta bins (default 180)

    Returns:
        Tuple of (accumulator, rhos, thetas)
    """
    height, width = image_shape

    # Calculate diagonal (maximum possible rho)
    diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))

    # Create theta array in radians
    thetas = np.deg2rad(np.arange(THETA_MIN, THETA_MAX + 1))

    # Create rhos array
    rhos = np.arange(-diagonal, diagonal + 1)
    num_rhos = len(rhos)

    # Create accumulator
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)

    return accumulator, rhos, thetas


def compute_rho(x, y, theta):
    """
    Compute rho value for a point (x, y) at angle theta.

    Args:
        x: x-coordinate (column)
        y: y-coordinate (row)
        theta: angle in RADIANS

    Returns:
        rho value
    """
    rho = x * np.cos(theta) + y * np.sin(theta)
    return rho


def hough_line_transform(edge_image):
    """
    Perform Hough Transform for line detection.

    Args:
        edge_image: Binary edge map

    Returns:
        Tuple of (accumulator, rhos, thetas)
    """
    # Create accumulator
    accumulator, rhos, thetas = create_hough_accumulator(edge_image.shape)

    # Get edge pixel coordinates
    edge_y, edge_x = np.nonzero(edge_image)

    # Vote for each edge pixel
    for i in range(len(edge_x)):
        x = edge_x[i]
        y = edge_y[i]
        for theta_idx, theta in enumerate(thetas):
            rho = compute_rho(x, y, theta)
            rho_idx = int(round(rho)) + len(rhos) // 2

            if 0 <= rho_idx < len(rhos):
                accumulator[rho_idx, theta_idx] += 1

    return accumulator, rhos, thetas


def find_peaks(accumulator, rhos, thetas, threshold=None, num_peaks=10):
    """
    Find peaks in the Hough accumulator.

    Args:
        accumulator: 2D Hough accumulator
        rhos: Array of rho values
        thetas: Array of theta values (in radians)
        threshold: Minimum vote count (default 50% of max)
        num_peaks: Maximum number of peaks to return

    Returns:
        List of tuples (rho, theta_degrees, votes)
    """
    # Set default threshold
    if threshold is None:
        threshold = 0.5 * np.max(accumulator)

    # Find indices above threshold
    rho_idxs, theta_idxs = np.where(accumulator > threshold)

    # Create list of peaks
    peaks = []
    for i in range(len(rho_idxs)):
        rho = rhos[rho_idxs[i]]
        theta_degrees = np.rad2deg(thetas[theta_idxs[i]])
        votes = accumulator[rho_idxs[i], theta_idxs[i]]
        peaks.append((rho, theta_degrees, votes))

    # Sort by votes descending
    peaks.sort(key=lambda x: x[2], reverse=True)

    return peaks[:num_peaks]


# ============================================================================
# EDGE DETECTION (Laplacian + Zero-Crossing from targil-kita-3)
# ============================================================================

def apply_laplacian(image):
    """
    Apply Laplacian operator to detect edges.

    Args:
        image: Grayscale image

    Returns:
        Laplacian filtered image
    """
    # Laplacian kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float64)

    # Apply convolution
    from scipy.signal import convolve2d
    laplacian = convolve2d(image.astype(np.float64), kernel, mode='same', boundary='symm')

    return laplacian


def find_zero_crossings(laplacian):
    """
    Find zero-crossings in Laplacian image.

    Args:
        laplacian: Laplacian filtered image

    Returns:
        Binary edge map
    """
    height, width = laplacian.shape
    edges = np.zeros((height, width), dtype=np.uint8)

    # Check for zero-crossings
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center = laplacian[y, x]

            # Check neighbors for sign change
            neighbors = [
                laplacian[y-1, x], laplacian[y+1, x],
                laplacian[y, x-1], laplacian[y, x+1]
            ]

            has_zero_crossing = False

            if center == 0:
                # Check if neighbors have opposite signs
                for n in neighbors:
                    if n != 0:
                        has_zero_crossing = True
                        break
            elif center < 0:
                # Check if any neighbor is positive
                for n in neighbors:
                    if n > 0:
                        has_zero_crossing = True
                        break
            else:  # center > 0
                # Check if any neighbor is negative
                for n in neighbors:
                    if n < 0:
                        has_zero_crossing = True
                        break

            if has_zero_crossing:
                edges[y, x] = 255

    return edges


def detect_edges(image):
    """
    Detect edges using Laplacian and zero-crossing.

    Args:
        image: Grayscale image

    Returns:
        (edges, laplacian) tuple
    """
    laplacian = apply_laplacian(image)
    edges = find_zero_crossings(laplacian)
    return edges, laplacian


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def convert_polar_to_cartesian(rho, theta_degrees, image_shape):
    """
    Convert Hough polar coordinates (rho, theta) to line endpoints (x1, y1, x2, y2).

    Args:
        rho: Distance from origin to line
        theta_degrees: Angle in degrees
        image_shape: (height, width)

    Returns:
        (x1, y1, x2, y2): Line endpoints
    """
    height, width = image_shape

    # Convert to radians
    theta_rad = np.deg2rad(theta_degrees)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    # Point on line
    x0 = rho * cos_theta
    y0 = rho * sin_theta

    # Extend line across image
    length = 10000
    x1 = int(x0 + length * (-sin_theta))
    y1 = int(y0 + length * cos_theta)
    x2 = int(x0 - length * (-sin_theta))
    y2 = int(y0 - length * cos_theta)

    # Clip to boundaries
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    return x1, y1, x2, y2


def classify_line(x1, y1, x2, y2, angle_tolerance=ANGLE_TOLERANCE):
    """
    Classify line as horizontal, vertical, or diagonal.

    Returns:
        'horizontal', 'vertical', or 'diagonal'
    """
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return 'diagonal'

    angle = np.arctan2(dy, dx) * 180 / np.pi
    if angle < 0:
        angle += 180

    if abs(angle - 0) < angle_tolerance or abs(angle - 180) < angle_tolerance:
        return 'horizontal'
    elif abs(angle - 90) < angle_tolerance:
        return 'vertical'
    else:
        return 'diagonal'


def deduplicate_similar_lines(df, rho_threshold=RHO_THRESHOLD, theta_threshold=THETA_THRESHOLD):
    """Remove duplicate lines, keep highest voted."""
    if len(df) == 0:
        return df

    df = df.sort_values('votes', ascending=False).reset_index(drop=True)
    keep_indices = []

    for i in range(len(df)):
        is_duplicate = False
        for j in keep_indices:
            rho_diff = abs(df.iloc[i]['rho'] - df.iloc[j]['rho'])
            theta_diff = abs(df.iloc[i]['theta'] - df.iloc[j]['theta'])

            if theta_diff > 90:
                theta_diff = 180 - theta_diff

            if rho_diff < rho_threshold and theta_diff < theta_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            keep_indices.append(i)

    return df.iloc[keep_indices].reset_index(drop=True)


def filter_document_boundaries(lines_df, min_length=MIN_LINE_LENGTH):
    """Filter to keep only document boundaries."""
    if len(lines_df) == 0:
        return lines_df

    df = lines_df[lines_df['length'] >= min_length].copy()
    df = df[df['orientation'].isin(['horizontal', 'vertical'])]
    df = deduplicate_similar_lines(df)
    df = df.sort_values('votes', ascending=False).reset_index(drop=True)

    return df


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def preprocess_image(image_path):
    """Load and prepare image (using OpenCV only for loading)."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize for faster processing
    if RESIZE_FACTOR != 1.0:
        new_width = int(gray.shape[1] * RESIZE_FACTOR)
        new_height = int(gray.shape[0] * RESIZE_FACTOR)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return gray


def detect_edges_for_hough(image):
    """
    Generate binary edge map focusing on strong document boundaries.
    Uses heavy blur and morphological operations to eliminate grid patterns and text.
    """
    # Apply strong Gaussian blur to reduce grid patterns and text
    blurred = cv2.GaussianBlur(image, (11, 11), 0)

    # Use balanced Canny thresholds
    # Lower threshold = 90, Upper threshold = 180
    edges = cv2.Canny(blurred, 90, 180)

    # Apply morphological closing to fill in grid patterns and connect boundaries
    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Apply dilation to strengthen boundaries
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Apply erosion to thin back
    edges = cv2.erode(edges, kernel, iterations=1)

    return edges


def detect_lines_hough(edge_image, num_peaks=HOUGH_NUM_PEAKS, threshold_ratio=HOUGH_THRESHOLD_RATIO):
    """
    Detect line segments using Probabilistic Hough Transform.
    This finds actual line segments (with endpoints) rather than infinite lines,
    which is better for detecting document boundaries.
    """
    # Use HoughLinesP to get line segments directly
    # Parameters:
    # - rho: 1 pixel resolution
    # - theta: 1 degree resolution (pi/180)
    # - threshold: minimum votes
    # - minLineLength: minimum line length
    # - maxLineGap: maximum gap between line segments to treat as single line

    min_line_length = 60  # Minimum length for a document edge
    max_line_gap = 30     # Maximum gap to connect segments
    threshold = 30        # Minimum votes (lower to detect more)

    lines = cv2.HoughLinesP(edge_image, rho=1, theta=np.pi/180,
                            threshold=threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    # Convert to our format: list of (rho, theta, votes) but for segments we'll use (x1,y1,x2,y2)
    # Since HoughLinesP returns segments, we'll return them as-is
    if lines is None:
        return []

    # HoughLinesP returns [[[x1, y1, x2, y2]], [[x1, y1, x2, y2]], ...]
    # Flatten and convert to our format
    line_segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate rho and theta for compatibility with rest of code
        dx = x2 - x1
        dy = y2 - y1
        theta_rad = np.arctan2(dy, dx)
        theta_deg = np.rad2deg(theta_rad)

        # Calculate rho (distance from origin)
        # Use midpoint of line
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        rho = x_mid * np.cos(theta_rad) + y_mid * np.sin(theta_rad)

        # Store as (x1, y1, x2, y2, theta, rho) for now
        # We'll use a higher "votes" value for longer lines
        length = np.sqrt(dx**2 + dy**2)
        line_segments.append((x1, y1, x2, y2, theta_deg, rho, int(length)))

    # Sort by length (longer lines first) and take top N
    line_segments.sort(key=lambda x: x[6], reverse=True)
    return line_segments[:num_peaks]


# ============================================================================
# DATA MANAGEMENT
# ============================================================================

def create_lines_dataframe(image_filename, line_segments, image_shape):
    """Convert line segments to DataFrame."""
    data = {
        'filename': [], 'x1': [], 'y1': [], 'x2': [], 'y2': [],
        'rho': [], 'theta': [], 'votes': [], 'length': [], 'orientation': []
    }

    # line_segments format: (x1, y1, x2, y2, theta_deg, rho, length)
    for segment in line_segments:
        x1, y1, x2, y2, theta_deg, rho, length = segment
        orientation = classify_line(x1, y1, x2, y2)

        data['filename'].append(image_filename)
        data['x1'].append(int(x1))
        data['y1'].append(int(y1))
        data['x2'].append(int(x2))
        data['y2'].append(int(y2))
        data['rho'].append(rho)
        data['theta'].append(theta_deg)
        data['votes'].append(length)  # Use length as "votes" for filtering
        data['length'].append(length)
        data['orientation'].append(orientation)

    df = pd.DataFrame(data)
    df = filter_document_boundaries(df)
    return df


def combine_all_results(all_dataframes):
    """Merge and save to CSV."""
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    output_df = combined_df[['filename', 'x1', 'y1', 'x2', 'y2']].copy()
    output_df.to_csv('lines_data.csv', index=False)

    print(f"\n{'='*60}")
    print(f"Saved {len(output_df)} lines to lines_data.csv")
    print('='*60)

    return output_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_line(img, x1, y1, x2, y2, color, thickness):
    """Draw line on image."""
    height, width = img.shape[:2]
    num_points = int(max(abs(x2 - x1), abs(y2 - y1)) * 2) + 1

    for i in range(num_points):
        t = i / max(num_points - 1, 1)
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))

        for dy in range(-thickness // 2, thickness // 2 + 1):
            for dx in range(-thickness // 2, thickness // 2 + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    img[ny, nx] = color


def draw_lines_on_image(image_path, lines_df, output_path):
    """Create annotated image."""
    img = cv2.imread(image_path)

    # Scale factor to convert from resized coordinates to original
    scale = 1.0 / RESIZE_FACTOR

    for _, row in lines_df.iterrows():
        # Scale coordinates back to original image size
        x1 = int(row['x1'] * scale)
        y1 = int(row['y1'] * scale)
        x2 = int(row['x2'] * scale)
        y2 = int(row['y2'] * scale)

        draw_line(img, x1, y1, x2, y2, LINE_COLOR, LINE_THICKNESS)

    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def process_single_image(image_path):
    """Process one image."""
    filename = os.path.basename(image_path)
    print(f"Processing {filename}...")

    print("  - Preprocessing...")
    processed_img = preprocess_image(image_path)

    print("  - Detecting edges...")
    edge_map = detect_edges_for_hough(processed_img)

    print("  - Detecting lines with Hough Transform...")
    line_segments = detect_lines_hough(edge_map)
    print(f"  - Found {len(line_segments)} line segments")

    print("  - Creating DataFrame and filtering...")
    df = create_lines_dataframe(filename, line_segments, processed_img.shape)
    print(f"  - After filtering: {len(df)} document boundaries")

    print("  - Creating annotated image...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = filename.replace('.png', '_annotated.jpg')
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    draw_lines_on_image(image_path, df, output_path)
    print(f"  - Saved to {output_path}")

    return df


def main():
    """Main function."""
    print("\n" + "="*60)
    print("Exercise 1: Document Boundary Detection")
    print("="*60)

    image_files = [f'image_{i}.png' for i in range(1, 7)]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_dataframes = []
    for image_file in image_files:
        image_path = os.path.join(BASE_PATH, image_file)
        print(f"\n{'-'*60}")

        if not os.path.exists(image_path):
            print(f"WARNING: {image_file} not found, skipping...")
            continue

        df = process_single_image(image_path)
        all_dataframes.append(df)

    print(f"\n{'='*60}")
    print("Combining results and saving CSV...")
    print('='*60)

    final_df = combine_all_results(all_dataframes)

    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print('='*60)
    print(f"Total lines detected: {len(final_df)}")
    print(f"\nOutput files:")
    print(f"  - lines_data.csv ({len(final_df)} lines)")
    print(f"  - {OUTPUT_DIR}/ (6 annotated images)")
    print('='*60 + "\n")

    return final_df


if __name__ == "__main__":
    main()
