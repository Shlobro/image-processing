"""
The Broken Robot Vision System - Student Solution Template
==========================================================

RoboVision Industries - Conveyor Belt Inspection System

Complete all TODO items (1-6) and add EXPLAIN comments after each.

Author: [Nadav Robinson]
Student ID: [212544779]
Date: [01/12/2025]

Usage:
    python student_solution.py --test      # Run all tests
    python student_solution.py --demo      # Run visual demo
    python student_solution.py --trace     # Show tracing outputs
"""

import numpy as np
import argparse
from typing import Tuple, List, Optional


# =============================================================================
# PART A: CODE TRACING ANSWERS (Fill in before coding!)
# =============================================================================

"""
TRACE_A1: For pixel (row=2, col=2), theta=45 degrees:
x = ____, y = ____
cos(45) = 0.707, sin(45) = 0.707
rho = ____ * 0.707 + ____ * 0.707 = ____
ANSWER: rho = ____
"""

"""
TRACE_A2:
Pixel (1,1): rho = ____
Pixel (2,2): rho = ____
Pixel (3,3): rho = ____
All three pixels vote for the SAME rho value: ____ (Yes/No)
Accumulator value at (rho=____, theta=45) = ____
"""

"""
TRACE_A3:
Pixel (2,2) vote weight = ____
Pixel (1,1) vote weight = ____
Ratio = ____
"""


# =============================================================================
# CONSTANTS
# =============================================================================

# Theta ranges from -90 to 89 degrees (180 values)
THETA_MIN = -90
THETA_MAX = 89
NUM_THETAS = 180

# Standard values for strong/weak edges (used in weighted voting)
STRONG_EDGE_VALUE = 255
WEAK_EDGE_VALUE = 128


# =============================================================================
# HELPER FUNCTIONS (PROVIDED - DO NOT MODIFY)
# =============================================================================

def generate_test_image(width: int = 100, height: int = 100) -> np.ndarray:
    """
    Generate a test image with a diagonal line and a circle.
    
    Returns:
        Binary edge image (0s and 255s)
    """
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Draw diagonal line from (10,10) to (90,90)
    for i in range(10, 90):
        image[i, i] = 255
    
    # Draw circle at center (50, 50) with radius 20
    center_y, center_x = 50, 50
    radius = 20
    for angle in np.linspace(0, 2*np.pi, 100):
        y = int(center_y + radius * np.sin(angle))
        x = int(center_x + radius * np.cos(angle))
        if 0 <= y < height and 0 <= x < width:
            image[y, x] = 255
    
    return image


def generate_gradient_magnitude(edge_image: np.ndarray) -> np.ndarray:
    """
    Generate synthetic gradient magnitudes for edge pixels.
    For real applications, this would come from Sobel/Canny.
    
    Returns:
        Gradient magnitude image (same shape as input)
    """
    # Simple gradient: higher values toward center of image
    height, width = edge_image.shape
    gradient = np.zeros_like(edge_image, dtype=np.float64)
    
    center_y, center_x = height // 2, width // 2
    max_dist = np.sqrt(center_y**2 + center_x**2)
    
    for y in range(height):
        for x in range(width):
            if edge_image[y, x] > 0:
                dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                gradient[y, x] = 255 * (1 - dist / max_dist) + 50
    
    return gradient


def deg2rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * np.pi / 180.0


def rad2deg(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / np.pi


# =============================================================================
# TODO IMPLEMENTATIONS
# =============================================================================

def create_hough_accumulator(image_shape: Tuple[int, int], 
                              num_thetas: int = NUM_THETAS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    
    HINT: The maximum possible rho value equals the diagonal length of the image.
          Use: diagonal = sqrt(height^2 + width^2)
          Rho ranges from -diagonal to +diagonal.
    """
    height, width = image_shape
    
    # Calculate the diagonal length (maximum possible rho)
    diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))
    
    # Create array of theta values (in radians)
    thetas = np.deg2rad(np.arange(THETA_MIN, THETA_MAX + 1))
    
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
    
    # EXPLAIN: I created the rhos array spanning from -diagonal to +diagonal to cover all possible distances, and initialized the accumulator with zeros to store votes for each (rho, theta) pair.
    
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
    # Remember: theta is already in radians, use np.cos() and np.sin()
    # -------------------------------------------------------------------------
    
    # EXPLAIN: I implemented the standard Hough transform line equation rho = x*cos(theta) + y*sin(theta) to calculate the perpendicular distance from the origin to the line passing through (x,y) at angle theta.
    
    return rho


def hough_line_transform(edge_image: np.ndarray, 
                          use_weighted: bool = False,
                          gradient_magnitude: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Hough Transform for line detection.
    
    For each edge pixel (x, y), vote for all possible lines that could pass
    through that point by iterating through all theta values and computing
    the corresponding rho.
    
    Args:
        edge_image: Binary image where edge pixels have value > 0
        use_weighted: If True, use gradient magnitude for weighted voting
        gradient_magnitude: Gradient magnitude image (required if use_weighted=True)
    
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
    
    # Your implementation here:
    # for i in range(len(edge_x)):
    #     x = edge_x[i]
    #     y = edge_y[i]
    #     for theta_idx, theta in enumerate(thetas):
    #         ... your code ...
    
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
    
    # EXPLAIN: I iterated through every edge pixel and every theta angle, calculated the corresponding rho, mapped it to an array index, and incremented the accumulator bin to vote for that line parameter.
    
    return accumulator, rhos, thetas


def weighted_vote(gradient_value: float, 
                  max_gradient: float = 255.0,
                  min_weight: float = 0.1) -> float:
    """
    Calculate weighted vote based on gradient magnitude.
    
    EXTENSION BEYOND CLASS MATERIAL:
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


def find_peaks(accumulator: np.ndarray, 
               rhos: np.ndarray, 
               thetas: np.ndarray,
               threshold: Optional[float] = None,
               num_peaks: int = 10) -> List[Tuple[float, float, int]]:
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
        2. Find all cells above threshold
        3. Sort by vote count (descending)
        4. Return top num_peaks results
    """
    # -------------------------------------------------------------------------
    # STEP 1: Set default threshold if None (use 0.5 * np.max(accumulator))
    if threshold is None:
        threshold = 0.5 * np.max(accumulator)
    # STEP 2: Find indices where accumulator > threshold using np.where()
    rho_idxs, theta_idxs = np.where(accumulator > threshold)
    # STEP 3: Create list of (rho, theta_degrees, votes) tuples
    #         - rho = rhos[rho_idx]
    #         - theta_degrees = rad2deg(thetas[theta_idx])  # Convert to degrees!
    #         - votes = accumulator[rho_idx, theta_idx]
    rho = rhos[rho_idxs]
    theta_degrees = rad2deg(thetas[theta_idxs])
    votes = accumulator[rho_idxs, theta_idxs]
    # STEP 4: Sort by votes (descending) and return top num_peaks
    peak_list = list(zip(rho, theta_degrees, votes))
    peak_list.sort(key=lambda x: x[2], reverse=True)
    # -------------------------------------------------------------------------
    
    peaks = peak_list[:num_peaks]
        
    return peaks


def hough_circle_transform(edge_image: np.ndarray,
                           radius_min: int = 10,
                           radius_max: int = 50,
                           radius_step: int = 1) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Perform Hough Transform for circle detection.
    
    For circles, the parameter space is 3D: (center_x, center_y, radius)
    
    Circle equation: (x - a)^2 + (y - b)^2 = r^2
    Where (a, b) is the center and r is the radius.
    
    For each edge point (x, y) and each radius r, the possible centers
    form a circle of radius r around (x, y).
    
    Args:
        edge_image: Binary image where edge pixels have value > 0
        radius_min: Minimum radius to detect
        radius_max: Maximum radius to detect
        radius_step: Step size for radius values
    
    Returns:
        Tuple of (accumulator_3d, detected_circles) where:
        - accumulator_3d: 3D array of shape (height, width, num_radii)
        - detected_circles: List of (center_x, center_y, radius, votes)
    """
    height, width = edge_image.shape
    
    # Create radius array
    radii = np.arange(radius_min, radius_max + 1, radius_step)
    num_radii = len(radii)
    
    # Create 3D accumulator: (y, x, radius_index)
    accumulator = np.zeros((height, width, num_radii), dtype=np.int32)
    
    # Get edge pixel coordinates
    edge_y, edge_x = np.nonzero(edge_image)
    
    # -------------------------------------------------------------------------
    # For each edge pixel at (x, y):
    #   For each radius index (r_idx) and radius value (r):
    #     For each angle from 0 to 360 degrees (use 360 steps):
    #       1. Calculate potential center: 
    #          a = x - r * cos(angle)
    #          b = y - r * sin(angle)
    #       2. Round a and b to integers
    #       3. Check bounds: 0 <= a < width and 0 <= b < height
    #       4. Increment accumulator[b, a, r_idx]
    #
    # HINT: Use np.deg2rad() to convert angles to radians
    # OPTIMIZATION: You can use np.arange(0, 360, 1) for angles
    # -------------------------------------------------------------------------
    
    # Your implementation here:
    angels = np.deg2rad(np.arange(0, 360, 1))
    for i in range(len(edge_x)):
        x = edge_x[i]
        y = edge_y[i]
        for r_idx, r in enumerate(radii):
            for angle in angels:
                a = int(round(x - r * np.cos(angle)))
                b = int(round(y - r * np.sin(angle)))
                
                if 0 <= a < width and 0 <= b < height:
                    accumulator[b, a, r_idx] += 1
        
    # EXPLAIN: For each edge point and radius, I iterated through all angles to find possible center coordinates (a,b) using the circle equation, and cast votes in the 3D accumulator for valid centers within image bounds.
    
    # Find circles (peaks in 3D accumulator)
    # This part is provided - finds local maxima
    detected_circles = []
    threshold = np.max(accumulator) * 0.5 if np.max(accumulator) > 0 else 0
    
    for r_idx, r in enumerate(radii):
        for y in range(height):
            for x in range(width):
                if accumulator[y, x, r_idx] > threshold:
                    detected_circles.append((x, y, r, accumulator[y, x, r_idx]))
    
    # Sort by votes
    detected_circles.sort(key=lambda c: c[3], reverse=True)
    
    return accumulator, detected_circles[:10]


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_create_hough_accumulator():
    """Test accumulator creation."""
    acc, rhos, thetas = create_hough_accumulator((100, 100))
    
    if acc is None:
        return False, "Accumulator is None - complete TODO 1"
    
    # Diagonal of 100x100 image is ~141
    expected_diagonal = int(np.ceil(np.sqrt(100**2 + 100**2)))  # 142
    expected_num_rhos = 2 * expected_diagonal + 1  # -142 to +142 = 285
    
    if acc.shape[1] != NUM_THETAS:
        return False, f"Expected {NUM_THETAS} theta bins, got {acc.shape[1]}"
    
    if len(rhos) != expected_num_rhos:
        return False, f"Expected {expected_num_rhos} rho values, got {len(rhos)}"
    
    if acc.shape[0] != len(rhos):
        return False, f"Accumulator rows ({acc.shape[0]}) should equal len(rhos) ({len(rhos)})"
    
    return True, "PASS"


def test_compute_rho():
    """Test rho calculation."""
    rho = compute_rho(2, 2, deg2rad(45))
    
    if rho is None:
        return False, "rho is None - complete TODO 2"
    
    expected = 2 * np.cos(deg2rad(45)) + 2 * np.sin(deg2rad(45))  # ~2.83
    
    if abs(rho - expected) > 0.01:
        return False, f"Expected rho ~{expected:.2f}, got {rho:.2f}"
    
    return True, "PASS"


def test_hough_line_transform():
    """Test Hough line transform."""
    # Create simple diagonal line image
    image = np.zeros((10, 10), dtype=np.uint8)
    for i in range(10):
        image[i, i] = 255
    
    try:
        acc, rhos, thetas = hough_line_transform(image)
    except ValueError as e:
        return False, str(e)
    
    if acc is None:
        return False, "Accumulator is None - complete TODO 3"
    
    # Check that diagonal line creates peak near theta=45 degrees
    # theta=-45 in our range (which is 45 degrees from vertical)
    max_idx = np.unravel_index(np.argmax(acc), acc.shape)
    max_theta_deg = rad2deg(thetas[max_idx[1]])
    
    # For a diagonal from (0,0) to (9,9), the perpendicular angle is 45 degrees
    if not (-50 < max_theta_deg < 50):
        return False, f"Peak at unexpected theta: {max_theta_deg:.1f} degrees"
    
    return True, "PASS"


def test_weighted_vote():
    """Test weighted voting."""
    w1 = weighted_vote(255)
    w2 = weighted_vote(127.5)
    w3 = weighted_vote(0)
    
    if w1 is None or w2 is None or w3 is None:
        return False, "weighted_vote returned None - complete TODO 4"
    
    if abs(w1 - 1.0) > 0.01:
        return False, f"Expected weight=1.0 for gradient=255, got {w1}"
    
    if abs(w2 - 0.5) > 0.01:
        return False, f"Expected weight=0.5 for gradient=127.5, got {w2}"
    
    if w3 < 0.1:
        return False, f"Expected minimum weight=0.1, got {w3}"
    
    return True, "PASS"


def test_find_peaks():
    """Test peak finding."""
    # Create fake accumulator with known peaks
    acc = np.zeros((100, 180))
    acc[50, 90] = 100  # Peak at rho_idx=50, theta_idx=90
    acc[30, 45] = 80   # Secondary peak
    
    rhos = np.arange(-50, 50)
    thetas = np.deg2rad(np.arange(-90, 90))
    
    peaks = find_peaks(acc, rhos, thetas, threshold=50)
    
    if not peaks:
        return False, "No peaks found - complete TODO 5"
    
    # Check that highest peak is found
    top_peak = peaks[0]
    if top_peak[2] != 100:
        return False, f"Expected top peak votes=100, got {top_peak[2]}"
    
    return True, "PASS"


def test_hough_circle_transform():
    """Test Hough circle transform."""
    # Create image with a circle
    image = np.zeros((100, 100), dtype=np.uint8)
    center_y, center_x = 50, 50
    radius = 20
    
    for angle in np.linspace(0, 2*np.pi, 100):
        y = int(center_y + radius * np.sin(angle))
        x = int(center_x + radius * np.cos(angle))
        if 0 <= y < 100 and 0 <= x < 100:
            image[y, x] = 255
    
    acc, circles = hough_circle_transform(image, radius_min=15, radius_max=25)
    
    if acc is None:
        return False, "Accumulator is None - complete TODO 6"
    
    if not circles:
        return False, "No circles detected - check TODO 6 implementation"
    
    # Check if detected circle is close to actual
    best_circle = circles[0]
    cx, cy, r, votes = best_circle
    
    if abs(cx - 50) > 5 or abs(cy - 50) > 5:
        return False, f"Expected center near (50,50), got ({cx},{cy})"
    
    if abs(r - 20) > 3:
        return False, f"Expected radius ~20, got {r}"
    
    return True, "PASS"


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("create_hough_accumulator", test_create_hough_accumulator),
        ("compute_rho", test_compute_rho),
        ("hough_line_transform", test_hough_line_transform),
        ("weighted_vote", test_weighted_vote),
        ("find_peaks", test_find_peaks),
        ("hough_circle_transform", test_hough_circle_transform),
    ]
    
    passed = 0
    total = len(tests)
    
    print("=" * 50)
    print("Running Robot Vision System Tests")
    print("=" * 50)
    
    for name, test_func in tests:
        try:
            success, message = test_func()
            status = "PASS" if success else "FAIL"
            if success:
                passed += 1
            print(f"[TEST] {name}: {status} - {message}")
        except Exception as e:
            print(f"[TEST] {name}: ERROR - {str(e)}")
    
    print("=" * 50)
    print(f"[RESULT] {passed}/{total} tests passed")
    print("=" * 50)
    
    return passed == total


def demo():
    """Run visual demonstration."""
    print("Generating test image...")
    image = generate_test_image()
    gradient = generate_gradient_magnitude(image)
    
    print(f"Image shape: {image.shape}")
    print(f"Number of edge pixels: {np.count_nonzero(image)}")
    
    print("\nRunning Hough Line Transform...")
    try:
        acc, rhos, thetas = hough_line_transform(image)
        print(f"Accumulator shape: {acc.shape}")
        print(f"Max votes: {np.max(acc)}")
        
        peaks = find_peaks(acc, rhos, thetas)
        print(f"\nTop detected lines:")
        for i, (rho, theta, votes) in enumerate(peaks[:5]):
            print(f"  Line {i+1}: rho={rho:.1f}, theta={theta:.1f}°, votes={votes}")
    except Exception as e:
        print(f"Line detection failed: {e}")
    
    print("\nRunning Hough Circle Transform...")
    try:
        acc_3d, circles = hough_circle_transform(image)
        print(f"\nTop detected circles:")
        for i, (cx, cy, r, votes) in enumerate(circles[:3]):
            print(f"  Circle {i+1}: center=({cx},{cy}), radius={r}, votes={votes}")
    except Exception as e:
        print(f"Circle detection failed: {e}")


def show_tracing():
    """Show tracing information for Part A."""
    print("=" * 50)
    print("CODE TRACING HELPER")
    print("=" * 50)
    
    print("\nFor pixel at (row=2, col=2):")
    print("  x (column) = 2")
    print("  y (row) = 2")
    print(f"  cos(45°) = {np.cos(deg2rad(45)):.4f}")
    print(f"  sin(45°) = {np.sin(deg2rad(45)):.4f}")
    
    print("\nCalculate rho = x*cos(45°) + y*sin(45°)")
    print("  rho = 2 * 0.7071 + 2 * 0.7071")
    print(f"  rho = {2 * np.cos(deg2rad(45)) + 2 * np.sin(deg2rad(45)):.4f}")
    
    print("\n" + "=" * 50)
    print("Now complete the TRACE answers in the code!")
    print("=" * 50)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Vision System - Student Solution")
    parser.add_argument("--test", action="store_true", help="Run all tests")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--trace", action="store_true", help="Show tracing helper")
    
    args = parser.parse_args()
    
    if args.test:
        run_all_tests()
    elif args.demo:
        demo()
    elif args.trace:
        show_tracing()
    else:
        print("Usage: python student_solution.py [--test | --demo | --trace]")
        print("  --test   Run all tests to check your implementation")
        print("  --demo   Run visual demonstration")
        print("  --trace  Show tracing helper for Part A")
