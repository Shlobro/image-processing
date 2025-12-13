"""
Frequency Forensics: Detecting Image Tampering Using FFT
=========================================================

Course: Digital Image Processing
Exercise: In-class FFT Application (30 minutes)

Author: [Your Name]
Student ID: [Your ID]

Instructions:
- Complete all TODO sections (numbered 1-4)
- Do NOT modify function signatures
- Follow the exact output format specified
- Run the tests at the bottom to verify your solution
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any

from numpy import ndarray


# =============================================================================
# HELPER FUNCTIONS (DO NOT MODIFY)
# =============================================================================

def create_test_image_clean(size: int = 128) -> np.ndarray:
    """Create a clean test image with natural patterns."""
    np.random.seed(42)
    x = np.linspace(0, 4 * np.pi, size)
    y = np.linspace(0, 4 * np.pi, size)
    xx, yy = np.meshgrid(x, y)
    image = np.sin(xx) * np.cos(yy) * 0.3 + 0.5
    image += np.random.normal(0, 0.05, image.shape)
    return np.clip(image, 0, 1)


def create_test_image_tampered(size: int = 128) -> np.ndarray:
    """Create a tampered test image with artificial periodic patterns."""
    np.random.seed(42)
    image = create_test_image_clean(size)
    # Add artificial periodic pattern (simulating copy-paste tampering)
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)
    # High-frequency periodic tampering pattern
    tampering = 0.15 * np.sin(2 * np.pi * xx * 8 / size) * np.sin(2 * np.pi * yy * 8 / size)
    tampering += 0.1 * np.cos(2 * np.pi * xx * 12 / size)
    image = image + tampering
    return np.clip(image, 0, 1)


def load_image(image_path: str) -> np.ndarray:
    """Load an image from file path. Returns grayscale float array [0,1]."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert('L')
        return np.array(img, dtype=np.float64) / 255.0
    except ImportError:
        # Fallback: create synthetic image for testing
        print("[WARNING] PIL not available, using synthetic test image")
        return create_test_image_clean(128)
    except FileNotFoundError:
        print(f"[WARNING] Image not found: {image_path}, using synthetic test image")
        return create_test_image_clean(128)


# =============================================================================
# TODO 1: COMPUTE MAGNITUDE SPECTRUM
# =============================================================================

def compute_magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Compute the centered, log-transformed, normalized magnitude spectrum.
    
    This function transforms an image from spatial domain to frequency domain
    and prepares it for analysis by:
    1. Computing 2D FFT
    2. Shifting zero-frequency to center
    3. Computing magnitude (absolute value)
    4. Applying log transformation for better visualization
    5. Normalizing to [0, 1] range
    
    Args:
        image: 2D numpy array (grayscale image), values in [0, 1]
        
    Returns:
        spectrum: 2D numpy array, normalized magnitude spectrum in [0, 1]
        
    Output format (print this):
        [SPECTRUM] Shape: (H, W), Min: X.XXXX, Max: X.XXXX
    """
    # TODO 1.1: Apply 2D FFT to the image
    # Hint: Use np.fft.fft2()
    fft_result = np.fft.fft2(image)
    
    # TODO 1.2: Shift zero-frequency component to the center
    # Hint: Use np.fft.fftshift()
    fft_shifted = np.fft.fftshift(fft_result)
    
    # TODO 1.3: Compute the magnitude (absolute value)
    # Hint: Use np.abs()
    magnitude = np.abs(fft_shifted)
    
    # TODO 1.4: Apply log transformation for better dynamic range
    # Hint: Use np.log1p() which computes log(1 + x) safely
    log_magnitude = np.log1p(magnitude)
    
    # TODO 1.5: Normalize to [0, 1] range
    # Hint: (x - min) / (max - min), handle edge case where max == min
    min_val = np.min(log_magnitude)
    max_val = np.max(log_magnitude)
    if max_val == min_val:
        spectrum = np.zeros_like(log_magnitude)
    else:
        spectrum = (log_magnitude - min_val) / (max_val - min_val)  # Replace with your code
    
    # TODO 1.6: Print the required output format
    # Format: [SPECTRUM] Shape: (H, W), Min: X.XXXX, Max: X.XXXX
    H, W = spectrum.shape
    min_s = np.min(spectrum)
    max_s = np.max(spectrum)
    print(f"[SPECTRUM] Shape: ({H}, {W}), Min: {min_s:.4f}, Max: {max_s:.4f}")
    
    return spectrum


# =============================================================================
# TODO 2: DETECT FREQUENCY PEAKS
# =============================================================================

def detect_frequency_peaks(spectrum: np.ndarray, 
                           threshold_percentile: float = 99.5) -> list[tuple[Any, Any, ndarray[tuple[int, ...], Any]]]:
    """
    Detect anomalous peaks in the frequency spectrum.
    
    Peaks far from the center (DC component) indicate periodic patterns
    that may be artifacts of image tampering or artificial generation.
    
    Args:
        spectrum: 2D numpy array, normalized magnitude spectrum
        threshold_percentile: Percentile threshold for peak detection (default 99.5)
        
    Returns:
        peaks: List of (row, col, magnitude) tuples, sorted by magnitude descending
        
    Output format (print this):
        [PEAKS] Found N peaks above XX.X percentile threshold
        [PEAK] Position: (row, col), Magnitude: X.XXXX
        (Print top 5 peaks maximum)
    """
    h, w = spectrum.shape
    center_r, center_c = h // 2, w // 2
    
    # TODO 2.1: Create a mask to exclude the DC component (center region)
    # The center region should be approximately 5% of the image size (radius)
    # Hint: Create a circular mask using distance from center
    center_radius = int(min(h, w) * 0.05)
    
    # Create coordinate grids
    rows = np.arange(h)
    cols = np.arange(w)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
    
    # TODO: Calculate distance from center for each pixel
    distance_from_center = distance_from_center = np.sqrt((row_grid - center_r)**2 + (col_grid - center_c)**2)
    
    # TODO: Create mask (True for pixels OUTSIDE the center region)
    mask = distance_from_center > center_radius
    
    # TODO 2.2: Get spectrum values outside the center region
    masked_values = spectrum[mask]
    
    # TODO 2.3: Calculate the threshold value using np.percentile
    threshold = np.percentile(masked_values, threshold_percentile)
    
    # TODO 2.4: Find all peaks above threshold (outside center region)
    # Hint: Use np.where() with combined conditions
    peak_positions = np.where((spectrum > threshold) & mask)
    
    # TODO 2.5: Create list of (row, col, magnitude) tuples
    peaks = []
    # Your code here: iterate through peak_positions and build the list
    for r, c in zip(*peak_positions):
        peaks.append((r, c, spectrum[r, c]))
    
    # TODO 2.6: Sort peaks by magnitude (descending order)
    # Hint: Use sorted() with key parameter
    peaks = sorted(peaks, key=lambda p: p[2], reverse=True)
    
    # TODO 2.7: Print the required output format
    # [PEAKS] Found N peaks above XX.X percentile threshold
    # [PEAK] Position: (row, col), Magnitude: X.XXXX (top 5 only)
    n_peaks = len(peaks)
    print(f"[PEAKS] Found {n_peaks} peaks above {threshold_percentile:.1f} percentile threshold")
    for i, (r, c, mag) in enumerate(peaks[:5]):
        print(f"[PEAK] Position: ({r}, {c}), Magnitude: {mag:.4f}")
    
    return peaks


# =============================================================================
# TODO 3: CALCULATE SUSPICION SCORE
# =============================================================================

def calculate_suspicion_score(peaks: List[Tuple[int, int, float]], 
                              spectrum_shape: Tuple[int, int]) -> Dict:
    """
    Calculate a suspicion score based on detected frequency peaks.
    
    Scoring components:
    1. Base score: min(num_peaks * 10, 50) points
    2. Symmetry bonus: +20 if symmetric peak pairs exist
    3. Edge bonus: +15 if peaks are near spectrum edges
    4. Intensity bonus: min(avg_magnitude * 20, 15) points
    
    Total score is clamped to [0, 100].
    
    Args:
        peaks: List of (row, col, magnitude) tuples
        spectrum_shape: (height, width) of the spectrum
        
    Returns:
        result: Dictionary with keys:
            - 'score': int (0-100)
            - 'level': str ('CLEAN', 'LOW', 'MEDIUM', 'HIGH')
            - 'base_score': int
            - 'symmetry_bonus': int
            - 'edge_bonus': int
            - 'intensity_bonus': int
            
    Output format (print this):
        [SCORE] Suspicion Score: XX/100 (LEVEL)
    """
    h, w = spectrum_shape
    center_r, center_c = h // 2, w // 2
    
    # TODO 3.1: Calculate base score from number of peaks
    # Formula: min(num_peaks * 10, 50)
    num_peaks = len(peaks)
    base_score = min(num_peaks * 10, 50)
    
    # TODO 3.2: Check for symmetric peak pairs and calculate symmetry bonus
    # A peak at (r, c) has a symmetric partner at (h - r, w - c) relative to center
    # Bonus: +20 if at least one symmetric pair exists
    symmetry_bonus = 0
    # Your code here: check for symmetric pairs
    peak_set = {(p[0], p[1]) for p in peaks}
    for r, c, _ in peaks:
        sym_r = h - 1 - r
        sym_c = w - 1 - c
        if (sym_r, sym_c) in peak_set and (sym_r, sym_c) != (r, c):
            symmetry_bonus = 20
            break

    # TODO 3.3: Check if any peaks are near the edges (high frequency)
    # "Near edge" means within 10% of the image size from any edge
    # Bonus: +15 if any peak is near an edge
    edge_bonus = 0
    edge_threshold = min(h, w) * 0.1
    # Your code here: check for edge peaks
    for r, c, _ in peaks:
        if r < edge_threshold or r >= h - edge_threshold or c < edge_threshold or c >= w - edge_threshold:
            edge_bonus = 15
            break

    # TODO 3.4: Calculate intensity bonus from average peak magnitude
    # Formula: min(average_magnitude * 20, 15)
    intensity_bonus = 0
    if num_peaks > 0:
        # Replace with your code
        avg_magnitude = np.mean([p[2] for p in peaks])
        intensity_bonus = min(avg_magnitude * 20, 15)

    # TODO 3.5: Calculate total score (clamped to 0-100)
    total_score = max(0, min(100, base_score + symmetry_bonus + edge_bonus + intensity_bonus))

    # TODO 3.6: Determine suspicion level based on score
    # 0-25: CLEAN, 26-50: LOW, 51-75: MEDIUM, 76-100: HIGH
    if total_score <= 25:
        level = 'CLEAN'
    elif total_score <= 50:
        level = 'LOW'
    elif total_score <= 75:
        level = 'MEDIUM'
    else:
        level = 'HIGH'

    # TODO 3.7: Print the required output format
    # [SCORE] Suspicion Score: XX/100 (LEVEL)
    print(f"[SCORE] Suspicion Score: {total_score}/100 ({level})")

    result = {
        'score': total_score,
        'level': level,
        'base_score': base_score,
        'symmetry_bonus': symmetry_bonus,
        'edge_bonus': edge_bonus,
        'intensity_bonus': intensity_bonus
    }

    return result


# =============================================================================
# TODO 4: MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_image_forensics(image_input) -> Dict:
    """
    Perform complete forensic analysis on an image.

    This function orchestrates the full analysis pipeline:
    1. Load/process the image
    2. Compute magnitude spectrum
    3. Detect frequency peaks
    4. Calculate suspicion score
    5. Compile and return results

    Args:
        image_input: Either a file path (str) or numpy array

    Returns:
        analysis: Dictionary with complete analysis results
            - 'image_shape': (H, W)
            - 'spectrum': numpy array
            - 'peaks': list of peaks
            - 'num_peaks': int
            - 'suspicion_score': int
            - 'suspicion_level': str
            - 'verdict': str (summary message)

    Output format (print this):
        [ANALYSIS] Image: <filename or 'array'>
        [ANALYSIS] Dimensions: WxH
        [ANALYSIS] Peaks detected: N
        [ANALYSIS] Final Verdict: LEVEL (Score: XX/100)
    """
    # TODO 4.1: Load or process the input image
    if isinstance(image_input, str):
        image = load_image(image_input)
        image_name = image_input.split('/')[-1] if '/' in image_input else image_input
    else:
        image = image_input
        image_name = 'array'

    # Ensure image is 2D grayscale
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    # TODO 4.2: Compute the magnitude spectrum
    # Hint: Call compute_magnitude_spectrum()
    spectrum = compute_magnitude_spectrum(image)  # Replace with your code

    # TODO 4.3: Detect frequency peaks
    # Hint: Call detect_frequency_peaks()
    peaks = detect_frequency_peaks(spectrum)  # Replace with your code

    # TODO 4.4: Calculate suspicion score
    # Hint: Call calculate_suspicion_score()
    score_result = calculate_suspicion_score(peaks, spectrum.shape)  # Replace with your code

    # TODO 4.5: Create verdict message
    verdict = f"Image analysis suggests {score_result['level']} suspicion level due to detected frequency anomalies."  # Replace with your code

    # TODO 4.6: Print the required output format
    # [ANALYSIS] Image: <name>
    # [ANALYSIS] Dimensions: WxH
    # [ANALYSIS] Peaks detected: N
    # [ANALYSIS] Final Verdict: LEVEL (Score: XX/100)
    H, W = image.shape
    num_peaks_detected = len(peaks) if peaks else 0
    print(f"[ANALYSIS] Image: {image_name}")
    print(f"[ANALYSIS] Dimensions: {W}x{H}")
    print(f"[ANALYSIS] Peaks detected: {num_peaks_detected}")
    print(f"[ANALYSIS] Final Verdict: {score_result['level']} (Score: {score_result['score']}/100)")

    # Compile results
    analysis = {
        'image_shape': image.shape,
        'spectrum': spectrum,
        'peaks': peaks,
        'num_peaks': len(peaks) if peaks else 0,
        'suspicion_score': score_result['score'] if score_result else 0,
        'suspicion_level': score_result['level'] if score_result else 'UNKNOWN',
        'verdict': verdict
    }

    return analysis


# =============================================================================
# TEST SUITE (DO NOT MODIFY)
# =============================================================================

def run_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("         FREQUENCY FORENSICS - TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    total_score = 0
    
    # Test 1: Spectrum computation
    print("\n[TEST 1] Spectrum computation", end="")
    try:
        test_img = create_test_image_clean(64)
        spectrum = compute_magnitude_spectrum(test_img)
        
        assert spectrum is not None, "Spectrum is None"
        assert spectrum.shape == test_img.shape, f"Shape mismatch: {spectrum.shape} vs {test_img.shape}"
        assert spectrum.min() >= 0, f"Min below 0: {spectrum.min()}"
        assert spectrum.max() <= 1, f"Max above 1: {spectrum.max()}"
        assert np.isfinite(spectrum).all(), "Contains NaN or Inf"
        
        print("." * 20 + " PASSED")
        total_score += 25
    except Exception as e:
        print("." * 20 + f" FAILED: {e}")
        all_passed = False
    
    # Test 2: Peak detection
    print("[TEST 2] Peak detection", end="")
    try:
        test_img = create_test_image_tampered(64)
        spectrum = compute_magnitude_spectrum(test_img)
        peaks = detect_frequency_peaks(spectrum, 99.0)
        
        assert peaks is not None, "Peaks is None"
        assert isinstance(peaks, list), "Peaks is not a list"
        assert len(peaks) > 0, "No peaks detected in tampered image"
        assert all(len(p) == 3 for p in peaks), "Peak tuples should have 3 elements"
        assert all(isinstance(p[0], (int, np.integer)) for p in peaks), "Row should be int"
        assert all(isinstance(p[1], (int, np.integer)) for p in peaks), "Col should be int"
        # Check sorting (descending by magnitude)
        if len(peaks) > 1:
            assert peaks[0][2] >= peaks[1][2], "Peaks not sorted by magnitude"
        
        print("." * 24 + " PASSED")
        total_score += 25
    except Exception as e:
        print("." * 24 + f" FAILED: {e}")
        all_passed = False
    
    # Test 3: Suspicion scoring
    print("[TEST 3] Suspicion scoring", end="")
    try:
        # Test with known peaks
        test_peaks = [(10, 10, 0.9), (54, 54, 0.85), (10, 54, 0.8)]
        result = calculate_suspicion_score(test_peaks, (64, 64))
        
        assert result is not None, "Result is None"
        assert 'score' in result, "Missing 'score' key"
        assert 'level' in result, "Missing 'level' key"
        assert 0 <= result['score'] <= 100, f"Score out of range: {result['score']}"
        assert result['level'] in ['CLEAN', 'LOW', 'MEDIUM', 'HIGH'], f"Invalid level: {result['level']}"
        
        # Test empty peaks
        empty_result = calculate_suspicion_score([], (64, 64))
        assert empty_result['score'] == 0, "Empty peaks should give score 0"
        assert empty_result['level'] == 'CLEAN', "Empty peaks should be CLEAN"
        
        print("." * 22 + " PASSED")
        total_score += 25
    except Exception as e:
        print("." * 22 + f" FAILED: {e}")
        all_passed = False
    
    # Test 4: Full pipeline
    print("[TEST 4] Full pipeline", end="")
    try:
        # Test with clean image
        clean_img = create_test_image_clean(64)
        clean_result = analyze_image_forensics(clean_img)
        
        assert clean_result is not None, "Result is None"
        assert 'suspicion_score' in clean_result, "Missing suspicion_score"
        assert 'suspicion_level' in clean_result, "Missing suspicion_level"
        
        # Test with tampered image
        tampered_img = create_test_image_tampered(64)
        tampered_result = analyze_image_forensics(tampered_img)
        
        # Tampered image should have higher suspicion score
        assert tampered_result['suspicion_score'] >= clean_result['suspicion_score'], \
            "Tampered image should have higher or equal suspicion score"
        
        print("." * 24 + " PASSED")
        total_score += 25
    except Exception as e:
        print("." * 24 + f" FAILED: {e}")
        all_passed = False
    
    # Final summary
    print("=" * 60)
    if all_passed:
        print(f"         ALL TESTS PASSED! Score: {total_score}/100")
    else:
        print(f"         SOME TESTS FAILED. Score: {total_score}/100")
    print("=" * 60)
    
    return total_score


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("    FREQUENCY FORENSICS - Image Tampering Detection")
    print("=" * 60 + "\n")
    
    # Run tests
    score = run_tests()
    
    # Demo with sample images if all tests pass
    if score == 100:
        print("\n" + "-" * 60)
        print("    DEMO: Analyzing Sample Images")
        print("-" * 60 + "\n")
        
        print(">>> Analyzing CLEAN image:")
        clean_result = analyze_image_forensics(create_test_image_clean(128))
        
        print("\n>>> Analyzing TAMPERED image:")
        tampered_result = analyze_image_forensics(create_test_image_tampered(128))
