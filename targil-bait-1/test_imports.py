"""
Quick test to check if imports work and get a sense of processing time
"""
import numpy as np
import sys
import os
from PIL import Image
import time

# Add paths
sys.path.append('D:/PycharmProjects/image-processing/targil-kita-4')
sys.path.append('D:/PycharmProjects/image-processing/targil-kita-3')

print("Testing imports...")
try:
    from student_solution import hough_line_transform, find_peaks
    print("[OK] Hough Transform imported successfully")
except Exception as e:
    print(f"[FAIL] Hough Transform import failed: {e}")
    sys.exit(1)

try:
    from edge_detection_student import detect_edges
    print("[OK] Edge detection imported successfully")
except Exception as e:
    print(f"[FAIL] Edge detection import failed: {e}")
    sys.exit(1)

print("\nTesting on small image...")
# Load and resize image_1
img_path = 'D:/PycharmProjects/image-processing/targil-bait-1/image_1.png'
img = Image.open(img_path)
print(f"Original image size: {img.size}")

# Resize to smaller for testing
small_img = img.resize((img.size[0] // 4, img.size[1] // 4), Image.Resampling.LANCZOS)
print(f"Resized to: {small_img.size}")

# Convert to grayscale
gray = small_img.convert('L')
gray_array = np.array(gray, dtype=np.uint8)

# Test edge detection
print("\nTesting edge detection...")
start = time.time()
edges, laplacian = detect_edges(gray_array)
print(f"Edge detection took {time.time() - start:.2f} seconds")
print(f"Number of edge pixels: {np.count_nonzero(edges)}")

# Test Hough Transform
print("\nTesting Hough Transform...")
start = time.time()
accumulator, rhos, thetas = hough_line_transform(edges)
print(f"Hough Transform took {time.time() - start:.2f} seconds")
print(f"Accumulator shape: {accumulator.shape}")
print(f"Max accumulator value: {np.max(accumulator)}")

# Find peaks
print("\nFinding peaks...")
start = time.time()
peaks = find_peaks(accumulator, rhos, thetas, threshold=None, num_peaks=10)
print(f"Peak finding took {time.time() - start:.2f} seconds")
print(f"Found {len(peaks)} peaks")

if len(peaks) > 0:
    print("\nTop 3 peaks:")
    for i, (rho, theta, votes) in enumerate(peaks[:3]):
        print(f"  {i+1}. rho={rho:.1f}, theta={theta:.1f}°, votes={votes}")

print("\n[OK] All tests passed!")
