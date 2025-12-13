"""
Edge Detection Exercise - Student Code
Image Processing Course

TODO: Complete the functions below to implement edge detection
      using Laplacian operator and zero-crossing detection.

Student Name: Shlomo Edelstien
Student ID: 318751625
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy.vector import Laplacian


# ==============================================================================
# EXAMPLE 1: Understanding Second Derivative
# ==============================================================================

def example_second_derivative():
    """
    Simple example to understand how second derivative detects edges.
    
    This creates a 1D signal with a step edge and shows:
    - Original signal
    - First derivative (shows slope/gradient)
    - Second derivative (crosses zero at edge)
    """
    # Create 1D signal: step edge from 50 to 200
    x = np.arange(20)
    signal = np.array([50]*8 + [100, 150] + [200]*10)
    
    # Compute derivatives using np.diff
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x, signal, 'b-o', linewidth=2)
    plt.title('Original Signal')
    plt.xlabel('Position')
    plt.ylabel('Intensity')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(x[:-1], first_deriv, 'g-o', linewidth=2)
    plt.title('First Derivative')
    plt.xlabel('Position')
    plt.ylabel('Rate of Change')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(x[:-2], second_deriv, 'r-o', linewidth=2)
    plt.title('Second Derivative (Zero-Crossing)')
    plt.xlabel('Position')
    plt.ylabel('Curvature')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('second_derivative_example.png', dpi=100)
    print("✓ Saved: second_derivative_example.png")
    print("\nNotice:")
    print("- First derivative shows WHERE intensity changes")
    print("- Second derivative CROSSES ZERO at the edge location")
    print("- Zero-crossing = precise edge location!")


# ==============================================================================
# PART 1: LAPLACIAN OPERATOR IMPLEMENTATION
# ==============================================================================

def apply_laplacian(image):
    """
    Apply the Laplacian operator to detect edges in an image.
    
    The Laplacian is the sum of second derivatives in x and y directions:
        ∇²I = ∂²I/∂x² + ∂²I/∂y²
    
    The discrete Laplacian kernel is:
        [ 0  1  0]
        [ 1 -4  1]
        [ 0  1  0]
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input grayscale image (2D array)
    
    Returns:
    --------
    laplacian : numpy.ndarray
        Laplacian filtered image (dtype=float64, same size as input)
        Positive values indicate bright regions, negative indicate dark regions
    
    TODO: Implement this function following the steps below
    """
    
    # TODO 1: Get image dimensions
    # Hint: height, width = image.shape
    height, width = image.shape
    
    # TODO 2: Initialize output array
    # Hint: Use np.zeros with dtype=np.float64 to preserve positive/negative values
    laplacian = np.zeros((height, width), dtype=np.float64)
    
    # TODO 3: Define the Laplacian kernel
    # The kernel is: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
    # TODO 4: Add zero-padding to handle borders
    # Hint: Use np.pad(image, pad_width=1, mode='constant', constant_values=0)
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    
    # TODO 5: Apply convolution
    # Loop through each pixel and apply the kernel
    # For each pixel at (i, j):
    #   1. Extract 3x3 neighborhood from padded image
    #   2. Multiply neighborhood by kernel element-wise
    #   3. Sum all products to get the Laplacian value
    
    for i in range(height):
        for j in range(width):
            # TODO: Extract 3x3 neighborhood
            # Remember: padded indices are offset by 1
            neighborhood = padded_image[i:i+3, j:j+3]
            
            # TODO: Apply kernel (element-wise multiply and sum)
            laplacian[i, j] = np.sum(neighborhood * kernel)
    
    return laplacian


# ==============================================================================
# PART 2: ZERO-CROSSING DETECTION
# ==============================================================================

def find_zero_crossings(laplacian):
    """
    Detect zero-crossings in the Laplacian response to locate edges.
    
    A pixel is a zero-crossing (edge) if the Laplacian changes sign
    between the pixel and its neighbors.
    
    Detection rules:
    ----------------
    CASE 1: If center pixel == 0
            AND (left < 0 and right > 0) OR (top < 0 and bottom > 0)
            → This is an edge
    
    CASE 2: If center pixel < 0
            AND any neighbor (top/bottom/left/right) > 0
            → This is an edge
    
    CASE 3: If center pixel > 0
            AND any neighbor (top/bottom/left/right) < 0
            → This is an edge
    
    Parameters:
    -----------
    laplacian : numpy.ndarray
        Laplacian filtered image (2D array with positive/negative values)
    
    Returns:
    --------
    edges : numpy.ndarray
        Binary edge map where:
        - 255 = edge pixel (white)
        - 0 = non-edge pixel (black)
        dtype=uint8
    
    TODO: Implement this function following the steps below
    """
    
    # TODO 1: Get dimensions
    height, width = laplacian.shape
    
    # TODO 2: Initialize edge map
    # All pixels start as 0 (non-edge), we'll mark edges as 255
    edges = np.zeros((height, width), dtype=np.uint8)
    
    # TODO 3: Loop through interior pixels (skip borders to avoid index errors)
    # Start from 1 and end at height-1, width-1
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            
            # TODO 4: Get center pixel and its 4 neighbors
            center = laplacian[i,j]
            top = laplacian[i-1,j]
            bottom = laplacian[i+1,j]
            left = laplacian[i,j-1]
            right = laplacian[i,j+1]
            
            # TODO 5: Initialize flag for zero-crossing detection
            is_edge = False
            
            # TODO 6: Check CASE 1 - center is exactly zero
            if center == 0:
                if top > 0 > bottom or top < 0 < bottom or left > 0 > right or left < 0 < right:
                    is_edge = True
            
            # TODO 7: Check CASE 2 - center is negative
            elif center < 0:
                if top > 0 or bottom > 0 or left > 0 or right > 0:
                    is_edge = True

            
            # TODO 8: Check CASE 3 - center is positive
            elif center > 0:
                if top < 0 or bottom < 0 or left < 0 or right < 0:
                    is_edge = True
            
            # TODO 9: Mark edge pixels
            if is_edge:
                edges[i, j] = 255  # REPLACE 0 with correct value (255)
    
    return edges


# ==============================================================================
# PART 3: COMPLETE EDGE DETECTION PIPELINE
# ==============================================================================

def detect_edges(image):
    """
    Complete edge detection pipeline:
    1. Apply Laplacian operator
    2. Find zero-crossings
    
    TODO: Implement this function by calling the two functions above
    """
    
    # TODO: Apply Laplacian
    laplacian = apply_laplacian(image)  # REPLACE THIS LINE
    
    # TODO: Find zero-crossings
    edges = find_zero_crossings(laplacian)
    
    return edges, laplacian


# ==============================================================================
# TESTING AND VISUALIZATION
# ==============================================================================

def test_simple_edge():
    """
    Test your implementation on a simple step edge.
    """
    print("=" * 70)
    print("TESTING: Simple Step Edge")
    print("=" * 70)
    
    # Create a simple step edge image
    test_image = np.array([
        [50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 200, 200, 200, 200],
        [200, 200, 200, 200, 200, 200, 200],
        [200, 200, 200, 200, 200, 200, 200],
        [200, 200, 200, 200, 200, 200, 200]
    ], dtype=np.float64)
    
    print("\nInput image:")
    print(test_image.astype(int))
    
    # Apply Laplacian
    laplacian = apply_laplacian(test_image)
    print("\nLaplacian response:")
    print(laplacian.astype(int))
    
    # Find edges
    edges = find_zero_crossings(laplacian)
    print("\nDetected edges (1 = edge, 0 = non-edge):")
    print((edges // 255).astype(int))
    
    # Check if edges were detected
    if np.sum(edges) > 0:
        print("\n✓ SUCCESS: Edges detected!")
        print(f"  Found {np.sum(edges > 0)} edge pixels")
    else:
        print("\n✗ ISSUE: No edges detected. Check your implementation.")
    
    print("=" * 70)


def visualize_results(image_path=None):
    """
    Visualize edge detection on a real or synthetic image.
    
    TODO: This is optional - you can enhance this if you want!
    """
    if image_path is None:
        # Create a synthetic image with multiple features
        image = np.zeros((100, 100))
        # Add rectangle
        image[20:40, 20:40] = 200
        # Add circle
        y, x = np.ogrid[:100, :100]
        mask = (x - 70)**2 + (y - 70)**2 <= 15**2
        image[mask] = 150
    else:
        # Load image (you'd need to add image loading code)
        image = plt.imread(image_path)
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)  # Convert to grayscale
    
    # Apply edge detection
    edges, laplacian = detect_edges(image)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(laplacian, cmap='gray')
    axes[1].set_title('Laplacian Response')
    axes[1].axis('off')
    
    axes[2].imshow(edges, cmap='gray')
    axes[2].set_title('Detected Edges')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('edge_detection_result.png', dpi=100)
    print("\n✓ Saved: edge_detection_result.png")


# ==============================================================================
# MAIN - RUN YOUR CODE
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EDGE DETECTION EXERCISE")
    print("=" * 70)
    
    # Show second derivative example
    print("\n1. Understanding Second Derivative...")
    example_second_derivative()
    
    # Test simple edge
    print("\n2. Testing Your Implementation...")
    test_simple_edge()
    
    # Visualize on synthetic image
    print("\n3. Creating Visualization...")
    visualize_results()
    
    print("\n" + "=" * 70)
    print("DONE! Check the generated PNG images.")
    print("=" * 70 + "\n")
