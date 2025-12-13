#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilateral Filter Implementation - Classroom Exercise
Image Processing Course - Spatial Operations
Basic edge-preserving smoothing filter

"""

import numpy as np
import time
from typing import Tuple

from scipy.cluster.hierarchy import weighted
from torch.utils.hipify.hipify_python import value


def gaussian_weight(distance: float, sigma: float) -> float:
    """
    Calculate Gaussian weight for given distance
    
    Parameters:
    -----------
    distance : float
        Distance value (spatial or intensity)
    sigma : float
        Standard deviation
    
    Returns:
    --------
    float
        Gaussian weight
    """
    # Hint: G(x) = exp(-x^2 / (2 * sigma^2))
    return np.exp((-distance**2)/ (2*sigma**2))

def bilateral_filter_basic(image: np.ndarray,
                          window_size: int = 5,
                          sigma_spatial: float = 3.0,
                          sigma_range: float = 50.0) -> np.ndarray:
    """
    Basic implementation of bilateral filter
    
    The bilateral filter is an edge-preserving smoothing filter that
    combines spatial and intensity-based weighting.
    
    Parameters:
    -----------
    image : np.ndarray
        Input grayscale image (0-255)
    window_size : int
        Filter window size (must be odd, e.g., 3, 5, 7)
    sigma_spatial : float
        Spatial standard deviation (controls spatial extent)
    sigma_range : float
        Range/intensity standard deviation (controls edge preservation)
    
    Returns:
    --------
    np.ndarray
        Filtered image with same size and type as input
    
    Raises:
    -------
    AssertionError
        If parameters are invalid
    """
    
    # Input validation
    assert window_size % 2 == 1, "Window size must be odd"
    assert window_size >= 3, "Minimum window size is 3"
    assert sigma_spatial > 0, "Spatial sigma must be positive"
    assert sigma_range > 0, "Range sigma must be positive"
    assert len(image.shape) == 2, "Input must be grayscale image"
    
    # Convert to float64 to prevent overflow
    img = image.astype(np.float64)
    output = np.zeros_like(img)
    
    # Image dimensions
    height, width = img.shape
    half_window = window_size // 2
    
    # TODO: Implement bilateral filter

    # Steps:
    # 1. For each pixel (i,j) in the image:
    #    a. Define the neighborhood window
    #    b. For each neighbor pixel (k,l) in the window:
    #       - Calculate spatial distance
    #       - Calculate intensity difference
    #       - Compute spatial weight using Gaussian
    #       - Compute range weight using Gaussian
    #       - Total weight = spatial_weight * range_weight
    #    c. Normalize weights and compute filtered value
    
    for i in range(height):
        for j in range(width):
            # TODO: Complete implementation
            # Initialize weighted sum and weight sum
            weighted_sum = 0.0
            weight_sum = 0.0
            
            # Get center pixel value
            center_value = img[i, j]
            
            # Process neighborhood
            # TODO: Add your code here
            for k in range(-half_window, half_window+1):
                for l in range(-half_window, half_window+1):
                    neighborX = i + k
                    neighborY = j + l

                    if 0 <= neighborX < height and 0 <= neighborY < width:

                        distance = np.sqrt(k**2 + l**2)

                        neighbor_value = img[neighborX, neighborY]
                        value_distance = center_value - neighbor_value

                        distance_weight = gaussian_weight(distance, sigma_spatial)
                        color_weight = gaussian_weight(value_distance, sigma_range)

                        total_weight = distance_weight * color_weight

                        weighted_sum += neighbor_value * total_weight
                        weight_sum += total_weight

            
            # Store filtered value
            output[i, j] = weighted_sum / weight_sum if weight_sum > 0 else center_value
    
    return np.clip(output, 0, 255).astype(np.uint8)

def load_image(filename: str) -> np.ndarray:
    """
    Load image from file (numpy array or simple formats)
    
    Parameters:
    -----------
    filename : str
        Path to image file
        
    Returns:
    --------
    np.ndarray
        Loaded image
    """
    if filename.endswith('.npy'):
        return np.load(filename)
    else:
        # For other formats, try to load as numpy array
        try:
            return np.loadtxt(filename, dtype=np.uint8)
        except:
            # Create a dummy image if loading fails
            print(f"Warning: Could not load {filename}, using dummy image")
            return np.random.randint(0, 256, (100, 100), dtype=np.uint8)

def save_image(image: np.ndarray, filename: str):
    """
    Save image to file
    
    Parameters:
    -----------
    image : np.ndarray
        Image to save
    filename : str
        Output filename
    """
    if filename.endswith('.npy'):
        np.save(filename, image)
    else:
        np.savetxt(filename, image, fmt='%d')

def test_basic_implementation():
    """Test function for basic implementation"""
    print("Testing basic bilateral filter implementation...")
    
    # Create test image with noise
    test_img = np.ones((100, 100), dtype=np.uint8) * 128
    test_img[40:60, 40:60] = 200  # Add bright square
    noise = np.random.normal(0, 20, test_img.shape)
    noisy_img = np.clip(test_img + noise, 0, 255).astype(np.uint8)
    
    # Apply filter
    start_time = time.time()
    filtered = bilateral_filter_basic(noisy_img)
    elapsed = time.time() - start_time
    
    print(f"Filtering completed in {elapsed:.3f} seconds")
    print(f"Input range: [{noisy_img.min()}, {noisy_img.max()}]")
    print(f"Output range: [{filtered.min()}, {filtered.max()}]")
    
    # Basic validation
    assert filtered.shape == noisy_img.shape, "Output shape mismatch"
    assert filtered.dtype == np.uint8, "Output type should be uint8"
    print("Basic tests passed!")

if __name__ == "__main__":
    test_basic_implementation()