#!/usr/bin/env python3

import numpy as np

def contrast_stretch(image, low_percentile=2, high_percentile=98):
    """
    Apply contrast stretching to enhance image contrast.
    
    Parameters:
    -----------
    image : numpy.ndarray
        2D grayscale image (uint8)
    low_percentile : float
        Lower percentile for stretching (0-100)
    high_percentile : float
        Upper percentile for stretching (0-100)
  
    """

    if image.size == 0:
        return {
            'output': image.copy(),
            'low_value': 0,
            'high_value': 0,
            'pixels_clipped': 0,
            'histogram_spread': 0
        }

    if low_percentile > high_percentile:
        low_percentile, high_percentile = high_percentile, low_percentile

    L = np.percentile(image, low_percentile)
    H = np.percentile(image, high_percentile)
    L = int(np.round(L))
    H = int(np.round(H))

    pixels_clipped = int(np.sum((image < L) | (image > H)))

    stretched = ((image.astype(float) - L) / (H - L) * 255).clip(0, 255)
    output = stretched.astype(np.uint8)
    histogram_spread = int(np.max(output) - np.min(output))

    if L == H:
        output = np.full_like(image, 128, dtype=np.uint8)
        histogram_spread = 0

    # Placeholder return - REPLACE WITH YOUR IMPLEMENTATION
    result = {
        'output': output,
        'low_value': L,
        'high_value': H,
        'pixels_clipped': pixels_clipped,
        'histogram_spread': histogram_spread
    }
    
    return result
    
    # YOUR CODE ENDS HERE
    # ===================


# Test code (will be removed during evaluation)
if __name__ == "__main__":
    # Simple test
    test_img = np.array([[50, 100, 150],
                          [75, 125, 175],
                          [25, 200, 225]], dtype=np.uint8)
    
    result = contrast_stretch(test_img, 10, 90)
    
    print("Keys in result:", list(result.keys()))
    print("Output shape:", result['output'].shape)
    print("Output dtype:", result['output'].dtype)
    print("Low value:", result['low_value'])
    print("High value:", result['high_value'])
    print("Pixels clipped:", result['pixels_clipped'])
    print("Histogram spread:", result['histogram_spread'])
