import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter

def create_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def get_dark_channel(img, patch_size=15):
    """
    Calculate dark channel of an image
    
    Args:
        img: Input image of shape (H, W, 3)
        patch_size: Size of the local patch
    
    Returns:
        dark_channel: Dark channel of the image
    """
    # Minimum across channels
    min_channel = np.min(img, axis=2)
    
    # Local minimum filter
    padded = np.pad(min_channel, patch_size//2, mode='edge')
    dark_channel = minimum_filter(padded, size=patch_size)
    
    # Remove padding
    dark_channel = dark_channel[patch_size//2:-(patch_size//2), patch_size//2:-(patch_size//2)]
    
    return dark_channel

def estimate_atmospheric_light(img, dark_channel, top_percent=0.001):
    """
    Estimate atmospheric light from the dark channel
    
    Args:
        img: Input image of shape (H, W, 3)
        dark_channel: Dark channel of the image
        top_percent: Percentage of pixels used to estimate atmospheric light
    
    Returns:
        A: Atmospheric light (3,)
    """
    # Flatten dark channel and find the brightest pixels
    flat_dark = dark_channel.flatten()
    num_pixels = len(flat_dark)
    top_indices = np.argsort(flat_dark)[-int(num_pixels * top_percent):]
    
    # Get corresponding coordinates in the original image
    rows, cols = np.unravel_index(top_indices, dark_channel.shape)
    
    # Use the brightest pixels in the original image
    A = np.zeros(3, dtype=np.float32)
    for i in range(3):
        A[i] = np.max(img[rows, cols, i])
    
    return A

def estimate_transmission(img, A, dark_channel, omega=0.95):
    """
    Estimate transmission map
    
    Args:
        img: Input image of shape (H, W, 3)
        A: Atmospheric light (3,)
        dark_channel: Dark channel of the image
        omega: Dehazing strength (0-1)
    
    Returns:
        t: Transmission map
    """
    # Normalize by atmospheric light
    normalized = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        normalized[:, :, i] = img[:, :, i] / max(A[i], 0.01)
    
    # Get dark channel of normalized image
    normalized_dark = get_dark_channel(normalized)
    
    # Estimate transmission
    t = 1 - omega * normalized_dark
    
    return t

def guided_filter_simple(guide, target, radius=60, eps=0.01):
    """
    Simple guided filter implementation
    
    Args:
        guide: Guidance image
        target: Target image to be filtered
        radius: Filter radius
        eps: Regularization parameter
    
    Returns:
        filtered: Filtered image
    """
    # Convert to grayscale if guide is color
    if len(guide.shape) == 3:
        guide = np.mean(guide, axis=2)
    
    # Mean filter for approximation
    from scipy.ndimage import uniform_filter
    mean_guide = uniform_filter(guide, size=radius)
    mean_target = uniform_filter(target, size=radius)
    corr = uniform_filter(guide * target, size=radius)
    var = uniform_filter(guide * guide, size=radius) - mean_guide * mean_guide
    
    # Linear coefficients
    a = (corr - mean_guide * mean_target) / (var + eps)
    b = mean_target - a * mean_guide
    
    # Apply coefficients
    mean_a = uniform_filter(a, size=radius)
    mean_b = uniform_filter(b, size=radius)
    
    return mean_a * guide + mean_b

def dehaze(img, transmission, A, t_min=0.47):
    """
    Dehaze an image using the atmospheric scattering model
    
    Args:
        img: Input image of shape (H, W, 3)
        transmission: Transmission map
        A: Atmospheric light (3,)
        t_min: Minimum transmission value
    
    Returns:
        dehazed: Dehazed image
    """
    # Limit the transmission
    t = np.maximum(transmission, t_min)
    
    # Expand t to match img shape
    t_expanded = np.expand_dims(t, axis=2).repeat(3, axis=2)
    
    # Apply atmospheric scattering model
    dehazed = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        dehazed[:, :, i] = (img[:, :, i] - A[i]) / t_expanded[:, :, i] + A[i]
    
    # Clip values to [0, 1]
    dehazed = np.clip(dehazed, 0, 1)
    
    return dehazed

def main():
    # Create results directory
    create_dir('results')
    
    # Load the hazy image
    image_path = 'data/test_images/hazy_image.jpg'
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    # Get dark channel
    dark_channel = get_dark_channel(img_np)
    
    # Estimate atmospheric light
    A = estimate_atmospheric_light(img_np, dark_channel)
    print(f"Estimated atmospheric light: {A}")
    
    # Estimate transmission map
    transmission = estimate_transmission(img_np, A, dark_channel)
    
    # Refine transmission using guided filter
    refined_trans = guided_filter_simple(img_np[:, :, 0], transmission)
    
    # Dehaze the image
    dehazed = dehaze(img_np, refined_trans, A)
    
    # Convert to 8-bit for display and saving
    img_8bit = (img_np * 255).astype(np.uint8)
    dark_channel_8bit = (dark_channel * 255).astype(np.uint8)
    transmission_8bit = (transmission * 255).astype(np.uint8)
    refined_trans_8bit = (refined_trans * 255).astype(np.uint8)
    dehazed_8bit = (dehazed * 255).astype(np.uint8)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.title('Original Hazy Image')
    plt.imshow(img_8bit)
    plt.axis('off')
    
    plt.subplot(232)
    plt.title('Dark Channel')
    plt.imshow(dark_channel_8bit, cmap='gray')
    plt.axis('off')
    
    plt.subplot(233)
    plt.title('Transmission Map')
    plt.imshow(transmission_8bit, cmap='gray')
    plt.axis('off')
    
    plt.subplot(234)
    plt.title('Refined Transmission')
    plt.imshow(refined_trans_8bit, cmap='gray')
    plt.axis('off')
    
    plt.subplot(235)
    plt.title('Dehazed Image')
    plt.imshow(dehazed_8bit)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/dcp_numpy_result.png')
    
    # Save individual images
    Image.fromarray(img_8bit).save('results/original.png')
    Image.fromarray(dark_channel_8bit).save('results/dark_channel.png')
    Image.fromarray(transmission_8bit).save('results/transmission.png')
    Image.fromarray(refined_trans_8bit).save('results/refined_transmission.png')
    Image.fromarray(dehazed_8bit).save('results/dehazed.png')
    
    print("Dehazing completed. Results saved in the 'results' directory.")

if __name__ == '_main_':
    main()