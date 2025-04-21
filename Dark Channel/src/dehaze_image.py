import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from test_dcp_numpy import (get_dark_channel, estimate_atmospheric_light, 
                           estimate_transmission, guided_filter_simple, dehaze)

def create_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def process_image(input_path, output_dir='results', show_results=False):
    """
    Process a single image with dark channel prior dehazing
    
    Args:
        input_path: Path to the input hazy image
        output_dir: Directory to save the results
        show_results: Whether to display the results
    """
    # Create output directory
    create_dir(output_dir)
    
    # Get base filename without extension
    base_filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(base_filename)[0]
    
    # Load the hazy image
    try:
        img = Image.open(input_path).convert('RGB')
        img_np = np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        print(f"Error loading image {input_path}: {e}")
        return
    
    print(f"Processing image: {input_path}")
    print(f"Image size: {img.size[0]}x{img.size[1]}")
    
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
    
    # Save individual images
    Image.fromarray(img_8bit).save(os.path.join(output_dir, f"{name_without_ext}_original.png"))
    Image.fromarray(dark_channel_8bit).save(os.path.join(output_dir, f"{name_without_ext}_dark_channel.png"))
    Image.fromarray(transmission_8bit).save(os.path.join(output_dir, f"{name_without_ext}_transmission.png"))
    Image.fromarray(refined_trans_8bit).save(os.path.join(output_dir, f"{name_without_ext}_refined_transmission.png"))
    Image.fromarray(dehazed_8bit).save(os.path.join(output_dir, f"{name_without_ext}_dehazed.png"))
    
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
    plt.savefig(os.path.join(output_dir, f"{name_without_ext}_result.png"))
    
    if show_results:
        plt.show()
    else:
        plt.close()
    
    print(f"Dehazing completed for {input_path}")
    print(f"Results saved in {output_dir}/")
    print(f"Dehazed image: {output_dir}/{name_without_ext}_dehazed.png")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image Dehazing using Dark Channel Prior')
    parser.add_argument('--input', type=str, required=True, help='Path to the input hazy image')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save the results')
    parser.add_argument('--show', action='store_true', help='Show the results')
    args = parser.parse_args()
    
    # Process the image
    process_image(args.input, args.output_dir, args.show)

if __name__ == '__main__':
    main()