# Adrishyam

A Python package for image dehazing using Dark Channel Prior algorithm.

## Installation

```bash
pip install adrishyam
```

## Usage

```python
from adrishyam import dehaze_image

# Basic usage
dehaze_image(
    input_path="path/to/hazy/image.jpg",
    output_dir="path/to/output/directory"
)

# Advanced usage with custom parameters
dehaze_image(
    input_path="path/to/hazy/image.jpg",
    output_dir="path/to/output/directory",
    t_min=0.1,  # Minimum transmission value (default: 0.1)
    patch_size=15,  # Size of the local patch (default: 15)
    omega=0.95,  # Dehazing strength (default: 0.95)
    radius=60,  # Filter radius for guided filter (default: 60)
    eps=0.01,  # Regularization parameter (default: 0.01)
    show_results=False  # Whether to display results (default: False)
)
```

## Output

The package will create the following files in the output directory:
- `original.png`: Original hazy image
- `dark_channel.png`: Dark channel of the image
- `transmission.png`: Estimated transmission map
- `refined_transmission.png`: Refined transmission map
- `dehazed.png`: Final dehazed image
- `result.png`: Combined visualization of all steps

## License

MIT License #
