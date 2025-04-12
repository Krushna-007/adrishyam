# Adrishyam 🖼️  
_A Python package for image dehazing using the Dark Channel Prior algorithm._

## 💡 Key Features  
- Implements the Dark Channel Prior algorithm for effective image dehazing.  
- Supports configurable parameters for advanced users.   
- Outputs intermediate steps for better visualization of the dehazing process.  

---

## 📦 Installation  

Install Adrishyam via `pip`:  

```bash
pip install adrishyam
```

---

## 🚀 Usage  

### 🔧 Basic Usage  
Dehaze an image by providing input and output paths:  

```python
from adrishyam import dehaze_image

dehaze_image(
    input_path="path/to/hazy/image.jpg",
    output_dir="path/to/output/directory"
)
```

### ⚙️ Advanced Usage  
Customize dehazing parameters for fine-tuned results:  

```python
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

---

## 📂 Output  

Adrishyam generates step-by-step outputs in your specified `output_dir`:  
- `original.png` ➡️ Original hazy image.  
- `dark_channel.png` ➡️ Dark channel visualization.  
- `transmission.png` ➡️ Estimated transmission map.  
- `refined_transmission.png` ➡️ Refined transmission map.  
- `dehazed.png` ➡️ Final dehazed image.  
- `result.png` ➡️ Combined visualization of all processing steps.  

---

## 🔍 Results  
**Example Outputs from Adrishyam:**  

![Original Hazy Image](path/to/origina 

---

## 📜 License  

Adrishyam is licensed under the MIT License. 📝  
Feel free to use and contribute!