import matplotlib.pyplot as plt
from typing import List
from PIL import Image

def plot_cfg_grid(images: List[Image.Image], scales: List[float], save_path: str):
    """
    Plots a 2x3 grid of images generated using different CFG scales
    and saves the figure to the specified path.
    """
    if len(images) != 6:
        raise ValueError(f"Expected exactly 6 images for the 2x3 grid comparison, got {len(images)}")
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (img, scale) in enumerate(zip(images, scales)):
        axes[idx].imshow(img)
        axes[idx].set_title(f"CFG Scale: {scale}", fontsize=14, fontweight="bold", pad=10)
        axes[idx].axis("off")
        
    plt.suptitle("Stable Diffusion Classifier-Free Guidance (CFG) Comparison", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
