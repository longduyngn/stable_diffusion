import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch

# Ensure the root directory of the pipeline script is in path for absolute imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from config.config import MODEL_ID, PROMPT, NEGATIVE_PROMPT, NUM_INFERENCE_STEPS, GUIDANCE_SCALES, SEED, DEVICE
from models.sd_pipeline import ManualStableDiffusion
from utility.logger import setup_logger, set_seed
from utility.visualize import plot_cfg_grid

def main():
    logger = setup_logger()
    logger.info("Initializing Custom OOP Stable Diffusion Inference Pipeline...")
    logger.info(f"Target Device: {DEVICE}")
    
    # 1. Setup global seed for reproducibility
    set_seed(SEED)
    
    # 2. Instantiate custom SD Pipeline
    pipeline = ManualStableDiffusion(model_id=MODEL_ID, device=DEVICE)
    
    # 3. Create torch generator for device
    generator = torch.Generator(device=DEVICE)
    
    images = []
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Prompt: '{PROMPT}'")
    logger.info(f"Negative Prompt: '{NEGATIVE_PROMPT}'")
    logger.info(f"Inference Steps: {NUM_INFERENCE_STEPS}")
    logger.info(f"CFG Guidance Scales: {GUIDANCE_SCALES}")
    
    # 4. Generate images for each CFG scale
    for scale in GUIDANCE_SCALES:
        logger.info(f"Generating image with CFG scale: {scale}...")
        
        # CRITICAL: Reset generator seed before each generation to ensure identical starting noise
        generator.manual_seed(SEED)
        
        img = pipeline.generate(
            prompt=PROMPT,
            guidance_scale=scale,
            generator=generator,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS
        )
        
        # Save individual image
        img_filename = f"astronaut_cfg_{scale}.png"
        img_path = os.path.join(output_dir, img_filename)
        img.save(img_path)
        logger.info(f"Saved individual image to {img_path}")
        images.append(img)
        
    # 5. Create comparison grid
    grid_path = os.path.join(output_dir, "cfg_comparison.png")
    logger.info(f"Generating 2x3 scale comparison grid at {grid_path}...")
    plot_cfg_grid(images, GUIDANCE_SCALES, grid_path)
    logger.info("Inference and visualization grid creation completed successfully!")

if __name__ == "__main__":
    main()
