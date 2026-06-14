import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

# Ensure root directory is in path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from config.config import MODEL_ID, PROMPT, NEGATIVE_PROMPT, NUM_INFERENCE_STEPS, SEED, DEVICE
from utility.logger import setup_logger, set_seed

def main():
    logger = setup_logger()
    logger.info("Initializing baseline comparison using official StableDiffusionPipeline...")
    
    # 1. Setup seed
    set_seed(SEED)
    
    # 2. Load official StableDiffusionPipeline
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    logger.info(f"Loading StableDiffusionPipeline with dtype={dtype}...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype
    )
    
    # Use DDIMScheduler to match the custom manual implementation
    logger.info("Configuring pipeline to use DDIMScheduler...")
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(DEVICE)
    pipeline.set_progress_bar_config(disable=True)
    
    # Set to eval and disable gradients
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.unet.eval()
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create generator
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(SEED)
    
    # Generate baseline image at scale 7.5
    scale = 7.5
    logger.info(f"Generating baseline image with CFG scale: {scale}...")
    
    with torch.no_grad():
        image = pipeline(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=scale,
            generator=generator
        ).images[0]
        
    baseline_path = os.path.join(output_dir, f"baseline_cfg_{scale}.png")
    image.save(baseline_path)
    logger.info(f"Saved baseline image to {baseline_path}")
    logger.info("Baseline comparison image generated. Compare it with output/astronaut_cfg_7.5.png.")

if __name__ == "__main__":
    main()
