# Stable Diffusion Pipeline from Components

Rebuilds the **Stable Diffusion v1.5** sampling loop by hand from its components — CLIP tokenizer and text encoder,
VAE, U-Net and DDIM scheduler — instead of calling `StableDiffusionPipeline`.

## What it shows

- Text conditioning with CLIP; unconditional and conditional embeddings batched together
- Manual **DDIM** denoising loop with **classifier-free guidance**
  `ε = ε_uncond + s · (ε_text − ε_uncond)`
- VAE decoding of latents (scaling factor 0.18215) to an image
- FP16 on GPU, gradients disabled, models in eval mode
- A **CFG sweep** (scales 1 → 20) with a fixed seed, saved as a comparison grid
- `compare_pipeline.py`: the same prompt through the official Diffusers pipeline as a baseline

## Usage

```bash
pip install torch diffusers transformers matplotlib pillow
python infer.py              # output/*.png and output/cfg_comparison.png
python compare_pipeline.py   # baseline from StableDiffusionPipeline
```

Prompt, steps, guidance scales and seed are set in `config/config.py`.
