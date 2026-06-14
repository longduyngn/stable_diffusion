import torch

# Model configuration
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Prompt configuration
PROMPT = "a mermaid with red hair, lying on a clif, singing a sad song in the ocean, crying, tears, on the otherside, a pirate ship is sailing toward ,highly detailed"
NEGATIVE_PROMPT = ""

# Inference configuration
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALES = [1.0, 3.0, 5.0, 7.5, 12.0, 20.0]
SEED =50

# Hardware and dimensions config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_CHANNELS = 4
LATENT_HEIGHT = 64
LATENT_WIDTH = 64
