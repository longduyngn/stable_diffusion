import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

class ManualStableDiffusion:
    def __init__(self, model_id: str, device: str = "cuda"):
        """
        Initializes the Stable Diffusion components manually.
        Casts models to float16 (if on GPU) and disables gradient computation.
        """
        self.device = device
        # Use float16 on CUDA to save memory (prevent OOM)
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load components manually
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Move models to device and cast to appropriate dtype
        self.text_encoder.to(self.device, dtype=self.dtype)
        self.vae.to(self.device, dtype=self.dtype)
        self.unet.to(self.device, dtype=self.dtype)
        
        # Set to eval mode
        self.text_encoder.eval()
        self.vae.eval()
        self.unet.eval()
        
        # Explicitly disable gradients to prevent memory overhead
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

    def _get_text_embeddings(self, prompt: str, negative_prompt: str):
        """
        Encodes positive and negative prompts using the CLIP text encoder
        and concatenates them into a single batch of shape [2, seq_len, hidden_dim].
        """
        # Tokenize and encode positive prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        
        # Tokenize and encode negative prompt
        uncond_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        uncond_input_ids = uncond_inputs.input_ids.to(self.device)
        negative_prompt_embeds = self.text_encoder(uncond_input_ids)[0]
        
        # Concatenate negative and positive prompt embeddings
        # [negative_prompt_embeds, prompt_embeds] -> shape: [2, seq_len, hidden_dim]
        # (unconditional is placed first so we can use chunk(2) on UNet output)
        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        return text_embeddings

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        guidance_scale: float,
        generator: torch.Generator,
        negative_prompt: str = "",
        num_inference_steps: int = 50
    ) -> Image.Image:
        """
        Generates an image for the prompt using manual DDIM inference loop.
        """
        # 1. Initialize random latent noise [1, 4, 64, 64]
        latents = torch.randn(
            (1, self.unet.config.in_channels, 64, 64),
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        
        # 2. Scale the initial noise using scheduler.init_noise_sigma
        latents = latents * self.scheduler.init_noise_sigma
        
        # Get text embeddings
        text_embeddings = self._get_text_embeddings(prompt, negative_prompt)
        
        # 3. Setup DDIM Scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # 4. Denoising Loop
        for t in self.scheduler.timesteps:
            # Duplicate latents to match batch size of text_embeddings [2, 4, 64, 64]
            latent_model_input = torch.cat([latents] * 2)
            # Scale model input based on the scheduler's requirements
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual with UNet
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Split predicted noise into unconditional (uncond) and conditional (text) predictions
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            
            # Apply Classifier-Free Guidance (CFG) formula
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute the previous noisy sample (x_t -> x_t-1)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        # 5. Decode final latent using VAE (scaled by scaling factor 1 / 0.18215)
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        
        # Postprocess: Denormalize from [-1, 1] to [0, 255] and cast to numpy
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image[0])
        return pil_image
