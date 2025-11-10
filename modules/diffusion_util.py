"""
Diffusion model utilities for T2I generation and DDIM inversion.
Provides helper functions for Stable Diffusion operations.
"""
import torch
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from PIL import Image


def load_model(model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
    """
    Load pretrained Stable Diffusion pipeline.
    
    Args:
        model_id (str): HuggingFace model ID
        device (str): Device to load model on
    
    Returns:
        StableDiffusionPipeline: Loaded pipeline
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Use DDIM scheduler for better inversion
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    return pipe


def generate_image(pipe, prompt, watermarked_noise_tensor, num_inference_steps=50, guidance_scale=7.5):
    """
    Generate image from watermarked noise using Stable Diffusion.
    
    Args:
        pipe: Stable Diffusion pipeline
        prompt (str): Text prompt for generation
        watermarked_noise_tensor (torch.Tensor): Watermarked latent noise (B, 4, H, W)
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): CFG guidance scale
    
    Returns:
        PIL.Image: Generated image
    """
    # Ensure noise is on correct device
    latents = watermarked_noise_tensor.to(pipe.device)
    
    # Generate image using watermarked latents as starting point
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
    
    return image


def invert_image(pipe, image, prompt="", num_inference_steps=50, guidance_scale=1.0):
    """
    Invert image back to latent noise using DDIM inversion.
    
    Args:
        pipe: Stable Diffusion pipeline
        image (PIL.Image or torch.Tensor): Image to invert
        prompt (str): Text prompt (usually empty for inversion)
        num_inference_steps (int): Number of inversion steps
        guidance_scale (float): CFG guidance scale (usually 1.0 for inversion)
    
    Returns:
        torch.Tensor: Inverted latent noise (1, 4, H, W)
    """
    # Convert PIL Image to tensor if needed
    if isinstance(image, Image.Image):
        image = pipe.image_processor.preprocess(image)
        image = image.to(pipe.device)
    
    # Encode image to latent space
    with torch.no_grad():
        latents = pipe.vae.encode(image).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
    
    # Setup inverse scheduler
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    timesteps = inverse_scheduler.timesteps
    
    # DDIM inversion loop
    with torch.no_grad():
        # Get text embeddings
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(pipe.device))[0]
        
        # Unconditioned embeddings for classifier free guidance
        uncond_inputs = pipe.tokenizer(
            "",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids.to(pipe.device))[0]
        
        # Concatenate for CFG
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Reverse diffusion process
        for i, t in enumerate(timesteps):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            
            # Predict noise
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample
            latents = inverse_scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents
