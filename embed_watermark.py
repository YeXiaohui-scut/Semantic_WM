"""
MetaSeal-Noise Watermark Embedding
Embed watermark into T2I generation by signing prompt and embedding in initial noise.
"""
import os
import argparse
import torch
from PIL import Image

from modules.inn_model import INNModel
from modules.pattern_util import encode_to_qr
from modules.crypto_util import generate_keys, sign_message
from modules import diffusion_util


def embed_watermark(args):
    """Embed watermark into image generation."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate keys if they don't exist
    if not os.path.exists(args.sk_path) or not os.path.exists(args.pk_path):
        print("Generating new key pair...")
        generate_keys(args.sk_path, args.pk_path)
    
    # Load trained INN model
    print(f"Loading INN model from {args.model_path}")
    inn_model = INNModel(num_blocks=args.num_blocks, channels=4).to(device)
    inn_model.load_state_dict(torch.load(args.model_path, map_location=device))
    inn_model.eval()
    
    # Load Stable Diffusion pipeline
    print("Loading Stable Diffusion pipeline...")
    pipe = diffusion_util.load_model(args.diffusion_model, device=device)
    
    # Get prompt from user
    prompt = args.prompt
    print(f"Prompt: {prompt}")
    
    # Sign the prompt
    print("Signing prompt...")
    signature = sign_message(args.sk_path, prompt)
    print(f"Signature generated ({len(signature)} bytes)")
    
    # Encode to QR code
    print("Encoding to QR code...")
    qr_tensor = encode_to_qr(prompt, signature).to(device)
    print(f"QR code shape: {qr_tensor.shape}")
    
    # Generate initial noise
    print("Generating initial noise...")
    noise_xT = torch.randn(1, 4, 64, 64, device=device)
    
    # Embed QR code into noise
    print("Embedding watermark into noise...")
    with torch.no_grad():
        watermarked_noise_xTw = inn_model.forward(noise_xT, qr_tensor)
    print(f"Watermarked noise shape: {watermarked_noise_xTw.shape}")
    
    # Generate image using watermarked noise
    print("Generating image with watermarked noise...")
    with torch.no_grad():
        image = diffusion_util.generate_image(
            pipe,
            prompt,
            watermarked_noise_xTw,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale
        )
    
    # Save the image
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    image.save(args.output_path)
    print(f"Watermarked image saved to: {args.output_path}")
    
    # Optionally save the watermarked noise
    if args.save_noise:
        noise_path = args.output_path.replace('.png', '_noise.pt')
        torch.save(watermarked_noise_xTw.cpu(), noise_path)
        print(f"Watermarked noise saved to: {noise_path}")
    
    print("\n=== Embedding Summary ===")
    print(f"Prompt: {prompt}")
    print(f"Image: {args.output_path}")
    print(f"Public Key: {args.pk_path}")
    print("========================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaSeal-Noise Watermark Embedding")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output_path", type=str, default="watermarked_image.png", 
                        help="Output path for watermarked image")
    parser.add_argument("--model_path", type=str, default="models/inn_finetuned.pth",
                        help="Path to trained INN model")
    parser.add_argument("--sk_path", type=str, default="sk.pem", help="Path to private key")
    parser.add_argument("--pk_path", type=str, default="pk.pem", help="Path to public key")
    parser.add_argument("--num_blocks", type=int, default=16, help="Number of INN blocks")
    parser.add_argument("--diffusion_model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Stable Diffusion model ID")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--save_noise", action="store_true", help="Save watermarked noise tensor")
    
    args = parser.parse_args()
    embed_watermark(args)
