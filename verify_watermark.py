"""
MetaSeal-Noise Watermark Verification
Verify watermark in generated image by extracting and validating signature.
"""
import os
import argparse
import torch
from PIL import Image

from modules.inn_model import INNModel
from modules.pattern_util import decode_from_qr
from modules.crypto_util import verify_signature
from modules import diffusion_util


def verify_watermark(args):
    """Verify watermark in image."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if public key exists
    if not os.path.exists(args.pk_path):
        print(f"Error: Public key not found at {args.pk_path}")
        return
    
    # Load trained INN model
    print(f"Loading INN model from {args.model_path}")
    inn_model = INNModel(num_blocks=args.num_blocks, channels=4).to(device)
    inn_model.load_state_dict(torch.load(args.model_path, map_location=device))
    inn_model.eval()
    
    # Load Stable Diffusion pipeline
    print("Loading Stable Diffusion pipeline...")
    pipe = diffusion_util.load_model(args.diffusion_model, device=device)
    
    # Load image
    print(f"Loading image from {args.image_path}")
    image = Image.open(args.image_path).convert('RGB')
    
    # Invert image to get noise
    print("Inverting image to latent noise (this may take a while)...")
    with torch.no_grad():
        corrupted_noise = diffusion_util.invert_image(
            pipe,
            image,
            prompt="",
            num_inference_steps=args.num_inference_steps
        )
    print(f"Inverted noise shape: {corrupted_noise.shape}")
    
    # Extract watermark from noise
    print("Extracting watermark from noise...")
    with torch.no_grad():
        aux_z = torch.randn_like(corrupted_noise)
        extracted_qr = inn_model.reverse(corrupted_noise, aux_z)
    print(f"Extracted QR code shape: {extracted_qr.shape}")
    
    # Decode QR code
    print("Decoding QR code...")
    extracted_message, extracted_signature = decode_from_qr(extracted_qr)
    
    # Verify
    if extracted_message is None:
        print("\n" + "="*50)
        print("VERIFICATION FAILED: QR code could not be decoded")
        print("="*50)
        print("\nPossible reasons:")
        print("- Image may not contain a watermark")
        print("- Image may have been heavily modified")
        print("- Watermark extraction failed")
        return
    
    print(f"Extracted message: {extracted_message}")
    print(f"Extracted signature: {len(extracted_signature)} bytes")
    
    # Verify signature
    print("Verifying signature...")
    is_valid = verify_signature(args.pk_path, extracted_signature, extracted_message)
    
    # Print results
    print("\n" + "="*50)
    if is_valid:
        print("VERIFICATION SUCCESSFUL ✓")
    else:
        print("VERIFICATION FAILED ✗")
    print("="*50)
    print(f"\nRecovered prompt: {extracted_message}")
    print(f"Signature valid: {is_valid}")
    print(f"Public key used: {args.pk_path}")
    print("="*50 + "\n")
    
    # Save extracted QR code if requested
    if args.save_qr:
        qr_path = args.image_path.replace('.png', '_extracted_qr.png')
        qr_img = (extracted_qr.squeeze().cpu().numpy() * 255).astype('uint8')
        Image.fromarray(qr_img).save(qr_path)
        print(f"Extracted QR code saved to: {qr_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaSeal-Noise Watermark Verification")
    parser.add_argument("--image_path", type=str, required=True, help="Path to watermarked image")
    parser.add_argument("--pk_path", type=str, default="pk.pem", help="Path to public key")
    parser.add_argument("--model_path", type=str, default="models/inn_finetuned.pth",
                        help="Path to trained INN model")
    parser.add_argument("--num_blocks", type=int, default=16, help="Number of INN blocks")
    parser.add_argument("--diffusion_model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Stable Diffusion model ID")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of DDIM inversion steps")
    parser.add_argument("--save_qr", action="store_true", help="Save extracted QR code image")
    
    args = parser.parse_args()
    verify_watermark(args)
