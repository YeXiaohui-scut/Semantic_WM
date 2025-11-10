"""
MetaSeal-Noise Stage 2 Training: Real Noise Finetuning
Finetune INN extractor on real DDIM-inverted noise to bridge sim-to-real gap.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import string

from modules.inn_model import INNModel
from modules.pattern_util import encode_to_qr
from modules.crypto_util import generate_keys, sign_message
from modules import diffusion_util


class RealNoiseDataset(Dataset):
    """Dataset that generates real noise through T2I diffusion inversion."""
    
    def __init__(self, pipe, size=100, noise_shape=(4, 64, 64)):
        self.pipe = pipe
        self.size = size
        self.noise_shape = noise_shape
        
        # Generate key pair for signing
        if not os.path.exists('sk.pem'):
            generate_keys('sk.pem', 'pk.pem')
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random prompt
        prompt_length = random.randint(10, 50)
        prompt = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=prompt_length))
        
        # Sign prompt
        signature = sign_message('sk.pem', prompt)
        
        # Generate QR code
        qr_tensor = encode_to_qr(prompt, signature)
        
        return qr_tensor.squeeze(0), prompt


def train_stage2(args):
    """Main training function for Stage 2."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained INN model
    inn_model = INNModel(num_blocks=args.num_blocks, channels=4).to(device)
    if os.path.exists(args.pretrained_path):
        print(f"Loading pretrained model from {args.pretrained_path}")
        inn_model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
    else:
        print(f"Warning: Pretrained model not found at {args.pretrained_path}")
        print("Starting from scratch...")
    
    # Load Stable Diffusion pipeline
    print("Loading Stable Diffusion pipeline...")
    pipe = diffusion_util.load_model(args.model_id, device=device)
    
    # Freeze T2I model (U-Net)
    for param in pipe.unet.parameters():
        param.requires_grad = False
    for param in pipe.vae.parameters():
        param.requires_grad = False
    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    print("Frozen U-Net, VAE, and text encoder")
    
    # Freeze INN embedder (forward path)
    # Only finetune the extractor (reverse path)
    for name, param in inn_model.named_parameters():
        if 'blocks' in name:
            # Keep block parameters trainable for reverse path
            param.requires_grad = True
        else:
            # Freeze DWT/IWT and projection layers
            param.requires_grad = False
    
    print(f"Trainable parameters: {sum(p.numel() for p in inn_model.parameters() if p.requires_grad)}")
    
    # Initialize optimizer (only for extractor parameters)
    trainable_params = [p for p in inn_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    
    # Create dataset (smaller size due to computational cost)
    dataset = RealNoiseDataset(pipe, size=args.dataset_size, noise_shape=(4, 64, 64))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Training loop
    inn_model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for qr_tensor_V, prompts in pbar:
            qr_tensor_V = qr_tensor_V.to(device)
            
            batch_size = qr_tensor_V.shape[0]
            
            # Generate initial noise
            noise_xT = torch.randn(batch_size, 4, 64, 64, device=device)
            
            # === Forward Pipeline (All Frozen) ===
            with torch.no_grad():
                # 1. Embed QR code into noise
                watermarked_noise_xTw = inn_model.forward(noise_xT, qr_tensor_V)
                
                # 2. Generate image using T2I model
                images = []
                for i in range(batch_size):
                    prompt = prompts[i] if prompts[i].strip() else "a photo"
                    image = diffusion_util.generate_image(
                        pipe, 
                        prompt, 
                        watermarked_noise_xTw[i:i+1],
                        num_inference_steps=args.num_inference_steps
                    )
                    images.append(image)
                
                # 3. Invert images back to noise
                corrupted_noises = []
                for image in images:
                    corrupted_noise = diffusion_util.invert_image(
                        pipe, 
                        image, 
                        prompt="",
                        num_inference_steps=args.num_inference_steps
                    )
                    corrupted_noises.append(corrupted_noise)
                
                corrupted_noise_batch = torch.cat(corrupted_noises, dim=0)
            
            # === Reverse Extraction (Trainable) ===
            # Generate auxiliary noise for extraction
            aux_z = torch.randn_like(corrupted_noise_batch)
            
            # Extract QR code (this is where we train)
            extracted_qr_tensor = inn_model.reverse(corrupted_noise_batch, aux_z)
            
            # Compute reconstruction loss
            loss_rec = F.mse_loss(qr_tensor_V, extracted_qr_tensor)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss_rec.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss_rec.item()
            
            pbar.set_postfix({'loss': loss_rec.item()})
        
        # Print epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f"checkpoints/inn_stage2_epoch{epoch+1}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': inn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    final_path = "models/inn_finetuned.pth"
    torch.save(inn_model.state_dict(), final_path)
    print(f"Stage 2 training complete! Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaSeal-Noise Stage 2 Training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (small due to diffusion)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--dataset_size", type=int, default=50, help="Dataset size per epoch")
    parser.add_argument("--num_blocks", type=int, default=16, help="Number of invertible blocks")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="DDIM steps for generation/inversion")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--pretrained_path", type=str, default="models/inn_pretrained.pth", 
                        help="Path to Stage 1 pretrained model")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Stable Diffusion model ID")
    
    args = parser.parse_args()
    train_stage2(args)
