"""
MetaSeal-Noise Stage 1 Training: Simulation Pretraining
Pretrain INN to embed/extract QR codes in simulated noise without using T2I model.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import kornia.augmentation as K
from tqdm import tqdm
import random
import string

from modules.inn_model import INNModel
from modules.pattern_util import encode_to_qr
from modules.crypto_util import generate_keys, sign_message


class SimulatedDataset(Dataset):
    """Dataset that generates random prompts and QR codes on-the-fly."""
    
    def __init__(self, size=1000, noise_shape=(4, 64, 64)):
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
        
        # Generate random noise
        noise = torch.randn(self.noise_shape)
        
        return noise, qr_tensor.squeeze(0)


class CorruptionLayer(nn.Module):
    """Simulates various corruptions that might occur during diffusion process."""
    
    def __init__(self):
        super(CorruptionLayer, self).__init__()
        self.corruptions = nn.Sequential(
            K.RandomGaussianNoise(mean=0., std=0.05, p=0.5),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.3),
        )
    
    def forward(self, x):
        return self.corruptions(x)


def train_stage1(args):
    """Main training function for Stage 1."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    inn_model = INNModel(num_blocks=args.num_blocks, channels=4).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(inn_model.parameters(), lr=args.lr)
    
    # Initialize corruption layer
    corruption_layer = CorruptionLayer().to(device)
    
    # Create dataset and dataloader
    dataset = SimulatedDataset(size=args.dataset_size, noise_shape=(4, 64, 64))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Training loop
    inn_model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_loss_rec = 0.0
        epoch_loss_emb = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for noise_xT, qr_tensor_V in pbar:
            noise_xT = noise_xT.to(device)
            qr_tensor_V = qr_tensor_V.to(device)
            
            # Forward: Embed QR code into noise
            watermarked_noise_xTw = inn_model.forward(noise_xT, qr_tensor_V)
            
            # Apply corruption
            corrupted_noise = corruption_layer(watermarked_noise_xTw)
            
            # Generate auxiliary noise for extraction
            aux_z = torch.randn_like(corrupted_noise)
            
            # Reverse: Extract QR code
            extracted_qr_tensor = inn_model.reverse(corrupted_noise, aux_z)
            
            # Compute losses
            # L_rec: QR code reconstruction loss
            loss_rec = F.mse_loss(qr_tensor_V, extracted_qr_tensor)
            
            # L_emb: Embedding invisibility loss (watermarked noise should be close to original)
            loss_emb = F.mse_loss(noise_xT, watermarked_noise_xTw)
            
            # Total loss
            total_loss = args.lambda_rec * loss_rec + args.lambda_emb * loss_emb
            
            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_loss += total_loss.item()
            epoch_loss_rec += loss_rec.item()
            epoch_loss_emb += loss_emb.item()
            
            pbar.set_postfix({
                'loss': total_loss.item(),
                'rec': loss_rec.item(),
                'emb': loss_emb.item()
            })
        
        # Print epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        avg_rec = epoch_loss_rec / len(dataloader)
        avg_emb = epoch_loss_emb / len(dataloader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Rec={avg_rec:.4f}, Emb={avg_emb:.4f}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f"checkpoints/inn_stage1_epoch{epoch+1}.pth"
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
    final_path = "models/inn_pretrained.pth"
    torch.save(inn_model.state_dict(), final_path)
    print(f"Stage 1 training complete! Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaSeal-Noise Stage 1 Training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Dataset size per epoch")
    parser.add_argument("--num_blocks", type=int, default=16, help="Number of invertible blocks")
    parser.add_argument("--lambda_rec", type=float, default=1.0, help="Weight for reconstruction loss")
    parser.add_argument("--lambda_emb", type=float, default=0.1, help="Weight for embedding loss")
    parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    train_stage1(args)
