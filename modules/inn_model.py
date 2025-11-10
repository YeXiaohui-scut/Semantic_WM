"""
Invertible Neural Network (INN) for watermark embedding and extraction.
Based on MetaSeal architecture with reversible blocks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubNetwork(nn.Module):
    """
    Sub-network used in invertible blocks (φ, ρ, η).
    Simple 3-layer convolutional network.
    """
    def __init__(self, in_channels, out_channels):
        super(SubNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class InvBlock(nn.Module):
    """
    Invertible block implementing MetaSeal's reversible architecture.
    
    Forward: (I^i, V) -> I^{i+1}
    Reverse: (I^{i+1}, z) -> V_hat
    """
    def __init__(self, channels):
        super(InvBlock, self).__init__()
        # Three sub-networks: φ, ρ, η
        self.phi = SubNetwork(channels, channels)
        self.rho = SubNetwork(channels, channels)
        self.eta = SubNetwork(channels, channels)
    
    def forward(self, x, v):
        """
        Forward pass: embed watermark pattern v into carrier x.
        
        Args:
            x (torch.Tensor): Carrier features (I^i)
            v (torch.Tensor): Watermark pattern (V)
        
        Returns:
            torch.Tensor: Watermarked features (I^{i+1})
        """
        # Split x into two halves
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        # Forward coupling equations (MetaSeal Eq. 9-10)
        y1 = x1 + self.phi(x2)
        y2 = x2 + self.rho(y1) + self.eta(v)
        
        # Concatenate back
        y = torch.cat([y1, y2], dim=1)
        return y
    
    def reverse(self, y, z):
        """
        Reverse pass: extract watermark pattern from watermarked features.
        
        Args:
            y (torch.Tensor): Watermarked features (I^{i+1})
            z (torch.Tensor): Auxiliary noise
        
        Returns:
            tuple: (x_recovered, v_extracted)
        """
        # Split y into two halves
        y1, y2 = torch.chunk(y, 2, dim=1)
        
        # Reverse coupling equations (MetaSeal Eq. 11-12)
        x2 = y2 - self.rho(y1) - self.eta(z)
        x1 = y1 - self.phi(x2)
        
        # Concatenate recovered carrier
        x = torch.cat([x1, x2], dim=1)
        
        # Extract watermark from residual
        v_extracted = self.eta(z)
        
        return x, v_extracted


class DWT(nn.Module):
    """
    Discrete Wavelet Transform using Haar wavelets.
    Decomposes image into LL, LH, HL, HH subbands.
    """
    def __init__(self):
        super(DWT, self).__init__()
        # Haar wavelet filters
        self.requires_grad = False
    
    def forward(self, x):
        """
        Apply 2D DWT.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            torch.Tensor: DWT coefficients (B, C*4, H/2, W/2)
        """
        # Simple Haar wavelet implementation
        # LL (average), LH (horizontal), HL (vertical), HH (diagonal)
        x01 = x[:, :, 0::2, :] / 2  # even rows
        x02 = x[:, :, 1::2, :] / 2  # odd rows
        
        LL = x01[:, :, :, 0::2] + x02[:, :, :, 0::2] + x01[:, :, :, 1::2] + x02[:, :, :, 1::2]
        LH = x01[:, :, :, 0::2] + x02[:, :, :, 0::2] - x01[:, :, :, 1::2] - x02[:, :, :, 1::2]
        HL = x01[:, :, :, 0::2] - x02[:, :, :, 0::2] + x01[:, :, :, 1::2] - x02[:, :, :, 1::2]
        HH = x01[:, :, :, 0::2] - x02[:, :, :, 0::2] - x01[:, :, :, 1::2] + x02[:, :, :, 1::2]
        
        return torch.cat([LL, LH, HL, HH], dim=1)


class IWT(nn.Module):
    """
    Inverse Discrete Wavelet Transform.
    Reconstructs image from DWT coefficients.
    """
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False
    
    def forward(self, x):
        """
        Apply 2D IWT.
        
        Args:
            x: DWT coefficients (B, C*4, H, W)
        
        Returns:
            torch.Tensor: Reconstructed image (B, C, H*2, W*2)
        """
        # Split into subbands
        C = x.shape[1] // 4
        LL, LH, HL, HH = x[:, :C], x[:, C:2*C], x[:, 2*C:3*C], x[:, 3*C:]
        
        # Reconstruct
        x1 = (LL + LH + HL + HH) / 4
        x2 = (LL + LH - HL - HH) / 4
        x3 = (LL - LH + HL - HH) / 4
        x4 = (LL - LH - HL + HH) / 4
        
        # Interleave
        B, C, H, W = LL.shape
        out = torch.zeros(B, C, H*2, W*2, device=x.device, dtype=x.dtype)
        out[:, :, 0::2, 0::2] = x1
        out[:, :, 0::2, 1::2] = x3
        out[:, :, 1::2, 0::2] = x2
        out[:, :, 1::2, 1::2] = x4
        
        return out


class INNModel(nn.Module):
    """
    Complete Invertible Neural Network for watermark embedding/extraction.
    """
    def __init__(self, num_blocks=16, channels=4):
        super(INNModel, self).__init__()
        self.num_blocks = num_blocks
        self.channels = channels
        
        # DWT and IWT layers
        self.dwt = DWT()
        self.iwt = IWT()
        
        # Stack of invertible blocks
        self.blocks = nn.ModuleList([
            InvBlock(channels * 4) for _ in range(num_blocks)
        ])
        
        # Projection layer for QR code to match DWT dimensions
        self.qr_proj = nn.Conv2d(1, channels * 4, kernel_size=1)
    
    def forward(self, x_T, V):
        """
        Embed QR code V into noise x_T.
        
        Args:
            x_T (torch.Tensor): Initial noise (B, C, H, W)
            V (torch.Tensor): QR code pattern (B, 1, QR_H, QR_W)
        
        Returns:
            torch.Tensor: Watermarked noise x_T^w
        """
        # Apply DWT to noise
        I = self.dwt(x_T)
        
        # Resize QR code to match DWT dimensions
        B, _, H, W = I.shape
        V_resized = F.interpolate(V, size=(H//4, W//4), mode='nearest')
        V_proj = self.qr_proj(V_resized)
        
        # Pass through invertible blocks
        for block in self.blocks:
            I = block(I, V_proj)
        
        # Apply IWT to get watermarked noise
        x_T_w = self.iwt(I)
        
        return x_T_w
    
    def reverse(self, x_T_w_corrupted, z):
        """
        Extract QR code from corrupted watermarked noise.
        
        Args:
            x_T_w_corrupted (torch.Tensor): Corrupted watermarked noise
            z (torch.Tensor): Auxiliary Gaussian noise
        
        Returns:
            torch.Tensor: Extracted QR code pattern V_hat
        """
        # Apply DWT to corrupted noise
        I = self.dwt(x_T_w_corrupted)
        
        # Get dimensions for auxiliary noise
        B, C, H, W = I.shape
        z_dwt = self.dwt(z)
        z_resized = F.interpolate(z_dwt, size=(H//4, W//4), mode='nearest')[:, :C]
        
        # Pass through invertible blocks in reverse
        v_list = []
        for block in reversed(self.blocks):
            I, v_extracted = block.reverse(I, z_resized)
            v_list.append(v_extracted)
        
        # Average extracted watermarks and project back to QR size
        v_avg = torch.stack(v_list).mean(dim=0)
        V_hat = F.interpolate(v_avg[:, :1], size=(53, 53), mode='bilinear', align_corners=False)
        
        return V_hat
