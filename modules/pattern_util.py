"""
QR code utilities for encoding and decoding watermark patterns.
Converts message + signature to QR codes and vice versa.
"""
import io
import base64
import qrcode
import numpy as np
import torch
from PIL import Image
from pyzbar import pyzbar


SEPARATOR = b'|||SEP|||'  # Special separator for message and signature
QR_SIZE = 128  # Fixed QR code size (larger for better decoding)


def encode_to_qr(message_str, signature_bytes):
    """
    Encode message and signature into a QR code tensor.
    
    Args:
        message_str (str): Message to encode
        signature_bytes (bytes): Signature bytes
    
    Returns:
        torch.Tensor: Binary QR code tensor of shape (1, 1, QR_SIZE, QR_SIZE)
    """
    # Pack message and signature together
    # Format: base64(message_utf8) + SEPARATOR + base64(signature)
    message_b64 = base64.b64encode(message_str.encode('utf-8'))
    signature_b64 = base64.b64encode(signature_bytes)
    packed_data = message_b64 + SEPARATOR + signature_b64
    
    # Generate QR code
    qr = qrcode.QRCode(
        version=None,  # Auto-size
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(packed_data)
    qr.make(fit=True)
    
    # Convert to PIL Image
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # Resize to fixed size
    qr_img = qr_img.resize((QR_SIZE, QR_SIZE), Image.LANCZOS)
    
    # Convert to numpy array and normalize to 0-1
    qr_array = np.array(qr_img.convert('L'))
    qr_array = (qr_array < 128).astype(np.float32)  # Binarize: black=1, white=0
    
    # Convert to torch tensor with shape (1, 1, H, W)
    qr_tensor = torch.from_numpy(qr_array).unsqueeze(0).unsqueeze(0)
    
    return qr_tensor


def decode_from_qr(qr_tensor):
    """
    Decode QR code tensor to extract message and signature.
    
    Args:
        qr_tensor (torch.Tensor): QR code tensor with values in [0, 1]
    
    Returns:
        tuple: (message_str, signature_bytes) or (None, None) if decoding fails
    """
    try:
        # Convert tensor to numpy array
        if qr_tensor.dim() == 4:
            qr_array = qr_tensor.squeeze().cpu().numpy()
        elif qr_tensor.dim() == 3:
            qr_array = qr_tensor.squeeze(0).cpu().numpy()
        else:
            qr_array = qr_tensor.cpu().numpy()
        
        # Binarize: threshold at 0.5 and convert to 0-255 range
        qr_array = (qr_array > 0.5).astype(np.uint8) * 255
        
        # Invert: QR codes need black on white
        qr_array = 255 - qr_array
        
        # Convert to PIL Image
        qr_img = Image.fromarray(qr_array, mode='L')
        
        # Try to decode QR code
        decoded_objects = pyzbar.decode(qr_img)
        
        if not decoded_objects:
            return None, None
        
        # Get data from first decoded object
        packed_data = decoded_objects[0].data
        
        # Split message and signature
        if SEPARATOR not in packed_data:
            return None, None
        
        parts = packed_data.split(SEPARATOR)
        if len(parts) != 2:
            return None, None
        
        message_b64, signature_b64 = parts
        
        # Decode from base64
        message_str = base64.b64decode(message_b64).decode('utf-8')
        signature_bytes = base64.b64decode(signature_b64)
        
        return message_str, signature_bytes
    
    except Exception as e:
        print(f"QR code decoding failed: {e}")
        return None, None
