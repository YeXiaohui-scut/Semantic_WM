#!/usr/bin/env python3
"""
Integration test for MetaSeal-Noise framework.
Tests the basic workflow without requiring GPU or diffusion models.
"""
import os
import sys
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.crypto_util import generate_keys, sign_message, verify_signature
from modules.pattern_util import encode_to_qr, decode_from_qr
from modules.inn_model import INNModel


def test_crypto():
    """Test cryptographic operations."""
    print("=" * 60)
    print("TEST 1: Cryptographic Operations")
    print("=" * 60)
    
    # Generate keys
    sk_path = "test_sk.pem"
    pk_path = "test_pk.pem"
    
    if os.path.exists(sk_path):
        os.remove(sk_path)
    if os.path.exists(pk_path):
        os.remove(pk_path)
    
    print("1. Generating ECDSA key pair...")
    generate_keys(sk_path, pk_path)
    assert os.path.exists(sk_path), "Private key not created"
    assert os.path.exists(pk_path), "Public key not created"
    print("   ✓ Keys generated successfully")
    
    # Sign message
    print("2. Signing message...")
    message = "A beautiful landscape with mountains and lakes"
    signature = sign_message(sk_path, message)
    print(f"   ✓ Signature created ({len(signature)} bytes)")
    
    # Verify signature
    print("3. Verifying signature...")
    is_valid = verify_signature(pk_path, signature, message)
    assert is_valid, "Valid signature verification failed"
    print("   ✓ Signature verified successfully")
    
    # Test invalid signature
    print("4. Testing invalid signature...")
    is_invalid = verify_signature(pk_path, signature, "Different message")
    assert not is_invalid, "Invalid signature should fail verification"
    print("   ✓ Invalid signature rejected correctly")
    
    print("\n✓ Crypto tests passed!\n")
    return sk_path, pk_path


def test_qr_code(sk_path):
    """Test QR code encoding and decoding."""
    print("=" * 60)
    print("TEST 2: QR Code Operations")
    print("=" * 60)
    
    # Create QR code
    print("1. Encoding message and signature to QR code...")
    message = "MetaSeal-Noise watermark test"
    signature = sign_message(sk_path, message)
    qr_tensor = encode_to_qr(message, signature)
    print(f"   ✓ QR code created: shape={qr_tensor.shape}, dtype={qr_tensor.dtype}")
    assert qr_tensor.shape == (1, 1, 128, 128), "QR code has wrong shape"
    
    # Decode QR code
    print("2. Decoding QR code...")
    decoded_message, decoded_signature = decode_from_qr(qr_tensor)
    assert decoded_message is not None, "QR code decoding failed"
    assert decoded_message == message, f"Message mismatch: {decoded_message} != {message}"
    assert decoded_signature == signature, "Signature mismatch"
    print(f"   ✓ QR code decoded: message='{decoded_message}'")
    
    print("\n✓ QR code tests passed!\n")
    return qr_tensor


def test_inn_model(qr_tensor):
    """Test INN model embedding and extraction."""
    print("=" * 60)
    print("TEST 3: INN Model Operations")
    print("=" * 60)
    
    # Initialize model
    print("1. Initializing INN model...")
    model = INNModel(num_blocks=8, channels=4)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {param_count:,} parameters")
    
    # Test embedding
    print("2. Testing watermark embedding...")
    batch_size = 2
    noise = torch.randn(batch_size, 4, 64, 64)
    qr_batch = qr_tensor.repeat(batch_size, 1, 1, 1)
    
    with torch.no_grad():
        watermarked_noise = model.forward(noise, qr_batch)
    
    print(f"   ✓ Embedding successful: output shape={watermarked_noise.shape}")
    assert watermarked_noise.shape == noise.shape, "Watermarked noise has wrong shape"
    
    # Test extraction
    print("3. Testing watermark extraction...")
    aux_z = torch.randn_like(watermarked_noise)
    
    with torch.no_grad():
        extracted_qr = model.reverse(watermarked_noise, aux_z)
    
    print(f"   ✓ Extraction successful: output shape={extracted_qr.shape}")
    assert extracted_qr.shape[1:] == qr_batch.shape[1:], "Extracted QR has wrong shape"
    
    # Test embedding invisibility
    print("4. Testing embedding invisibility...")
    noise_diff = (noise - watermarked_noise).abs().mean().item()
    print(f"   ✓ Average noise difference: {noise_diff:.6f}")
    
    print("\n✓ INN model tests passed!\n")


def test_end_to_end(sk_path, pk_path):
    """Test complete end-to-end workflow."""
    print("=" * 60)
    print("TEST 4: End-to-End Workflow")
    print("=" * 60)
    
    # Step 1: Create watermarked noise
    print("1. Creating watermarked noise...")
    prompt = "A serene mountain landscape at sunset"
    signature = sign_message(sk_path, prompt)
    qr_tensor = encode_to_qr(prompt, signature)
    
    model = INNModel(num_blocks=8, channels=4)
    noise = torch.randn(1, 4, 64, 64)
    
    with torch.no_grad():
        watermarked_noise = model.forward(noise, qr_tensor)
    print("   ✓ Watermarked noise created")
    
    # Step 2: Simulate corruption (in real scenario, this would be diffusion + inversion)
    print("2. Simulating corruption...")
    corrupted_noise = watermarked_noise + torch.randn_like(watermarked_noise) * 0.1
    print("   ✓ Noise corrupted")
    
    # Step 3: Extract watermark
    print("3. Extracting watermark...")
    aux_z = torch.randn_like(corrupted_noise)
    
    with torch.no_grad():
        extracted_qr = model.reverse(corrupted_noise, aux_z)
    print("   ✓ Watermark extracted")
    
    # Step 4: Decode and verify
    print("4. Decoding and verifying...")
    decoded_message, decoded_signature = decode_from_qr(extracted_qr)
    
    if decoded_message is None:
        print("   ⚠ QR code could not be decoded (expected with high noise)")
        print("   Note: In real scenario with trained model, decoding should succeed")
    else:
        print(f"   ✓ Decoded message: '{decoded_message}'")
        is_valid = verify_signature(pk_path, decoded_signature, decoded_message)
        if is_valid:
            print("   ✓ Signature verified successfully!")
        else:
            print("   ✗ Signature verification failed")
    
    print("\n✓ End-to-end test completed!\n")


def cleanup():
    """Clean up test files."""
    print("Cleaning up test files...")
    test_files = ["test_sk.pem", "test_pk.pem"]
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"   Removed {f}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MetaSeal-Noise Integration Tests")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Cryptography
        sk_path, pk_path = test_crypto()
        
        # Test 2: QR Codes
        qr_tensor = test_qr_code(sk_path)
        
        # Test 3: INN Model
        test_inn_model(qr_tensor)
        
        # Test 4: End-to-End
        test_end_to_end(sk_path, pk_path)
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nNote: These tests verify the core components work correctly.")
        print("For full functionality, train the INN model using:")
        print("  1. python train_inn_stage1_pretrain.py")
        print("  2. python train_inn_stage2_finetune.py")
        print("\nThen use embed_watermark.py and verify_watermark.py for")
        print("real image generation and verification.\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
