# MetaSeal-Noise Examples

This directory contains example scripts demonstrating how to use MetaSeal-Noise.

## Quick Start Example

### 1. Generate Keys (one-time setup)
```python
from modules.crypto_util import generate_keys

# Generate ECDSA key pair
generate_keys('sk.pem', 'pk.pem')
```

### 2. Embed Watermark
```bash
python embed_watermark.py \
    --prompt "A serene mountain landscape at sunset" \
    --output_path outputs/watermarked_image.png \
    --model_path models/inn_finetuned.pth
```

### 3. Verify Watermark
```bash
python verify_watermark.py \
    --image_path outputs/watermarked_image.png \
    --pk_path pk.pem \
    --model_path models/inn_finetuned.pth
```

## Training Pipeline

### Stage 1: Simulation Pretraining
```bash
# Train INN on simulated noise (fast, ~1-2 hours on GPU)
python train_inn_stage1_pretrain.py \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --dataset_size 1000
```

### Stage 2: Real Noise Finetuning
```bash
# Finetune on real DDIM-inverted noise (slow, ~4-6 hours on GPU)
python train_inn_stage2_finetune.py \
    --epochs 20 \
    --batch_size 2 \
    --lr 5e-5 \
    --dataset_size 50 \
    --pretrained_path models/inn_pretrained.pth
```

## Advanced Usage

### Custom Diffusion Model
```bash
python embed_watermark.py \
    --prompt "Your prompt here" \
    --diffusion_model "stabilityai/stable-diffusion-2-1" \
    --output_path custom_output.png
```

### Higher Quality Generation
```bash
python embed_watermark.py \
    --prompt "Your prompt here" \
    --num_inference_steps 100 \
    --guidance_scale 9.0
```

### Save Watermarked Noise
```bash
python embed_watermark.py \
    --prompt "Your prompt here" \
    --save_noise  # Saves noise tensor for debugging
```

## Integration Test

Run the integration test to verify all components work:
```bash
python test_integration.py
```

This will test:
- ECDSA key generation and signature verification
- QR code encoding and decoding
- INN model forward and reverse passes
- End-to-end workflow (without trained model)

## Notes

- Training requires a GPU with at least 16GB VRAM for Stage 2
- Stage 1 can run on CPU but will be slower
- For best results, use the finetuned model from Stage 2
- QR code decoding may fail on heavily corrupted images
- Increase `num_inference_steps` for better inversion quality
