# MetaSeal-Noise: Semantic Watermarking for Text-to-Image Generation

MetaSeal-Noise is a PyTorch-based framework for embedding semantic watermarks into Text-to-Image (T2I) diffusion models. It innovatively applies MetaSeal's (arXiv:2509.10766) "asymmetric encryption" and "invertible embedding" concepts to the initial noise ($x_T$) of T2I diffusion models.

## Overview

The core workflow:
1. **Sign** the user's prompt using ECDSA digital signature
2. **Encode** the prompt + signature into a QR code
3. **Embed** the QR code into the initial noise $x_T$ using an Invertible Neural Network (INN)
4. **Generate** the image using the watermarked noise $x_T^w$ with Stable Diffusion
5. **Verify** by inverting the image back to noise and extracting the QR code

## Project Structure

```
.
├── embed_watermark.py              # Inference: Embed watermark in image generation
├── verify_watermark.py             # Inference: Verify watermark in image
├── train_inn_stage1_pretrain.py   # Training Stage 1: Simulation pretraining
├── train_inn_stage2_finetune.py   # Training Stage 2: Real noise finetuning
├── modules/
│   ├── crypto_util.py             # ECDSA signature generation and verification
│   ├── pattern_util.py            # QR code encoding and decoding
│   ├── inn_model.py               # Invertible Neural Network architecture
│   └── diffusion_util.py          # Stable Diffusion utilities
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YeXiaohui-scut/Semantic_WM.git
cd Semantic_WM

# Install dependencies
pip install -r requirements.txt

# For QR code decoding, you may need to install zbar:
# Ubuntu/Debian: sudo apt-get install libzbar0
# macOS: brew install zbar
# Windows: Download from http://zbar.sourceforge.net/
```

## Usage

### 1. Training (Two-Stage Pipeline)

#### Stage 1: Simulation Pretraining
Quickly pretrain the INN to embed/extract QR codes in simulated noise without using the T2I model.

```bash
python train_inn_stage1_pretrain.py \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --dataset_size 1000 \
    --lambda_rec 1.0 \
    --lambda_emb 0.1
```

This saves the pretrained model to `models/inn_pretrained.pth`.

#### Stage 2: Real Noise Finetuning
Finetune the INN extractor on real DDIM-inverted noise to bridge the simulation-to-reality gap.

```bash
python train_inn_stage2_finetune.py \
    --epochs 20 \
    --batch_size 2 \
    --lr 5e-5 \
    --dataset_size 50 \
    --pretrained_path models/inn_pretrained.pth \
    --num_inference_steps 50
```

This saves the finetuned model to `models/inn_finetuned.pth`.

**Note:** Stage 2 requires a GPU with sufficient memory for Stable Diffusion inference and is computationally expensive.

### 2. Embedding Watermarks

Generate an image with an embedded watermark:

```bash
python embed_watermark.py \
    --prompt "A beautiful sunset over the ocean" \
    --output_path watermarked_image.png \
    --model_path models/inn_finetuned.pth \
    --num_inference_steps 50 \
    --guidance_scale 7.5
```

This will:
- Generate ECDSA keys (if they don't exist) → `sk.pem`, `pk.pem`
- Sign the prompt with the private key
- Encode prompt + signature into a QR code
- Embed the QR code into the initial noise
- Generate the image using Stable Diffusion
- Save the watermarked image

### 3. Verifying Watermarks

Verify the watermark in a generated image:

```bash
python verify_watermark.py \
    --image_path watermarked_image.png \
    --pk_path pk.pem \
    --model_path models/inn_finetuned.pth \
    --num_inference_steps 50
```

This will:
- Load the image and invert it back to noise using DDIM inversion
- Extract the QR code from the noise using the INN
- Decode the QR code to get the prompt and signature
- Verify the signature using the public key
- Display the verification result and recovered prompt

## Architecture Details

### Modules

#### 1. crypto_util.py
- `generate_keys()`: Generate ECDSA key pair (NIST256p curve)
- `sign_message(sk_path, message)`: Sign a message with private key
- `verify_signature(pk_path, signature, message)`: Verify signature with public key

#### 2. pattern_util.py
- `encode_to_qr(message, signature)`: Encode message + signature into 53×53 binary QR code tensor
- `decode_from_qr(qr_tensor)`: Decode QR code tensor back to message + signature

#### 3. inn_model.py
- `InvBlock`: Invertible block with φ, ρ, η sub-networks (MetaSeal architecture)
- `INNModel`: Complete INN with DWT/IWT and 16 stacked invertible blocks
  - `forward(x_T, V)`: Embed QR code V into noise x_T → watermarked noise x_T^w
  - `reverse(x_T^w, z)`: Extract QR code from watermarked noise

#### 4. diffusion_util.py
- `load_model(model_id)`: Load Stable Diffusion pipeline
- `generate_image(pipe, prompt, watermarked_noise)`: Generate image from watermarked noise
- `invert_image(pipe, image)`: DDIM inversion to recover noise from image

### Training Strategy

**Stage 1 (Simulation):**
- Fast pretraining on simulated noise
- Losses:
  - L_rec: QR code reconstruction loss
  - L_emb: Embedding invisibility loss (watermarked noise ≈ original noise)
- Uses corruption layers (Gaussian noise, blur) to simulate real-world degradation

**Stage 2 (Real Noise):**
- Finetune only the extractor (reverse path) on real DDIM-inverted noise
- Freeze: T2I model (U-Net, VAE, text encoder) + INN embedder
- Unfreeze: INN extractor only
- Loss: L_rec (QR code reconstruction)
- Bridges the gap between simulation and real diffusion process

## Key Features

✅ **Asymmetric Cryptography**: ECDSA signatures for authentication  
✅ **Invertible Embedding**: Reversible watermark embedding with minimal distortion  
✅ **Semantic Binding**: Watermark is bound to the prompt  
✅ **Robust Verification**: Survives the diffusion generation process  
✅ **Two-Stage Training**: Efficient pretraining + real-world finetuning  

## Technical Details

- **Base Model**: Stable Diffusion v1.5
- **INN Architecture**: 16 invertible blocks with DWT/IWT
- **QR Code Size**: 53×53 pixels
- **Latent Space**: 4 channels, 64×64 resolution
- **Signature**: ECDSA with NIST256p curve

## Citation

This project is inspired by MetaSeal (arXiv:2509.10766). If you use this code, please cite:

```bibtex
@article{metaseal2025,
  title={MetaSeal: Semantic Watermarking via Meta-Learning},
  journal={arXiv preprint arXiv:2509.10766},
  year={2025}
}
```

## License

This project is for research purposes only.

## Troubleshooting

**QR Code Decoding Issues:**
- Install zbar library for your OS
- Ensure image has not been heavily compressed or modified
- Try increasing `num_inference_steps` for better inversion quality

**GPU Memory Issues:**
- Reduce `batch_size` in training scripts
- Use smaller `num_inference_steps`
- Consider using mixed precision training

**Verification Fails:**
- Ensure you're using the correct public key
- Check that the image was generated with the same model weights
- Verify that DDIM inversion steps match embedding steps