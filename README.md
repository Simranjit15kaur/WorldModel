# Video Prediction World Model

A from-scratch implementation of four video prediction architectures trained on Moving MNIST, built in PyTorch and designed to run on Google Colab.

The project progresses from a vanilla VAE encoder through to a causal GPT that predicts video frame-by-frame as discrete tokens — the same approach used by VideoGPT and DALL-E.

---

## Models

| Model | Description |
|---|---|
| **VAE** | Convolutional encoder/decoder compressing 64×64 frames into 128-channel 8×8 spatial latent maps |
| **ConvLSTM Dynamics** | Two stacked ConvLSTM layers predicting future latent maps while preserving spatial structure |
| **Spatial Transformer Dynamics** | Full self-attention over spatial tokens; sharper and more coherent than ConvLSTM at longer horizons |
| **VQ-VAE + VideoGPT** | Frames tokenised into a discrete 512-entry codebook; a causal GPT autoregressively predicts future tokens |

---

## Dataset

**Moving MNIST** — generated on-the-fly. Each sample is a 20-frame sequence (T=20, 64×64, grayscale):
- Frames 0–9 are fed as observed context
- Frames 10–19 are the prediction targets

A single MNIST digit bounces around the canvas with random velocity and wall reflection.

---

## Architecture

```
Observed Frames (10 × 64×64)
         │
         ▼
   VAE Encoder  →  Latent Maps (128, 8, 8)
         │
         ▼
ConvLSTM / Spatial Transformer Dynamics
   + Scheduled Sampling during training
         │
         ▼
Autoregressive Rollout  +  optional noise injection
         │
         ▼
   VAE Decoder  →  10 Future Frames (64×64)

── Parallel discrete path ─────────────────
VQ-VAE  →  Token grid (16×16 = 256 tokens/frame)
VideoGPT  →  Causal token prediction across frames
```

### VideoGPT specifics
- **2D spatial positional encoding** — each token at grid position (row, col) gets a unique sinusoidal signal, preserving within-frame structure
- **[FRAME] separator tokens** — explicit boundaries between frames in the token sequence
- **Weight tying** — input embedding matrix shared with the output projection head
- **Nucleus sampling** during generation for diversity

---

## Evaluation

Models are evaluated with two metrics:

- **SSIM** (Structural Similarity Index) — measures pixel-level structural fidelity
- **LPIPS** (Learned Perceptual Image Patch Similarity) — measures perceptual quality using deep features

Both are computed per future step so you can see how prediction quality degrades over the rollout horizon.

---

## Getting Started

### Requirements

```bash
pip install torch torchvision matplotlib scikit-image imageio tqdm lpips
```

### Run on Colab

1. Open `WorldModel.ipynb` in Google Colab
2. Mount your Google Drive — checkpoints and output images are saved to `/content/drive/MyDrive/WorldModel/`
3. Run all cells in order; each section is self-contained

A GPU runtime is strongly recommended (Runtime → Change runtime type → T4 GPU).

### Saved Checkpoints

| File | Contents |
|---|---|
| `vae.pth` | VAE weights + training loss history |
| `dynamics.pth` | ConvLSTM dynamics model |
| `spatial_transformer.pth` | Spatial Transformer dynamics model |
| `vqvae.pth` | VQ-VAE weights |
| `video_gpt_light.pth` | VideoGPT weights |

---

## Output Visualisations

The notebook saves the following images to your Drive:

| File | What it shows |
|---|---|
| `vae_reconstruction.png` | Original vs VAE-reconstructed frames |
| `convlstm_predictions.png` | Observed / ConvLSTM predicted / ground truth grid |
| `spatial_transformer_predictions.png` | Same grid for the Spatial Transformer |
| `stochastic_futures.png` | One past → three diverging futures at different noise scales |
| `vqvae_reconstruction.png` | VQ-VAE reconstruction from discrete codebook |
| `gpt_predictions.png` | VideoGPT 5-frame autoregressive generation |
| `dual_metric.png` | SSIM + LPIPS curves for all models |
| `training_curves.png` | Loss curves for all four models |
| `model_comparison.png` | Side-by-side prediction grid for all models |
| `token_visualization.png` | How a video sequence looks as a GPT token stream |
| `architecture_card.png` | Summary card (dark theme, shareable) |

---

## Project Structure

```
WorldModel.ipynb          # Main notebook — all code and experiments
README.md
```

---

## References

- [VQ-VAE: Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
- [Convolutional LSTM Network](https://arxiv.org/abs/1506.04214)
- [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)

---

## License

MIT
