# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VQ-VAE implementations on MNIST using Flax NNX (not Linen) and JAX. Two variants:
- `main_vq-vae.py` — Standard Euclidean VQ-VAE with embedding-based codebook lookup
- `main_hypervq.py` — Hyperbolic VQ-VAE (HyperVQ) based on [Goswami et al. 2025](https://arxiv.org/abs/2403.13015). Uses Poincare ball MLR for quantization with Gumbel-Softmax straight-through estimator.

Both share the same encoder/decoder architecture (Conv → AvgPool → ConvTranspose) and training pattern (nnx.value_and_grad, nnx.Optimizer with adamw, nnx.MultiMetric).

## Commands

```bash
# Install dependencies
uv sync

# Run standard VQ-VAE training
uv run python main_vq-vae.py

# Run hyperbolic VQ-VAE training
uv run python main_hypervq.py

# Override config via tyro CLI flags
uv run python main_vq-vae.py --num-epochs 5 --batch-size 64
```

## Verification (quick smoke test)

```bash
uv run python main_vq-vae.py --num-epochs 1 --batch-size 32
uv run python main_hypervq.py --num-epochs 1 --batch-size 32
```
Passes if it prints loss values for at least one step without error. Checkpoints saved to `checkpoints/{vqvae,hypervq}/epoch_N/`.

```bash
# Visualize reconstructions (after training)
uv run python visualize.py --model vqvae
uv run python visualize.py --model hypervq --checkpoint-path checkpoints/hypervq/epoch_0
```

## Architecture

Both models follow: **Config dataclass → nnx.Module → loss_fn → train_step/eval_step → training loop**

- Config: `tyro.cli(Config)` parses CLI args from a `@dataclass`
- Data: HuggingFace `datasets` (MNIST), normalized to [0,1] at training time (not in forward pass — intentional, to compute loss against normalized targets)
- Quantization differs between variants:
  - Euclidean: L2 distance to `nnx.Embed` codebook via vmapped per-vector distance, with commitment loss + stop_gradient STE on the code vectors
  - Hyperbolic: `expmap_0` → `HypRegressionPoincarePP` (MLR) → Gumbel-Softmax STE at the **one-hot weights level** → `st_weights @ (r_k * a_k)` codebook matmul. Codebook vectors are `r_k * a_k` (MLR bias * kernel). The STE must be on the categorical weights (`soft + sg(hard - soft)`), NOT on the final z_q vector, otherwise the codebook receives no gradients.
- Euclidean VQ loss: reconstruction MSE + commitment loss
- HyperVQ loss: reconstruction MSE only (Gumbel-Softmax STE handles gradient flow to all parameter groups: θ_E, θ_D, θ_Q)
- Visualization: `visualize.py` loads checkpoints and plots ground truth vs reconstruction

## Key Dependencies

- `flax.nnx` (NOT `flax.linen`) — module system, optimizer, metrics, jit
- `hyperbolix` — Poincare manifold ops, hyperbolic regression layer
- `tyro` — CLI config from dataclasses
- `optax` — optimizer schedules
- `grain` — data loading (used in hypervq variant)
