"""Visualize VQ-VAE reconstructions from a saved checkpoint."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
import tyro
from datasets import load_dataset
from flax import nnx

# Import models from training scripts (hyphenated filenames require importlib)
_spec = importlib.util.spec_from_file_location("_vqvae", Path(__file__).parent / "main_vq-vae.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
VQVAE = _mod.VQVAE

_spec2 = importlib.util.spec_from_file_location("_hypervq", Path(__file__).parent / "main_hypervq.py")
_mod2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
HyperVQ = _mod2.HyperVQ


@dataclass
class VizConfig:
    model: Literal["vqvae", "hypervq"] = "vqvae"
    checkpoint_path: str = "checkpoints/vqvae/epoch_0"
    num_samples: int = 10
    seed: int = 42

    # Must match training config
    latent_dim: int = 128
    num_codes: int = 512

    # HyperVQ-specific (ignored for vqvae)
    curvature: float = 1.0
    hyp_dtype: Literal["float32", "float64"] = "float64"


def _build_abstract_model(cfg: VizConfig) -> nnx.Module:
    if cfg.model == "vqvae":
        return nnx.eval_shape(lambda: VQVAE(num_codes=cfg.num_codes, latent_dim=cfg.latent_dim, rngs=nnx.Rngs(0)))
    elif cfg.model == "hypervq":
        return nnx.eval_shape(
            lambda: HyperVQ(
                num_codes=cfg.num_codes,
                latent_dim=cfg.latent_dim,
                c=cfg.curvature,
                hyp_dtype=cfg.hyp_dtype,
                rngs=nnx.Rngs(0),
            )
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model}. Expected 'vqvae' or 'hypervq'.")


if __name__ == "__main__":
    cfg = tyro.cli(VizConfig)

    # Create abstract model (no memory allocated), then restore checkpoint
    abstract_model = _build_abstract_model(cfg)
    graphdef, abstract_state = nnx.split(abstract_model)

    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(Path(cfg.checkpoint_path).resolve(), abstract_state)
    model = nnx.merge(graphdef, state)
    model.eval()

    # Load test set and sample
    dataset = load_dataset("mnist", split="test").with_format("jax")
    x_BHWC = jnp.array(dataset[: cfg.num_samples]["image"], dtype=jnp.float32)[..., None] / 255.0

    # Reconstruct (rngs needed for Dropout, though deterministic in eval mode)
    rngs = nnx.Rngs(default=cfg.seed)
    _, _, x_recon_BHWC = model(x_BHWC, rngs)

    # Plot: ground truth (top row), reconstruction (bottom row)
    fig, axes = plt.subplots(2, cfg.num_samples, figsize=(2 * cfg.num_samples, 4))
    for i in range(cfg.num_samples):
        axes[0, i].imshow(x_BHWC[i, :, :, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_recon_BHWC[i, :, :, 0], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Ground Truth", fontsize=12)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=12)

    plt.tight_layout()
    out_path = Path("reconstruction.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()
