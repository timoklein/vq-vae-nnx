from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import grain
import jax
import jax.numpy as jnp

# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)  # Hyperbolic operations often require higher precision
import optax
import orbax.checkpoint as ocp
import tyro
from datasets import load_dataset
from flax import nnx
from hyperbolix.manifolds import Poincare
from hyperbolix.nn_layers import HypRegressionPoincarePP


@dataclass
class Config:
    seed: int = 23

    latent_dim: int = 128
    num_codes: int = 512

    batch_size: int = 128
    learning_rate: float = 0.01
    num_epochs: int = 1

    curvature: float = 0.1
    hyp_dtype: Literal["float32", "float64"] = "float64"


class HyperVQ(nnx.Module):
    """Hyperbolic VQ-VAE with MLR-based quantization.

    Compared to the paper, this model does not use softmax temperature annealing because it was unstable in practice.
    """

    def __init__(
        self,
        num_codes: int,
        latent_dim: int,
        c: float,
        hyp_dtype: Literal["float32", "float64"],
        rngs: nnx.Rngs,
    ) -> None:
        # Encoder
        self.conv1 = nnx.Conv(in_features=1, out_features=32, kernel_size=(3, 3), rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=32, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.025, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_features=64, rngs=rngs)
        self.conv3 = nnx.Conv(in_features=64, out_features=latent_dim, kernel_size=(3, 3), rngs=rngs)

        # Function for avg pooling
        self.avgpool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))

        # Hyperbolic quantization
        self.num_codes = num_codes
        self.curvature = c
        hyp_dtype = jnp.float64 if hyp_dtype == "float64" else jnp.float32
        self.manifold = Poincare(hyp_dtype)
        self.mlr = HypRegressionPoincarePP(manifold_module=self.manifold, in_dim=latent_dim, out_dim=num_codes, rngs=rngs)

        # Decoder
        self.deconv1 = nnx.ConvTranspose(in_features=latent_dim, out_features=64, kernel_size=(3, 3), rngs=rngs)
        self.bn3 = nnx.BatchNorm(num_features=64, rngs=rngs)
        self.deconv2 = nnx.ConvTranspose(in_features=64, out_features=32, kernel_size=(3, 3), strides=(2, 2), rngs=rngs)
        self.bn4 = nnx.BatchNorm(num_features=32, rngs=rngs)
        self.deconv3 = nnx.ConvTranspose(in_features=32, out_features=1, kernel_size=(3, 3), strides=(2, 2), rngs=rngs)

    def encode(self, x_BHWC: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        x_BHWC = self.avgpool(nnx.relu(self.bn1(self.dropout1(self.conv1(x_BHWC), rngs=rngs))))
        x_BHWC = self.avgpool(nnx.relu(self.bn2(self.conv2(x_BHWC))))
        x_BHWC = self.conv3(x_BHWC)

        return x_BHWC

    def decode(self, z: jax.Array) -> jax.Array:
        z = nnx.relu(self.bn3(self.deconv1(z)))
        z = nnx.relu(self.bn4(self.deconv2(z)))
        z = nnx.sigmoid(self.deconv3(z))  # Output in [0, 1]
        return z

    def quantize(self, x_BHWC: jax.Array, rngs: nnx.Rngs) -> jax.Array:

        b, h, w, c = x_BHWC.shape
        # Sanity check. Must use checkify to be jittable.
        assert c == self.mlr.kernel[...].shape[1], "Latent dimension mismatch between encoder and embedding."
        z_ND = x_BHWC.reshape(b * h * w, c)

        # Step 1: Apply MLR (logits = unidirectional_mlr(zh) (Eq. (6)))
        hyp_z_ND = self.manifold.expmap_0(z_ND, self.curvature)  # Map to manifold
        scores_ND = self.mlr(hyp_z_ND)

        # Step 2: Gumbel-Softmax with straight-through estimator (refs [4],[14])
        # Soft probabilities (differentiable)
        noised_scores_NK = scores_ND + jax.random.gumbel(rngs(), scores_ND.shape)
        soft_weights_NK = jax.nn.softmax(noised_scores_NK, axis=-1)
        # Hard selection (non-differentiable)
        k_idx_N = jnp.argmax(soft_weights_NK, axis=-1)
        hard_weights_NK = jax.nn.one_hot(k_idx_N, self.num_codes, dtype=soft_weights_NK.dtype)
        # STE: forward uses hard (discrete), backward uses soft (differentiable)
        st_weights_NK = soft_weights_NK + jax.lax.stop_gradient(hard_weights_NK - soft_weights_NK)

        # Step 3: Compute codebook vectors zq = rk * ak (Eq. 9)
        # NOTE: The matmul trick here is crucial to make the operation differentiable w.r.t. the codebook parameters.
        all_codes_KD = self.mlr.bias[...] * self.mlr.kernel[...]  # (K, D)
        codes_ND = st_weights_NK @ all_codes_KD  # (N, D) — gradients flow to θ_Q

        z_q_BHWC = codes_ND.reshape(b, h, w, c)
        return z_q_BHWC

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> tuple[jax.Array, jax.Array, jax.Array]:
        z_e = self.encode(x, rngs)
        # Quantization step would go here
        z_q = self.quantize(z_e, rngs)
        x_recon = self.decode(z_q)
        return z_e, z_q, x_recon


def batch_to_jax(batch: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert a numpy batch dict to (images_BHW, labels_B) JAX arrays."""
    images_BHWC = jnp.array(batch["image"], dtype=jnp.float32) / 255.0  # Normalize to [0, 1]
    labels_B = jnp.array(batch["label"], dtype=jnp.int32)
    return images_BHWC[..., None], labels_B


def loss_fn(model: HyperVQ, x: jax.Array, rngs: nnx.Rngs) -> tuple[jax.Array, dict[str, jax.Array]]:
    *_, x_recon = model(x, rngs)

    # MSE pixel reconstruction
    recon_loss = jnp.mean((x - x_recon) ** 2)

    return recon_loss, {"recon_loss": recon_loss}


def train_step(model: HyperVQ, x: jax.Array, optimizer: nnx.Optimizer, rngs: nnx.Rngs, metrics: nnx.Metric) -> None:
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, info), grads = grad_fn(model, x, rngs)
    metrics.update(loss=loss, recon_loss=info["recon_loss"])
    optimizer.update(model, grads)


def eval_step(model: HyperVQ, x: jax.Array, rngs: nnx.Rngs, metrics: nnx.Metric) -> None:
    loss, info = loss_fn(model, x, rngs)
    metrics.update(loss=loss, recon_loss=info["recon_loss"])


if __name__ == "__main__":
    cfg = tyro.cli(Config)

    rngs = nnx.Rngs(default=cfg.seed)

    # https://huggingface.co/docs/datasets/en/use_with_jax
    dataset = load_dataset("mnist").with_format("jax")
    dataloader = grain.MapDataset.source(dataset["train"])
    dataloader = dataloader.seed(cfg.seed).shuffle().batch(cfg.batch_size, drop_remainder=True)

    model = HyperVQ(
        num_codes=cfg.num_codes,
        latent_dim=cfg.latent_dim,
        c=cfg.curvature,
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=cfg.learning_rate), wrt=nnx.Param)

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        recon_loss=nnx.metrics.Average("recon_loss"),
    )

    train_step_jit = nnx.jit(train_step, donate_argnames=("model", "optimizer"))

    ckpt_dir = Path("checkpoints/hypervq").resolve()
    checkpointer = ocp.StandardCheckpointer()

    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(dataloader):
            model.train()
            x_BHWC, y_B = batch_to_jax(batch)

            train_step_jit(model, x_BHWC, optimizer, rngs, metrics)

            computed = metrics.compute()
            print(f"Epoch {epoch}, Step {step}, Loss: {computed['loss']:.4f}, Recon: {computed['recon_loss']:.4f}")

        _, state = nnx.split(model)
        checkpointer.save(ckpt_dir / f"epoch_{epoch}", state, force=True)
        checkpointer.wait_until_finished()
        print(f"Checkpoint saved to {ckpt_dir / f'epoch_{epoch}'}")
