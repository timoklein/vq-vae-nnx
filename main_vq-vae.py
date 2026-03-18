from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_disable_jit", True)
import optax
import orbax.checkpoint as ocp
import tyro
from datasets import load_dataset
from flax import nnx


@dataclass
class Config:
    seed: int = 23

    latent_dim: int = 128
    num_codes: int = 512
    commitment_cost: float = 0.25

    batch_size: int = 128
    learning_rate: float = 0.01
    num_epochs: int = 1


class VQVAE(nnx.Module):
    def __init__(self, num_codes: int, latent_dim: int, rngs: nnx.Rngs) -> None:
        # Encoder
        self.conv1 = nnx.Conv(in_features=1, out_features=32, kernel_size=(3, 3), rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_features=32, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.025, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_features=64, rngs=rngs)
        self.conv3 = nnx.Conv(in_features=64, out_features=latent_dim, kernel_size=(3, 3), rngs=rngs)

        # Function for avg pooling
        self.avgpool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))

        self.embedding = nnx.Embed(num_embeddings=num_codes, features=latent_dim, rngs=rngs)
        initializer = nnx.initializers.uniform(2 / num_codes)
        self.embedding.embedding.value = initializer(rngs(), (num_codes, latent_dim), jnp.float32) - 1 / num_codes

        # Decoder
        self.deconv1 = nnx.ConvTranspose(in_features=latent_dim, out_features=64, kernel_size=(3, 3), rngs=rngs)
        self.bn3 = nnx.BatchNorm(num_features=64, rngs=rngs)
        self.deconv2 = nnx.ConvTranspose(in_features=64, out_features=32, kernel_size=(3, 3), strides=(2, 2), rngs=rngs)
        self.bn4 = nnx.BatchNorm(num_features=32, rngs=rngs)
        self.deconv3 = nnx.ConvTranspose(in_features=32, out_features=1, kernel_size=(3, 3), strides=(2, 2), rngs=rngs)

    def encode(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        x = self.avgpool(nnx.relu(self.bn1(self.dropout1(self.conv1(x), rngs=rngs))))
        x = self.avgpool(nnx.relu(self.bn2(self.conv2(x))))
        x = self.conv3(x)
        return x

    def decode(self, z: jax.Array) -> jax.Array:
        z = nnx.relu(self.bn3(self.deconv1(z)))
        z = nnx.relu(self.bn4(self.deconv2(z)))
        z = nnx.sigmoid(self.deconv3(z))  # Output in [0, 1]
        return z

    def quantize(self, z_e: jax.Array) -> jax.Array:
        b, h, w, c = z_e.shape
        # Sanity check. Must use checkify to be jittable.
        assert c == self.embedding.embedding.value.shape[1], "Latent dimension mismatch"
        z_flattened = z_e.reshape(b * h * w, c)

        def calculate_distances(z: jax.Array, codes: jax.Array) -> jax.Array:
            z_norm = jnp.sum(z**2)
            c_norm = jnp.sum(codes**2, axis=-1)
            z_c_prod = z @ codes.T
            return (z_norm + c_norm - 2 * z_c_prod).squeeze()

        distances = jax.vmap(calculate_distances, in_axes=(0, None))(z_flattened, self.embedding.embedding.value)

        indices = jnp.argmin(distances, axis=-1)

        z_q = self.embedding(indices).reshape(b, h, w, c)

        return z_q

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> tuple[jax.Array, jax.Array, jax.Array]:
        z_e = self.encode(x, rngs)
        # Quantization step would go here
        z_q = self.quantize(z_e)
        x_recon = self.decode(z_q)
        return z_e, z_q, x_recon


def loss_fn(model: VQVAE, x: jax.Array, commitment_cost: float, rngs: nnx.Rngs) -> tuple[jax.Array, dict[str, jax.Array]]:
    z_e, z_q, x_recon = model(x, rngs)

    # MSE pixel reconstruction
    recon_loss = jnp.mean((x - x_recon) ** 2)
    # Commitment loss
    commitment_loss = jnp.mean((z_q - jax.lax.stop_gradient(z_e)) ** 2) + commitment_cost * jnp.mean(
        (jax.lax.stop_gradient(z_q) - z_e) ** 2
    )

    return recon_loss + commitment_loss, {"recon_loss": recon_loss, "commitment_loss": commitment_loss}


def train_step(
    model: VQVAE, x: jax.Array, optimizer: nnx.Optimizer, commitment_cost: float, rngs: nnx.Rngs, metrics: nnx.Metric
) -> None:
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, info), grads = grad_fn(model, x, commitment_cost, rngs)
    metrics.update(loss=loss, recon_loss=info["recon_loss"], commitment_loss=info["commitment_loss"])
    optimizer.update(model, grads)


def eval_step(model: VQVAE, x: jax.Array, commitment_cost: float, rngs: nnx.Rngs, metrics: nnx.Metric) -> None:
    loss, info = loss_fn(model, x, commitment_cost, rngs)
    metrics.update(loss=loss, recon_loss=info["recon_loss"], commitment_loss=info["commitment_loss"])


if __name__ == "__main__":
    cfg = tyro.cli(Config)

    rngs = nnx.Rngs(default=cfg.seed)

    # https://huggingface.co/docs/datasets/en/use_with_jax
    dataset = load_dataset("mnist").with_format("jax")
    # normalized_dataset = dataset.map(lambda batch: {"image": batch["image"] / 255.0})

    model = VQVAE(num_codes=cfg.num_codes, latent_dim=cfg.latent_dim, rngs=rngs)

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=cfg.learning_rate), wrt=nnx.Param)

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        recon_loss=nnx.metrics.Average("recon_loss"),
        commitment_loss=nnx.metrics.Average("commitment_loss"),
    )

    train_step_jit = nnx.jit(train_step, static_argnames=("commitment_cost",), donate_argnames=("model", "optimizer"))

    ckpt_dir = Path("checkpoints/vqvae").resolve()
    checkpointer = ocp.StandardCheckpointer()

    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(dataset["train"].iter(batch_size=cfg.batch_size)):
            model.train()
            x = batch["image"][..., None]
            # Normalize here and not in forward to prevent from computing the loss against unnormalized data
            x = x.astype(jnp.float32) / 255.0  # Normalize to [0, 1]

            train_step_jit(model, x, optimizer, cfg.commitment_cost, rngs, metrics)

            computed = metrics.compute()
            print(
                f"Epoch {epoch}, Step {step}, Loss: {computed['loss']:.4f}, "
                f"Recon: {computed['recon_loss']:.4f}, Commit: {computed['commitment_loss']:.4f}"
            )

        _, state = nnx.split(model)
        checkpointer.save(ckpt_dir / f"epoch_{epoch}", state, force=True)
        checkpointer.wait_until_finished()
        print(f"Checkpoint saved to {ckpt_dir / f'epoch_{epoch}'}")
