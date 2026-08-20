from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class VisualEncoder(nn.Module):
    """
    Маленький encoder для двух LIBERO-камер.

    На входе:
        [B, N_CAMERAS, 3, H, W]

    На выходе:
        [B, latent_dim]
    """

    def __init__(
        self,
        num_cameras: int,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()

        self.num_cameras = num_cameras

        self.encoder = nn.Sequential(
            nn.Conv2d(3 * num_cameras, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),

            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.GELU(),

            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.projection = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError(
                f"Expected images [B, N, C, H, W], got {tuple(images.shape)}"
            )

        batch_size, num_cameras, channels, height, width = images.shape

        if num_cameras != self.num_cameras:
            raise ValueError(
                f"Expected {self.num_cameras} cameras, got {num_cameras}"
            )

        if channels != 3:
            raise ValueError(f"Expected RGB images, got {channels} channels")

        x = images.reshape(
            batch_size,
            num_cameras * channels,
            height,
            width,
        )

        # LeRobot возвращает RGB в диапазоне [0, 1].
        x = x.float().clamp(0.0, 1.0)
        x = (x - 0.5) / 0.5

        return self.projection(self.encoder(x))


def encode_language(
    base_policy: nn.Module,
    batch: Mapping[str, Tensor],
) -> Tensor:
    """
    Получает frozen language embedding из SmolVLA.

    Градиенты через языковую часть базовой политики не проходят.
    """
    tokens = batch[OBS_LANGUAGE_TOKENS]
    attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]

    with torch.no_grad():
        language_embeddings = (
            base_policy.model.vlm_with_expert
            .embed_language_tokens(tokens)
            .float()
        )

    mask = attention_mask.bool().unsqueeze(-1)
    masked = language_embeddings * mask
    denominator = mask.sum(dim=1).clamp_min(1)

    return masked.sum(dim=1) / denominator


class JepaAdapter(nn.Module):
    """
    Action-conditioned JEPA + residual action adapter.

    JEPA часть:
        current image + language + state + base action chunk + goal prototype
            -> predicted future latent

    Policy часть:
        current image + language + state + predicted future latent + prototype
            -> residual action chunk
    """

    def __init__(
        self,
        *,
        num_cameras: int,
        language_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        ema_decay: float = 0.995,
    ) -> None:
        super().__init__()

        self.num_cameras = num_cameras
        self.language_dim = language_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        self.encoder = VisualEncoder(
            num_cameras=num_cameras,
            latent_dim=latent_dim,
        )

        self.target_encoder = VisualEncoder(
            num_cameras=num_cameras,
            latent_dim=latent_dim,
        )

        self.target_encoder.load_state_dict(self.encoder.state_dict())

        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

        self.language_projection = nn.Sequential(
            nn.Linear(language_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        self.state_projection = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        self.action_projection = nn.Sequential(
            nn.Linear(action_dim * chunk_size, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        context_dim = latent_dim * 5

        self.predictor = make_mlp(
            input_dim=context_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
        )

        action_context_dim = latent_dim * 5

        self.action_head = nn.Sequential(
            nn.Linear(action_context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    @torch.no_grad()
    def update_ema(self) -> None:
        for online, target in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters(),
            strict=True,
        ):
            target.data.mul_(self.ema_decay)
            target.data.add_(
                online.data,
                alpha=1.0 - self.ema_decay,
            )

    def _context(
        self,
        *,
        current_latent: Tensor,
        language_latent: Tensor,
        state_latent: Tensor,
        base_action: Tensor,
        goal_prototype: Tensor,
    ) -> Tensor:
        base_action = base_action[:, : self.chunk_size, : self.action_dim]
        base_action = base_action.reshape(base_action.shape[0], -1)
        action_latent = self.action_projection(base_action)

        return torch.cat(
            [
                current_latent,
                language_latent,
                state_latent,
                action_latent,
                goal_prototype,
            ],
            dim=-1,
        )

    def act(
        self,
        *,
        current_images: Tensor,
        language_embedding: Tensor,
        state: Tensor,
        base_action: Tensor,
        goal_prototype: Tensor,
    ) -> Tensor:
        current_latent = self.encoder(current_images)

        language_latent = self.language_projection(
            language_embedding.float()
        )

        state_latent = self.state_projection(state.float())

        context = self._context(
            current_latent=current_latent,
            language_latent=language_latent,
            state_latent=state_latent,
            base_action=base_action,
            goal_prototype=goal_prototype,
        )

        predicted_future_latent = self.predictor(context)

        action_context = torch.cat(
            [
                current_latent,
                language_latent,
                state_latent,
                predicted_future_latent,
                goal_prototype,
            ],
            dim=-1,
        )

        residual = self.action_head(action_context)
        residual = residual.reshape(
            current_images.shape[0],
            self.chunk_size,
            self.action_dim,
        )

        return base_action[:, : self.chunk_size, : self.action_dim] + residual

    def forward(
        self,
        *,
        current_images: Tensor,
        future_images: Tensor,
        language_embedding: Tensor,
        state: Tensor,
        base_action: Tensor,
        target_action: Tensor,
        goal_prototype: Tensor,
        action_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        current_latent = self.encoder(current_images)

        with torch.no_grad():
            future_latent = self.target_encoder(future_images)

        language_latent = self.language_projection(
            language_embedding.float()
        )

        state_latent = self.state_projection(state.float())

        context = self._context(
            current_latent=current_latent,
            language_latent=language_latent,
            state_latent=state_latent,
            base_action=base_action,
            goal_prototype=goal_prototype,
        )

        predicted_future_latent = self.predictor(context)

        action_context = torch.cat(
            [
                current_latent,
                language_latent,
                state_latent,
                predicted_future_latent,
                goal_prototype,
            ],
            dim=-1,
        )

        residual = self.action_head(action_context)
        residual = residual.reshape(
            current_images.shape[0],
            self.chunk_size,
            self.action_dim,
        )

        predicted_action = (
            base_action[:, : self.chunk_size, : self.action_dim]
            + residual
        )

        jepa_loss = F.smooth_l1_loss(
            predicted_future_latent,
            future_latent.detach(),
        )

        if action_mask is None:
            action_loss = F.mse_loss(
                predicted_action,
                target_action[:, : self.chunk_size, : self.action_dim],
            )
        else:
            target_action = target_action[:, : self.chunk_size, : self.action_dim]
            mask = action_mask[:, : self.chunk_size].float()
            mask = mask.unsqueeze(-1)

            squared_error = (predicted_action - target_action).square()
            action_loss = (
                squared_error * mask
            ).sum() / mask.sum().clamp_min(1.0)

        total_loss = action_loss + jepa_loss

        return {
            "loss": total_loss,
            "jepa_loss": jepa_loss.detach(),
            "action_loss": action_loss.detach(),
            "predicted_action": predicted_action,
        }