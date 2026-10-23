import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Time Embedding
class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for the diffusion timestep t
    """

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.base_channels = base_channels
        emb_dim = base_channels * 4
        self.mlp = nn.Sequential(
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.base_channels // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


# Direction Embedding
class DirectionEmbedding(nn.Module):
    """
    Explicit translation direction signal.
    """

    def __init__(self, base_channels: int, num_directions: int = 2) -> None:
        super().__init__()
        emb_dim = base_channels * 4
        self.table = nn.Embedding(num_directions, base_channels)
        self.mlp = nn.Sequential(
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, direction: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.table(direction))


# Residual Block
class ResBlock(nn.Module):
    """
    Residual block with Adaptive Group Normalization (AdaGN) conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        num_groups: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # First conv path
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # AdaGN embedding projection: emb → scale + shift (2 × out_channels)
        # Leading SiLU activates the shared embedding before the per-block proj.
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * out_channels),
        )

        # Second conv path — norm comes before AdaGN modulation
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection: 1×1 conv if channel width changes, else identity
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Preserves height and width. Allow be used for both encoder and decoder blocks.
        Args:
            x: [B, C_in, H, W] input feature map
            emb: [B, emb_dim]  combined time + direction embedding
        Returns:
            feature: [B, C_out, H, W] output feature map
        """
        h = self.conv1(self.act1(self.norm1(x)))

        # AdaGN: split into scale and shift, broadcast over spatial dims
        emb_out = self.emb_proj(emb)  # [B, 2*out_channels]
        scale, shift = emb_out.chunk(2, dim=1)  # each [B, out_channels]
        scale = scale[:, :, None, None]  # [B, C, 1, 1]
        shift = shift[:, :, None, None]

        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(self.act2(h)))

        return h + self.skip(x)


class ConvResBlock(nn.Module):
    """
    Plain residual block without time/direction conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


# Attention Blocks
class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention with residual connection.
    """

    def __init__(self, channels: int, head_dim: int = 64) -> None:
        super().__init__()
        self.num_heads = max(1, channels // head_dim)
        self.head_dim = channels // self.num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)  # [B, heads, d, HW]

        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)  # [B, heads, d, HW]

        return x + self.proj(out.reshape(B, C, H, W))


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        context_channels: int,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_heads = max(1, channels // head_dim)
        self.head_dim = channels // self.num_heads

        # Separate GroupNorms for the query (U-Net) and context (encoder) streams
        self.norm_q = nn.GroupNorm(32, channels)
        self.norm_kv = nn.GroupNorm(32, context_channels)

        # Q projection (from main U-Net), K/V projection (from modality encoder)
        self.conv_q = nn.Conv2d(channels, channels, 1)
        self.conv_kv = nn.Conv2d(context_channels, channels * 2, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x:       [B, C,     H, W]  main U-Net feature (noisy target)
        # context: [B, C_ctx, H, W]  modality encoder feature (source image)
        B, C, H, W = x.shape

        q = self.conv_q(self.norm_q(x))  # [B, C, H, W]
        k, v = self.conv_kv(self.norm_kv(context)).chunk(2, dim=1)  # [B, C, H, W] each

        # Reshape for multi-head attention over flattened spatial positions
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)  # [B, heads, d, HW]

        return x + self.proj(out.reshape(B, C, H, W))


# Downsample / Upsample
class Downsample(nn.Module):
    """Strided 3×3 conv (stride=2) to halve spatial resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour 2× interpolation followed by a 3×3 conv to refine."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# Modality Encoder
class ModalityEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 128) -> None:
        super().__init__()
        C = base_channels

        # Stem: 256×256, C//2 channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C // 2, 3, padding=1),
            nn.SiLU(),
        )

        # Progressive downsampling with ConvResBlocks (Extraction Blocks)
        # 256→128: C//2 → C
        self.stage1 = nn.Sequential(
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1),
            ConvResBlock(C, C),
        )
        # 128→64: C → C
        self.stage2 = nn.Sequential(
            nn.Conv2d(C, C, 3, stride=2, padding=1),
            ConvResBlock(C, C),
        )
        # 64→32: C → 2C
        # Used for cross attention at 32×32 resolution
        self.stage3 = nn.Sequential(
            nn.Conv2d(C, C * 2, 3, stride=2, padding=1),
            ConvResBlock(C * 2, C * 2),
            ConvResBlock(C * 2, C * 2),
            ConvResBlock(C * 2, C * 2),
        )
        # 32→16: 2C → 2C
        # Used for cross attention at 16×16 resolution
        self.stage4 = nn.Sequential(
            nn.Conv2d(C * 2, C * 2, 3, stride=2, padding=1),
            ConvResBlock(C * 2, C * 2),
            ConvResBlock(C * 2, C * 2),
            ConvResBlock(C * 2, C * 2),
        )
        # 16→8: 2C → 4C
        # Used for cross attention at 8×8 resolution
        self.stage5 = nn.Sequential(
            nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1),
            ConvResBlock(C * 4, C * 4),
            ConvResBlock(C * 4, C * 4),
            ConvResBlock(C * 4, C * 4),
        )

    def forward(self, x: torch.Tensor) -> dict:
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        level3_h = self.stage3(h)
        level4_h = self.stage4(level3_h)
        level5_h = self.stage5(level4_h)
        return {"3": level3_h, "4": level4_h, "5": level5_h}
