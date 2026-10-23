import torch
import torch.nn as nn

from hipredict.diffusion.models.unet_components import ModalityEncoder

from .unet_components import (
    CrossAttentionBlock,
    DirectionEmbedding,
    Downsample,
    ResBlock,
    SinusoidalTimeEmbedding,
    Upsample,
)


class BidirectionalDDPM(nn.Module):
    """
    Bidirectional conditional DDPM
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 128,
        num_res_blocks: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        C = base_channels
        emb_dim = C * 4

        # Levels to use attention
        self._ATTN_LEVELS = {3, 4, 5}

        # Encoder resolutions at each level
        ch = [C // 2, C, C, C * 2, C * 2, C * 4]

        # Context resoltuions at deep level for attention
        ctx_ch = {3: C * 2, 4: C * 2, 5: C * 4}

        # Time embeddings
        self.time_emb = SinusoidalTimeEmbedding(C)
        # Direction embedding
        self.direction_emb = DirectionEmbedding(C)

        # Source encoder for conditioning
        self.red_source_encoder = ModalityEncoder(in_channels=1, base_channels=C)
        self.ir_source_encoder = ModalityEncoder(in_channels=1, base_channels=C)

        # Covert input channels to base channels
        self.stem = nn.Conv2d(in_channels, ch[0], 3, padding=1)

        # Encoders
        self.enc_res = nn.ModuleList()
        self.enc_attn = nn.ModuleDict()
        self.enc_down = nn.ModuleList()

        in_ch = ch[0]
        for level, out_ch in enumerate(ch):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, emb_dim, dropout=dropout))
                in_ch = out_ch

            self.enc_res.append(blocks)

            # Use attention at deep levels in encoders
            if level in self._ATTN_LEVELS:
                self.enc_attn[str(level)] = CrossAttentionBlock(out_ch, ctx_ch[level])

            # Use downsampling between levels in encoders
            if level < len(ch) - 1:
                self.enc_down.append(Downsample(out_ch))

        # Bottleneck
        self.bottleneck_res1 = ResBlock(ch[-1], ch[-1], emb_dim, dropout=dropout)
        self.bottleneck_attn = CrossAttentionBlock(ch[-1], ctx_ch[5])
        self.bottleneck_res2 = ResBlock(ch[-1], ch[-1], emb_dim, dropout=dropout)

        # Decoders
        dec_in_ch = [
            ch[5] + ch[5],
            ch[5] + ch[4],
            ch[4] + ch[3],
            ch[3] + ch[2],
            ch[2] + ch[1],
            ch[1] + ch[0],
        ]
        dec_out_ch = list(reversed(ch))

        self.dec_res = nn.ModuleList()
        self.dec_attn = nn.ModuleDict()
        self.dec_up = nn.ModuleList()

        for dec_idx in range(len(ch)):
            enc_level = len(ch) - 1 - dec_idx
            in_ch = dec_in_ch[dec_idx]
            out_ch = dec_out_ch[dec_idx]

            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, emb_dim, dropout=dropout))
                in_ch = out_ch

            self.dec_res.append(blocks)

            # Use attention at deep levels in decoders
            if enc_level in self._ATTN_LEVELS:
                self.dec_attn[str(dec_idx)] = CrossAttentionBlock(
                    out_ch, ctx_ch[enc_level]
                )

            # Use upsampling between levels in decoders
            if dec_idx < len(ch) - 1:
                self.dec_up.append(Upsample(out_ch))

        # Output head
        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch[0], out_channels, 3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,
        x_src: torch.Tensor,
        edge: torch.Tensor,
        direction: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for bidirectional DDPM U-Net. Predicting noise at time step t.

        Args:
            x_t: Noisy input image at time step t, shape (B, C_in, H, W).
            x_src: Source image for conditioning (e.g., RED), shape (B, 1, H, W).
            edge: Edge map for conditioning, shape (B, 1, H, W).
            direction: Direction tensor, shape (B,).
            t: Time step tensor, shape (B,).
        Returns:
            Predicted noise, shape (B, C_out, H, W).
        """
        # Concate noise with condition
        parts = [x_t, x_src] + ([edge] if edge is not None else [])
        z = torch.cat(parts, dim=1)

        # Time embedding
        emb = self.time_emb(t) + self.direction_emb(direction)

        # context for attention blocks
        ctx_ir = self.ir_source_encoder(x_src)
        ctx_red = self.red_source_encoder(x_src)

        def pick_ctx(scale_key: str) -> torch.Tensor:
            mask = (direction == 0)[:, None, None, None].float()
            return ctx_ir[scale_key] * mask + ctx_red[scale_key] * (1.0 - mask)

        # Encoder forward
        h = self.stem(z)

        # Record skip connections for decoder
        skips = []

        for level, blocks in enumerate(self.enc_res):
            # All ResBlocks in Encoders
            for res in blocks:
                h = res(h, emb)
            # Cross attention at deep levels
            if str(level) in self.enc_attn:
                h = self.enc_attn[str(level)](h, pick_ctx(str(level)))

            # Record skip connection for decoder
            skips.append(h)

            # Downsample between levels
            if level < len(self.enc_down):
                h = self.enc_down[level](h)

        # Bottleneck
        h = self.bottleneck_res1(h, emb)
        h = self.bottleneck_attn(h, pick_ctx("5"))
        h = self.bottleneck_res2(h, emb)

        # Decoder forward
        for dec_idx, blocks in enumerate(self.dec_res):
            enc_level = len(self.enc_res) - 1 - dec_idx
            # Concatenate skip connection from encoder
            h = torch.cat([h, skips[enc_level]], dim=1)
            # All ResBlocks in Decoders
            for res in blocks:
                h = res(h, emb)
            # Cross attention at deep levels
            if str(dec_idx) in self.dec_attn:
                h = self.dec_attn[str(dec_idx)](h, pick_ctx(str(enc_level)))
            # Upsample between levels
            if dec_idx < len(self.dec_up):
                h = self.dec_up[dec_idx](h)

        # Output head
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h
