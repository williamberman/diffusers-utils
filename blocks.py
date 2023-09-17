import math

import torch
import torch.nn.functional as F
import xformers
from torch import nn


def get_sinusoidal_embedding(
    indices: torch.Tensor,
    embedding_dim: int,
):
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=indices.device)
    exponent = exponent / half_dim

    emb = torch.exp(exponent)
    emb = indices.unsqueeze(-1).float() * emb
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

    return emb


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()

        self.time_emb_proj = nn.Linear(time_embedding_dim, out_channels)

        self.norm1 = torch.nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.nonlinearity = nn.SiLU()

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None

    def forward(self, hidden_states, temb):
        residual = hidden_states

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        hidden_states = hidden_states + residual

        return hidden_states


class Transformer2DModel(nn.Module):
    def __init__(self, channels, encoder_hidden_states_dim, num_transformer_blocks):
        super().__init__()

        self.norm = nn.GroupNorm(32, channels, eps=1e-06)
        self.proj_in = nn.Linear(channels, channels)

        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(channels, encoder_hidden_states_dim) for _ in range(num_transformer_blocks)])

        self.proj_out = nn.Linear(channels, channels)

    def forward(self, hidden_states, encoder_hidden_states):
        batch_size, channels, height, width = hidden_states.shape

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()

        hidden_states = hidden_states + residual

        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(self, channels, encoder_hidden_states_dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn1 = Attention(channels, channels)

        self.norm2 = nn.LayerNorm(channels)
        self.attn2 = Attention(channels, encoder_hidden_states_dim)

        self.norm3 = nn.LayerNorm(channels)
        self.ff = nn.ModuleDict(dict(net=nn.Sequential(GEGLU(channels, 4 * channels), nn.Dropout(0.0), nn.Linear(4 * channels, channels))))

    def forward(self, hidden_states, encoder_hidden_states):
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states

        hidden_states = self.attn2(self.norm2(hidden_states), encoder_hidden_states) + hidden_states

        hidden_states = self.ff["net"](self.norm3(hidden_states)) + hidden_states

        return hidden_states


class Attention(nn.Module):
    def __init__(self, channels, encoder_hidden_states_dim):
        super().__init__()
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(encoder_hidden_states_dim, channels, bias=False)
        self.to_v = nn.Linear(encoder_hidden_states_dim, channels, bias=False)
        self.to_out = nn.Sequential(nn.Linear(channels, channels), nn.Dropout(0.0))

    def forward(self, hidden_states, encoder_hidden_states=None):
        batch_size, q_seq_len, channels = hidden_states.shape
        head_dim = 64

        if encoder_hidden_states is not None:
            kv = encoder_hidden_states
        else:
            kv = hidden_states

        kv_seq_len = kv.shape[1]

        query = self.to_q(hidden_states)
        key = self.to_k(kv)
        value = self.to_v(kv)

        query = query.reshape(batch_size, q_seq_len, channels // head_dim, head_dim).contiguous()
        key = key.reshape(batch_size, kv_seq_len, channels // head_dim, head_dim).contiguous()
        value = value.reshape(batch_size, kv_seq_len, channels // head_dim, head_dim).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(query, key, value)

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = hidden_states.reshape(batch_size, q_seq_len, channels).contiguous()

        hidden_states = self.to_out(hidden_states)

        return hidden_states


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)
