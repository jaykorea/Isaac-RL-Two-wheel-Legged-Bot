# network/model.py
from __future__ import annotations
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn


class DepthPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, d_model: int, patch: int):
        super().__init__()
        self.patch = int(patch)
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=self.patch, stride=self.patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W)
        x = self.proj(x)                   # (B,D,H',W')
        x = x.flatten(2).transpose(1, 2)   # (B,L,D)
        return x


class CbfCrossAttnTransformer(nn.Module):
    def __init__(
        self,
        depth_hw: Tuple[int, int],
        d_model: int = 256,
        patch: int = 16,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        H, W = int(depth_hw[0]), int(depth_hw[1])
        if H % patch != 0 or W % patch != 0:
            raise ValueError(f"patch={patch} must divide depth H,W. got H={H}, W={W}")

        self.patch_embed = DepthPatchEmbed(in_ch=1, d_model=d_model, patch=patch)

        L = (H // patch) * (W // patch)
        self.pos = nn.Parameter(torch.zeros(1, L, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.ee_embed = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, depth: torch.Tensor, ee: torch.Tensor) -> torch.Tensor:
        tok = self.patch_embed(depth)  # (B,L,D)
        tok = tok + self.pos
        tok = self.encoder(tok)        # (B,L,D)

        q = self.ee_embed(ee).unsqueeze(1)              # (B,1,D)
        out, _ = self.cross_attn(q, tok, tok, need_weights=False)
        pred = self.head(out.squeeze(1))                # (B,1)
        return pred
    

# =====================================================================================================================

class PatchEmbed(nn.Module):
    """ (C, H, W) 이미지를 (L, D) 토큰으로 변환 + Positional Embedding """
    def __init__(self, in_ch: int, img_hw: Tuple[int, int], d_model: int, patch: int):
        super().__init__()
        H, W = img_hw
        if H % patch != 0 or W % patch != 0:
            raise ValueError(f"Image size {img_hw} not divisible by patch {patch}")
        
        self.patch = patch
        self.H_tokens = H // patch
        self.W_tokens = W // patch
        self.num_tokens = self.H_tokens * self.W_tokens
        
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)
        
        # Learnable Position Embedding for this specific image source
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)                  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, L, D)
        x = x + self.pos_embed            # Add Pos Embed
        return x


class MultiModalTransformer(nn.Module):
    def __init__(
        self,
        input_configs: Dict[str, Dict[str, Any]], # {"tv_cam_rgb": {"shape": (H,W), "ch": 3}, ...}
        d_model: int = 256,
        patch: int = 16,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 1. Build Embeddings for each input key
        self.embeds = nn.ModuleDict()
        self.input_keys = sorted(list(input_configs.keys())) # 고정된 순서 보장
        
        for key, cfg in input_configs.items():
            shape_hw = cfg["shape"] # (H, W)
            ch = cfg["ch"]          # 3(RGB) or 1(Depth)
            
            self.embeds[key] = PatchEmbed(
                in_ch=ch, 
                img_hw=shape_hw, 
                d_model=d_model, 
                patch=patch
            )

        # 2. Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # 3. EE Position Embedding & Fusion
        self.ee_embed = nn.Sequential(
            nn.Linear(7, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 4. Cross Attention (Query: EE, Key/Value: Image Tokens)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 5. Output Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, images: Dict[str, torch.Tensor], ee: torch.Tensor) -> torch.Tensor:
        """
        images: 딕셔너리 형태의 이미지 배치들
        ee: (B, 3) End Effector Position
        """
        tokens_list = []
        
        # 순서대로 Embedding 및 리스트 추가
        for key in self.input_keys:
            img_tensor = images[key] # (B, C, H, W)
            emb = self.embeds[key](img_tensor) # (B, L_k, D)
            tokens_list.append(emb)
            
        # 모든 이미지 토큰 연결 (Concatenate along sequence dim)
        # shape: (B, Total_L, D)
        all_tokens = torch.cat(tokens_list, dim=1)
        
        # Encoder Pass
        enc_out = self.encoder(all_tokens) # (B, Total_L, D)

        # Cross Attention
        q = self.ee_embed(ee).unsqueeze(1) # (B, 1, D)
        out, _ = self.cross_attn(query=q, key=enc_out, value=enc_out)
        
        # Prediction
        pred = self.head(out.squeeze(1))   # (B, 1)
        return pred