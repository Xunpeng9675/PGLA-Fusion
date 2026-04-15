# model/PSFusion_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

# --------------------------
# Helper / Norm / Utils
# --------------------------
def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")

def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# --------------------------
# Ghost Module (light conv)
# --------------------------
class GhostModule(nn.Module):
    """A simple GhostModule implementation"""
    def __init__(self, inp, out, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_channels = out
        init_channels = int((out + ratio - 1) / ratio)
        new_channels = out - init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, stride=1, padding=dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.out_channels, :, :]

class GhostBottleneck(nn.Module):
    """Ghost bottleneck similar to GhostNet: cheap expansion -> depthwise -> projection"""
    def __init__(self, in_ch, mid_ch, out_ch, kernel_size=3, stride=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        self.ghost1 = GhostModule(in_ch, mid_ch, kernel_size=1, relu=True)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride=stride, padding=kernel_size//2, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
        ) if stride == 1 else nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride=stride, padding=kernel_size//2, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
        )
        self.ghost2 = GhostModule(mid_ch, out_ch, kernel_size=1, relu=False)
        self.use_shortcut = (in_ch == out_ch and stride == 1)
    def forward(self, x):
        res = x
        x = self.ghost1(x)
        x = self.dw_conv(x)
        x = self.ghost2(x)
        if self.use_shortcut:
            return x + res
        else:
            return x

# ====== 将你的原有 LocalWindowAttention / EfficientAttention 整段替换为下面实现 ======
import torch.nn.functional as F
class LocalWindowAttention(nn.Module):
    """
    Local window attention with automatic padding.
    - 将输入 pad 到能被 window_size 整除（右/下方向填充），
      在每个 window 内做多头注意力，最后裁回原始 H,W。
    - 使用反射填充（reflect）以减小边界伪影。
    """
    def __init__(self, dim, num_heads=4, window_size=8):
        super(LocalWindowAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        ws = self.window_size

        # --- pad if needed (pad right and bottom) ---
        pad_h = (ws - (H % ws)) % ws
        pad_w = (ws - (W % ws)) % ws
        if pad_h != 0 or pad_w != 0:
            # F.pad accepts pad = (pad_left, pad_right, pad_top, pad_bottom)
            # we do pad_right = pad_w, pad_bottom = pad_h
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        Hp, Wp = x.shape[2], x.shape[3]  # padded sizes
        num_h = Hp // ws
        num_w = Wp // ws
        num_windows = num_h * num_w

        # project qkv
        q = self.q_proj(x)  # B, C, Hp, Wp
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape to (B, num_heads, head_dim, num_h, ws, num_w, ws)
        # then permute to (B, num_h, num_w, num_heads, head_dim, ws, ws)
        q = q.view(B, self.num_heads, self.head_dim, num_h, ws, num_w, ws)
        q = q.permute(0, 3, 5, 1, 2, 4, 6).contiguous()  # B, num_h, num_w, head, head_dim, ws, ws
        q = q.view(B * num_windows, self.num_heads, self.head_dim, ws * ws)  # (B*num_windows, head, head_dim, S)

        k = k.view(B, self.num_heads, self.head_dim, num_h, ws, num_w, ws)
        k = k.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        k = k.view(B * num_windows, self.num_heads, self.head_dim, ws * ws)

        v = v.view(B, self.num_heads, self.head_dim, num_h, ws, num_w, ws)
        v = v.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        v = v.view(B * num_windows, self.num_heads, self.head_dim, ws * ws)

        # attention within each window: attn shape (B*num_windows, head, S, S)
        attn = torch.einsum("b h c i, b h c j -> b h i j", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        # attn @ v -> (B*num_windows, head, head_dim, S)
        out = torch.einsum("b h i j, b h c j -> b h c i", attn, v)

        # reshape back to (B, C, Hp, Wp)
        out = out.view(B, num_h, num_w, self.num_heads, self.head_dim, ws, ws)
        out = out.permute(0, 3, 4, 1, 5, 2, 6).contiguous()  # (B, head, head_dim, num_h, ws, num_w, ws)
        out = out.view(B, self.num_heads * self.head_dim, Hp, Wp)  # (B, C, Hp, Wp)
        out = self.out_proj(out)

        # remove padding if any
        if pad_h != 0 or pad_w != 0:
            out = out[:, :, :H, :W]

        return out
# ====== 替换结束 ======



# --------------------------
# FeedForward (light)
# --------------------------
class LightFFN(nn.Module):
    def __init__(self, dim, expansion=2, bias=False):
        super(LightFFN, self).__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=bias)
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.project_out(x)
        return x

# --------------------------
# Light Block (Norm + EfficientAttention + FFN)
# --------------------------
class LightBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion=2.0, layer_norm_type="WithBias", downsample_kv=2):
        super(LightBlock, self).__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = LocalWindowAttention(dim, num_heads=num_heads, window_size=8)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = LightFFN(dim, expansion=ffn_expansion)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# --------------------------
# BaseSemanticEncoder (light variant using Ghost + EfficientAttention)
# --------------------------
class SemanticFeatureEncoder(nn.Module):
    def __init__(self, dim=64, num_heads=4, ffn_expansion_factor=2.0, qkv_bias=False):
        super(SemanticFeatureEncoder, self).__init__()
        # initial lightweight distill via ghost
        self.distill_conv = GhostModule(dim, dim, kernel_size=1, ratio=2)
        # efficient attention stack (two blocks)
        self.attn_block1 = LightBlock(dim, num_heads=num_heads, ffn_expansion=ffn_expansion_factor)
        self.attn_block2 = LightBlock(dim, num_heads=num_heads, ffn_expansion=ffn_expansion_factor)
        # high-frequency compensation using depthwise separable conv
        self.hf_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        # channel gating (light SE)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(dim, max(dim // 8, 4), kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // 8, 4), dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        x0 = x
        x = self.distill_conv(x)
        x = self.attn_block1(x)
        x = self.attn_block2(x)
        hf = x0 - F.avg_pool2d(x0, kernel_size=3, stride=1, padding=1)
        hf = self.hf_conv(hf)
        gate = self.se(self.pool(x))
        x = x + hf
        x = x * gate
        return x

# --------------------------
# ProgressiveDetailEncoder (light)
# --------------------------
class ProgressiveTextureEncoder(nn.Module):
    def __init__(self, num_layers=2, dim=64):
        super(ProgressiveTextureEncoder, self).__init__()
        layers = []
        # use GhostBottleneck / inverted residual style lightweight blocks
        for i in range(num_layers):
            layers.append(
                nn.Sequential(
                    GhostBottleneck(dim, mid_ch=max(dim//2, 16), out_ch=dim, kernel_size=3, stride=1),
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True)
                )
            )
        self.layers = nn.ModuleList(layers)
        # cross-layer fusion compressed
        self.fusion = nn.Conv2d(dim * num_layers, dim, kernel_size=1, bias=False)
    def forward(self, x):
        out_list = []
        xi = x
        for layer in self.layers:
            xi = layer(xi)
            out_list.append(xi)
        out = torch.cat(out_list, dim=1)
        out = self.fusion(out)
        return out

# --------------------------
# OverlapPatchEmbed (kept simple)
# --------------------------
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=1, embed_dim=64, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
    def forward(self, x):
        return self.proj(x)

# --------------------------
# DualBranchContextEncoder (light)
# --------------------------
class DualStreamContextEncoder(nn.Module):
    def __init__(self, inp_channels=1, out_channels=1, dim=64, num_blocks=2, heads=4, ffn_expansion_factor=2, bias=False, LayerNorm_type="WithBias"):
        super(DualStreamContextEncoder, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        # shallow shared blocks (few LightBlock to capture context)
        self.encoder_level1 = nn.Sequential(*[
            LightBlock(dim, num_heads=heads, ffn_expansion=ffn_expansion_factor, layer_norm_type=LayerNorm_type, downsample_kv=2)
            for _ in range(num_blocks)
        ])
        # global context (light)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        # branch modules
        self.baseFeature = SemanticFeatureEncoder(dim=dim, num_heads=heads)
        self.detailFeature = ProgressiveTextureEncoder(num_layers=2, dim=dim)
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        # global context modulation
        gc = self.global_context(out_enc_level1)
        out_enc_level1 = out_enc_level1 * gc
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1

# --------------------------
# DualBranchContextDecoder (light)
# --------------------------
class DualStreamContextDecoder(nn.Module):
    def __init__(self, inp_channels=1, out_channels=1, dim=64, num_blocks=2, heads=4, ffn_expansion_factor=2, bias=False, LayerNorm_type="WithBias"):
        super(DualStreamContextDecoder, self).__init__()
        # reduce channel for fused features
        self.reduce_channel = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        # small light decoder blocks
        self.encoder_level2 = nn.Sequential(*[
            LightBlock(dim, num_heads=heads, ffn_expansion=ffn_expansion_factor, layer_norm_type=LayerNorm_type, downsample_kv=2)
            for _ in range(num_blocks)
        ])
        # output head: light depthwise separable
        self.output = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim//2 if dim//2>0 else 1, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//2 if dim//2>0 else 1, out_channels, kernel_size=3, padding=1, bias=bias)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0

# --------------------------
# Exports for train.py compatibility
# --------------------------
__all__ = [
    "SemanticFeatureEncoder",
    "ProgressiveTextureEncoder",
    "DualStreamContextEncoder",
    "DualStreamContextDecoder",
]
