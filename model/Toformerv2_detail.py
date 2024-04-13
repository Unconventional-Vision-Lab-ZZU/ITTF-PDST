import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from pytorch_msssim import ssim
# CF-Mlp
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.PReLU(),
            nn.Conv2d(hidden_features, out_features, 1)
        )

    def forward(self, x):
        return self.mlp(x)

def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, C)
    return windows

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x

class RLN(nn.Module):
    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad
        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)
    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias

# DSC-MHSA
class SwinTransformerBlock(nn.Module):
    feature = 0
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(dim, window_size, num_heads)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        self.V = nn.Conv2d(dim, dim, 1)
        self.QK = nn.Conv2d(dim, dim * 2, 1)

        self.pointwise_conv1 = nn.Conv2d(dim, dim, 1)
        self.pointwise_conv2 = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        shortcut = x
        x, rescale, rebias = self.norm1(x)

        B, C, H, W = x.shape
        V = self.V(x)
        QK = self.QK(x)
        QKV = torch.cat([QK, V], dim=1)

        # cyclic shift
        shifted_QKV = self.check_size(QKV, self.shift_size > 0)
        Ht, Wt = shifted_QKV.shape[2:]

        # partition windows
        shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
        qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

        attn_windows = self.attn(qkv)

        # merge windows
        shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)

        # reverse cyclic shift
        out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
        attn_out = out.permute(0, 3, 1, 2)

        conv_out = self.conv(V)

        addition = conv_out + attn_out

        weight = F.avg_pool2d(addition, addition.size()[2:]).view(addition.size(0), addition.size(1), 1, 1)
        weight = self.softmax(weight)

        x = self.pointwise_conv1(addition) * weight + self.pointwise_conv2(addition) * (1 - weight)


        x = x * rescale + rebias
        x = shortcut + x

        shortcut = x
        x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        x = x * rescale + rebias
        x = shortcut + x

        x = x.view(B, C, H*W)
        x = x.permute(0, 2, 1)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"




class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4*dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"



# MultiUP & BAF Subnet
class UpSample(nn.Module):
    def __init__(self, input_resolution, in_channels, scale_factor):
        super(UpSample, self).__init__()
        self.input_resolution = input_resolution
        self.factor = scale_factor
        if self.factor == 2:
            self.gate = nn.Conv2d(in_channels, 2, 3, 1, 1, bias=True)
            self.conv = nn.Conv2d(in_channels, in_channels//2, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 2*in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(scale_factor),
                                      nn.Conv2d(in_channels//2, in_channels//2, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False))

        elif self.factor == 4:
            self.gate = nn.Conv2d(in_channels, 2, 3, 1, 1, bias=True)
            self.conv = nn.Conv2d(in_channels//2, in_channels, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(nn.Conv2d(in_channels, 2 * in_channels, 1, 1, 0, bias=False),
                                      nn.PReLU(),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(in_channels // 2, in_channels // 2, 1, stride=1, padding=0, bias=False))

            self.up_b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                                      nn.PReLU(),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                      nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        """
        x: B, L = H*W, C
        """
        if type(self.input_resolution) == int:
            H = self.input_resolution
            W = self.input_resolution

        elif type(self.input_resolution) == tuple:
            H, W = self.input_resolution

        B, L, C = x.shape
        x = x.view(B, H, W, C)  # B, H, W, C
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x_p = self.up_p(x)  # pixel shuffle
        x_b = self.up_b(x)  # bilinear

        gates = self.gate(torch.cat((x_p, x_b), dim=1))
        gated_y = x_p * gates[:, [0], :, :] + x_b * gates[:, [1], :, :]
        if self.factor == 4:
            gated_y = self.conv(gated_y)

        out = gated_y.permute(0, 2, 3, 1)  # B, H, W, C
        if self.factor == 2:
            out = out.view(B, -1, C // 2)

        return out



class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"



class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = UpSample(input_resolution, in_channels=dim, scale_factor=2)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x




class Toformerv2(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, out_chans=3,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=[2, 2, 4, 4], qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="MultiUP", **kwargs):
        super(Toformerv2, self).__init__()

        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.prelu = nn.PReLU()
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1, padding_mode='reflect')

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio[i_layer],
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=RLN,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)


        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = UpSample(input_resolution=patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                    in_channels=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), scale_factor=2)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio[(self.num_layers - 1 - i_layer)],
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=RLN,
                                         upsample=UpSample if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "MultiUP":
            self.up = UpSample(input_resolution=(img_size // patch_size, img_size // patch_size),
                               in_channels=embed_dim, scale_factor=4)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.out_chans, kernel_size=3, stride=1,
                                    padding=1, bias=False)  # kernel = 1

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        residual = x
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        # x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, residual, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)  # concat last dimension
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "MultiUP":
            x = self.up(x)
            # x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W

        return x

    def forward(self, x):
        x = self.conv_first(x)
        x, residual, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        out = self.output(x)
        # x = x + residual
        return out

class Combined_loss(nn.Module):
    def __init__(self, w):
        super(Combined_loss, self).__init__()

        self.w = float(w)
        self.l1 = nn.L1Loss().cuda()


    def forward(self, pre, GT):
        output_gray = pre.mean(dim=1, keepdim=True)
        target_gray = GT.mean(dim=1, keepdim=True)
        output_gray = (output_gray * 255).type(torch.int)
        target_gray = (target_gray * 255).type(torch.int)
        Npre = torch.histc(output_gray, bins=256, min=0, max=255)
        Ngt = torch.histc(target_gray, bins=256, min=0, max=255)
        L_loss = torch.sum((Npre - Ngt) ** 2)
        L_loss = L_loss / (output_gray.size(0) * output_gray.size(2) * output_gray.size(3))
        L_loss = L_loss / (256 ** 2)
        L_loss = self.w*L_loss
        Ssim_loss=1-ssim(pre, GT, data_range=1, size_average=True)
        L1loss = self.l1(pre, GT)
        loss = L1loss+self.w*Ssim_loss+L_loss
        return loss

if __name__ == '__main__':
    height = 64
    width = 64
    x = torch.randn((1, 3, height, width))  # .cuda()
    model = Toformerv2(img_size=256, patch_size=4, in_chans=3, out_chans=3,
                  embed_dim=96, depths=[8, 8, 8, 8],
                  num_heads=[8, 8, 8, 8],
                  window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                  use_checkpoint=False, final_upsample="MultiUP")  # .cuda()
    # print(model)
    print('input image size: (%d, %d)' % (height, width))
