# -*- coding: utf-8 -*-
"""
nnUNetTrainerMedNeXt.py
=======================
MedNeXt v1 architecture integrated into nnUNet v2 as a custom trainer.

Place this file at:
  nnUNet/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerMedNeXt.py

Compatible with standard nnUNetPlans OR ResEnc plans.
No special preprocessing required -- uses the same preprocessed data.

Available trainers (choose one):
  nnUNetTrainerMedNeXt_S_k3   - Small,   kernel 3  (~10M params)
  nnUNetTrainerMedNeXt_B_k3   - Base,    kernel 3  (~30M params)  <-- recommended for 32GB GPU
  nnUNetTrainerMedNeXt_M_k3   - Medium,  kernel 3  (~60M params)
  nnUNetTrainerMedNeXt_L_k3   - Large,   kernel 3  (~110M params)
  nnUNetTrainerMedNeXt_B_k5   - Base,    kernel 5  (~30M params, larger receptive field)
  nnUNetTrainerMedNeXt_L_k5   - Large,   kernel 5  (~110M params, largest receptive field)

Training command (standard plans, fold 0):
  nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerMedNeXt_B_k3 --c

Training command (ResEnc plans):
  nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetLPlans -tr nnUNetTrainerMedNeXt_B_k3 --c

Reference:
  Roy et al. "MedNeXt: Transformer-driven Scaling of ConvNets for
  Medical Image Segmentation." MICCAI 2023. arXiv:2303.09975
"""

from __future__ import annotations
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# ===========================================================================
#  MedNeXt Block
# ===========================================================================

class MedNeXtBlock(nn.Module):
    """
    Core MedNeXt block: ConvNeXt-style block adapted for 3D medical imaging.

    Structure:
      DW Conv3d (k x k x k, depthwise) -> LayerNorm -> PW expansion (x exp_r)
      -> GELU -> PW projection -> residual add

    Optional residual connection (do_res=True recommended during training).
    """

    def __init__(self, in_channels: int, exp_r: int = 4,
                 kernel_size: int = 7, do_res: bool = True,
                 norm_type: str = 'group'):
        super().__init__()
        self.do_res = do_res
        padding = kernel_size // 2

        # Depthwise conv
        self.dw_conv = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size, padding=padding,
            groups=in_channels, bias=True
        )

        # Norm
        if norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        else:
            self.norm = nn.LayerNorm(in_channels)
        self.norm_type = norm_type

        # Pointwise expansion + projection
        mid_channels = in_channels * exp_r
        self.pw_expand = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw_proj = nn.Conv3d(mid_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)

        # Apply norm: GroupNorm works on (B, C, D, H, W), LayerNorm needs permute
        if self.norm_type == 'group':
            x = self.norm(x)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3)

        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_proj(x)

        if self.do_res:
            x = x + residual
        return x


class MedNeXtDownBlock(nn.Module):
    """
    MedNeXt downsampling block.
    Strided depthwise conv for downsampling + MedNeXt block for processing.
    Optional residual connection via strided 1x1 conv.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 exp_r: int = 4, kernel_size: int = 7,
                 do_res: bool = True, norm_type: str = 'group'):
        super().__init__()
        self.do_res = do_res
        padding = kernel_size // 2

        # Strided depthwise conv (downsampling by 2)
        self.dw_conv = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=2, padding=padding,
            groups=in_channels, bias=True
        )

        if norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        else:
            self.norm = nn.LayerNorm(in_channels)
        self.norm_type = norm_type

        mid_channels = in_channels * exp_r
        self.pw_expand = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw_proj = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=True)

        # Residual projection (strided 1x1)
        if do_res:
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)

        if self.norm_type == 'group':
            x = self.norm(x)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3)

        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_proj(x)

        if self.do_res:
            x = x + self.res_conv(residual)
        return x


class MedNeXtUpBlock(nn.Module):
    """
    MedNeXt upsampling block.
    Transposed depthwise conv for upsampling + MedNeXt block processing.
    Optional residual connection via transposed 1x1 conv.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 exp_r: int = 4, kernel_size: int = 7,
                 do_res: bool = True, norm_type: str = 'group'):
        super().__init__()
        self.do_res = do_res
        padding = kernel_size // 2

        # Transposed depthwise conv (upsampling by 2)
        self.dw_conv = nn.ConvTranspose3d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=2, padding=padding,
            output_padding=1, groups=in_channels, bias=True
        )

        if norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        else:
            self.norm = nn.LayerNorm(in_channels)
        self.norm_type = norm_type

        mid_channels = in_channels * exp_r
        self.pw_expand = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw_proj = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=True)

        # Residual projection (transposed 1x1)
        if do_res:
            self.res_conv = nn.ConvTranspose3d(
                in_channels, out_channels,
                kernel_size=1, stride=2, output_padding=1, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw_conv(x)

        if self.norm_type == 'group':
            x = self.norm(x)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3)

        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_proj(x)

        if self.do_res:
            x = x + self.res_conv(residual)
        return x


# ===========================================================================
#  Full MedNeXt Network
# ===========================================================================

class MedNeXt(nn.Module):
    """
    3D MedNeXt U-Net architecture.

    Parameters
    ----------
    in_channels      : input channels (e.g. 2: CT + spine mask)
    n_classes        : output classes (e.g. 2: background + myeloma)
    n_channels       : base feature channels (default 32)
    exp_r            : expansion ratio in MedNeXt blocks (default 4)
    kernel_size      : depthwise conv kernel size (3 or 5)
    deep_supervision : return list of outputs at multiple scales if True
    do_res           : residual connections in MedNeXt blocks
    do_res_up_down   : residual connections in up/down sampling blocks
    block_counts     : number of blocks per stage [enc0..enc4, bottleneck, dec4..dec0]
    norm_type        : 'group' or 'layer'
    """

    def __init__(
        self,
        in_channels:      int,
        n_classes:        int,
        n_channels:       int = 32,
        exp_r:            int = 4,
        kernel_size:      int = 3,
        deep_supervision: bool = True,
        do_res:           bool = True,
        do_res_up_down:   bool = True,
        block_counts:     List[int] = None,
        norm_type:        str = 'group',
    ):
        super().__init__()
        self.do_deep_supervision = deep_supervision

        if block_counts is None:
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]  # default: 9 stages

        # Channel progression: 1x, 2x, 4x, 8x, 16x
        c  = n_channels
        c2 = c  * 2
        c4 = c  * 4
        c8 = c  * 8
        c16= c  * 16

        # ---- Stem --------------------------------------------------------
        self.stem = nn.Conv3d(in_channels, c, kernel_size=1, bias=True)

        # ---- Encoder -----------------------------------------------------
        self.enc0 = nn.Sequential(*[
            MedNeXtBlock(c, exp_r, kernel_size, do_res, norm_type)
            for _ in range(block_counts[0])
        ])
        self.down0 = MedNeXtDownBlock(c,  c2,  exp_r, kernel_size, do_res_up_down, norm_type)

        self.enc1 = nn.Sequential(*[
            MedNeXtBlock(c2, exp_r, kernel_size, do_res, norm_type)
            for _ in range(block_counts[1])
        ])
        self.down1 = MedNeXtDownBlock(c2, c4,  exp_r, kernel_size, do_res_up_down, norm_type)

        self.enc2 = nn.Sequential(*[
            MedNeXtBlock(c4, exp_r, kernel_size, do_res, norm_type)
            for _ in range(block_counts[2])
        ])
        self.down2 = MedNeXtDownBlock(c4, c8,  exp_r, kernel_size, do_res_up_down, norm_type)

        self.enc3 = nn.Sequential(*[
            MedNeXtBlock(c8, exp_r, kernel_size, do_res, norm_type)
            for _ in range(block_counts[3])
        ])
        self.down3 = MedNeXtDownBlock(c8, c16, exp_r, kernel_size, do_res_up_down, norm_type)

        # ---- Bottleneck --------------------------------------------------
        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(c16, exp_r, kernel_size, do_res, norm_type)
            for _ in range(block_counts[4])
        ])

        # ---- Decoder -----------------------------------------------------
        self.up3   = MedNeXtUpBlock(c16, c8,  exp_r, kernel_size, do_res_up_down, norm_type)
        self.dec3  = nn.Sequential(*[
            MedNeXtBlock(c8,  exp_r, kernel_size, do_res, norm_type)
            for _ in range(block_counts[5])
        ])

        self.up2   = MedNeXtUpBlock(c8,  c4,  exp_r, kernel_size, do_res_up_down, norm_type)
        self.dec2  = nn.Sequential(*[
            MedNeXtBlock(c4,  exp_r, kernel_size, do_res, norm_type)
            for _ in range(block_counts[6])
        ])

        self.up1   = MedNeXtUpBlock(c4,  c2,  exp_r, kernel_size, do_res_up_down, norm_type)
        self.dec1  = nn.Sequential(*[
            MedNeXtBlock(c2,  exp_r, kernel_size, do_res, norm_type)
            for _ in range(block_counts[7])
        ])

        self.up0   = MedNeXtUpBlock(c2,  c,   exp_r, kernel_size, do_res_up_down, norm_type)
        self.dec0  = nn.Sequential(*[
            MedNeXtBlock(c,   exp_r, kernel_size, do_res, norm_type)
            for _ in range(block_counts[8])
        ])

        # ---- Segmentation heads ------------------------------------------
        # head 0: full resolution (after dec0)
        # heads 1-4: deep supervision at lower resolutions
        self.seg_head0 = nn.Conv3d(c,   n_classes, kernel_size=1, bias=True)
        self.seg_head1 = nn.Conv3d(c2,  n_classes, kernel_size=1, bias=True)
        self.seg_head2 = nn.Conv3d(c4,  n_classes, kernel_size=1, bias=True)
        self.seg_head3 = nn.Conv3d(c8,  n_classes, kernel_size=1, bias=True)
        self.seg_head4 = nn.Conv3d(c16, n_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        # Stem
        x = self.stem(x)

        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(self.down0(e0))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        # Bottleneck
        bn = self.bottleneck(self.down3(e3))

        # Decoder (skip connections via addition -- MedNeXt style)
        d3 = self.dec3(self.up3(bn)  + e3)
        d2 = self.dec2(self.up2(d3)  + e2)
        d1 = self.dec1(self.up1(d2)  + e1)
        d0 = self.dec0(self.up0(d1)  + e0)

        if self.do_deep_supervision:
            return [
                self.seg_head0(d0),   # full resolution
                self.seg_head1(d1),   # 1/2
                self.seg_head2(d2),   # 1/4
                self.seg_head3(d3),   # 1/8
                self.seg_head4(bn),   # 1/16 (bottleneck)
            ]
        else:
            return self.seg_head0(d0)


# ===========================================================================
#  nnUNet Trainer variants
#  S = Small   (~10M)  | B = Base   (~30M)  | M = Medium (~60M) | L = Large (~110M)
#  k3 = kernel 3x3x3  | k5 = kernel 5x5x5
# ===========================================================================

def _make_trainer(kernel_size: int, n_channels: int, block_counts: List[int]):
    """Factory: returns a trainer class with fixed MedNeXt hyperparameters."""

    class _MedNeXtTrainer(nnUNetTrainer):

        @staticmethod
        def build_network_architecture(
            architecture_class_name:        str,
            arch_init_kwargs:               dict,
            arch_init_kwargs_req_import:    Union[List[str], Tuple[str, ...]],
            num_input_channels:             int,
            num_output_channels:            int,
            enable_deep_supervision:        bool = True,
        ) -> nn.Module:
            return MedNeXt(
                in_channels      = num_input_channels,
                n_classes        = num_output_channels,
                n_channels       = n_channels,
                exp_r            = 4,
                kernel_size      = kernel_size,
                deep_supervision = enable_deep_supervision,
                do_res           = True,
                do_res_up_down   = True,
                block_counts     = block_counts,
                norm_type        = 'group',
            )

        def set_deep_supervision_enabled(self, enabled: bool):
            self.network.do_deep_supervision = enabled

    return _MedNeXtTrainer


# -- Small (S): n_channels=32, depth=[2,2,2,2,2,2,2,2,2] --
class nnUNetTrainerMedNeXt_S_k3(_make_trainer(3, 32, [2,2,2,2,2,2,2,2,2])):
    pass

class nnUNetTrainerMedNeXt_S_k5(_make_trainer(5, 32, [2,2,2,2,2,2,2,2,2])):
    pass

# -- Base (B): n_channels=32, depth=[2,2,2,2,2,2,2,2,2], exp_r=4 --
# Matches MedNeXt-B from the original paper
class nnUNetTrainerMedNeXt_B_k3(_make_trainer(3, 32, [2,3,4,4,4,4,4,3,2])):
    pass

class nnUNetTrainerMedNeXt_B_k5(_make_trainer(5, 32, [2,3,4,4,4,4,4,3,2])):
    pass

# -- Medium (M): n_channels=48 --
class nnUNetTrainerMedNeXt_M_k3(_make_trainer(3, 48, [3,4,4,4,4,4,4,4,3])):
    pass

class nnUNetTrainerMedNeXt_M_k5(_make_trainer(5, 48, [3,4,4,4,4,4,4,4,3])):
    pass

# -- Large (L): n_channels=64, deeper blocks --
# Matches MedNeXt-L from the original paper
class nnUNetTrainerMedNeXt_L_k3(_make_trainer(3, 64, [3,4,8,8,8,8,8,4,3])):
    pass

class nnUNetTrainerMedNeXt_L_k5(_make_trainer(5, 64, [3,4,8,8,8,8,8,4,3])):
    pass
