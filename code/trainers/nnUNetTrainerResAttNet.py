# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

def _t3(x):
    if isinstance(x, (int, float)):
        return (int(x),) * 3
    return tuple(int(v) for v in x)

def _needs_proj(in_ch, out_ch, stride):
    s = _t3(stride)
    return any(si != 1 for si in s) or (in_ch != out_ch)

class _SafeNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm3d(num_features, affine=True)
    def forward(self, x):
        if x.shape[2] > 1 or x.shape[3] > 1 or x.shape[4] > 1:
            return self.norm(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        s = _t3(stride)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=s, padding=1, bias=False)
        self.norm1 = _SafeNorm(out_ch)
        self.act   = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.norm2 = _SafeNorm(out_ch)
        if _needs_proj(in_ch, out_ch, s):
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1, stride=s, bias=False)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        r = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + r)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, 1, bias=True),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, 1, bias=True),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, 1, bias=True),
            nn.InstanceNorm3d(1, affine=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=False)
        alpha = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * alpha

class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks, stride):
        super().__init__()
        blocks = [ResBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            blocks.append(ResBlock(out_ch, out_ch, stride=1))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x):
        return self.blocks(x)

class DecoderStage(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, n_blocks, up_stride):
        super().__init__()
        s = _t3(up_stride)
        self.upsample = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=s, stride=s, bias=False)
        F_int = max(skip_ch // 2, 8)
        self.attn = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=F_int)
        concat_ch = in_ch + skip_ch
        blocks = [ResBlock(concat_ch, out_ch, stride=1)]
        for _ in range(n_blocks - 1):
            blocks.append(ResBlock(out_ch, out_ch, stride=1))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x, skip):
        x_up     = self.upsample(x)
        skip_att = self.attn(g=x_up, x=skip)
        return self.blocks(torch.cat([x_up, skip_att], dim=1))

class ResAttUNet(nn.Module):
    def __init__(self, input_channels, num_classes, n_stages,
                 features_per_stage, strides, n_blocks_encoder,
                 n_blocks_decoder, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.n_stages = n_stages
        self.encoders = nn.ModuleList()
        in_ch = input_channels
        for i in range(n_stages):
            self.encoders.append(
                EncoderStage(in_ch=in_ch, out_ch=features_per_stage[i],
                             n_blocks=n_blocks_encoder[i], stride=strides[i]))
            in_ch = features_per_stage[i]
        self.decoders = nn.ModuleList()
        for i in range(n_stages - 1):
            dec_in   = features_per_stage[n_stages - 1 - i]
            dec_skip = features_per_stage[n_stages - 2 - i]
            self.decoders.append(
                DecoderStage(dec_in, dec_skip, dec_skip,
                             n_blocks_decoder[i], strides[n_stages - 1 - i]))
        self.seg_heads = nn.ModuleList([
            nn.Conv3d(features_per_stage[j], num_classes, kernel_size=1)
            for j in range(n_stages - 1)
        ])
    def forward(self, x):
        enc_outs = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_outs.append(x)
        x = enc_outs[-1]
        dec_outs = []
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, enc_outs[self.n_stages - 2 - i])
            dec_outs.append(x)
        if self.deep_supervision:
            return [self.seg_heads[j](dec_outs[(self.n_stages - 2) - j])
                    for j in range(self.n_stages - 1)]
        else:
            return self.seg_heads[0](dec_outs[-1])

class nnUNetTrainerResAttNet(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        n_stages = arch_init_kwargs['n_stages']
        features = arch_init_kwargs['features_per_stage']
        strides  = arch_init_kwargs['strides']
        n_blocks_enc = arch_init_kwargs.get(
            'n_blocks_per_stage',
            arch_init_kwargs.get('n_conv_per_stage', [1] * n_stages))
        n_blocks_dec = arch_init_kwargs.get(
            'n_conv_per_stage_decoder', [1] * (n_stages - 1))
        return ResAttUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            n_stages=n_stages,
            features_per_stage=features,
            strides=strides,
            n_blocks_encoder=n_blocks_enc,
            n_blocks_decoder=n_blocks_dec,
            deep_supervision=enable_deep_supervision)

    def set_deep_supervision_enabled(self, enabled: bool):
        self.network.deep_supervision = enabled
