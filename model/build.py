# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import FocalTransformer
import torch.nn as nn

def build_model():
   
    model = FocalTransformer(img_size=224,
                            patch_size=4,
                            in_chans=3,
                            num_classes=6,
                            embed_dim=96,
                            depths=[2,2,6,2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            drop_path_rate=0.1,
                            ape=False,
                            norm_layer=nn.LayerNorm,
                            patch_norm=True,
                            use_checkpoint=False,
                            fused_window_process=False,
                            use_shift=False, 
                            focal_stages=[0, 1, 2, 3], 
                            focal_levels=[1, 1, 1, 1], 
                            focal_windows=[7, 5, 3, 1], 
                            focal_pool="fc", 
                            expand_stages=[0, 1, 2, 3], 
                            expand_sizes=[3, 3, 3, 3],
                            expand_layer="all", 
                            use_conv_embed=False, 
                            use_layerscale=False, 
                            layerscale_value=1e-4, 
                            use_pre_norm=False)
                	

    return model
