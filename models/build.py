from .CloFormer import CloFormer

def build_model():
    
    model = CloFormer(
        in_chans=3,
        num_classes=9,
        embed_dims=[32, 64, 128, 256],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 16],
        group_splits=[[1, 1], [1, 1, 2], [2, 2, 4], [4, 12]],
        kernel_sizes=[[5], [5, 3], [5, 3], [3]],
        window_sizes=[8, 4, 2, 1],
        mlp_kernel_sizes=[5, 5, 3, 3],
        mlp_ratios=[4, 4, 4, 4],
        attn_drop=0.,
        mlp_drop=0.,
        qkv_bias=True,
        drop_path_rate=0.,
        use_checkpoint=False
    )

    return model
