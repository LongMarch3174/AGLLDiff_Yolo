"""neck = dict(
    type='RepGDNeck',
    num_repeats=[12, 12, 12, 12],
    out_channels=[256, 128, 128, 256, 256, 512],
    channels_list=[64, 64, 64, 256, 512, 128, 64, 64, 128, 128, 256],
    extra_cfg=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        depths=2,
        fusion_in=960,
        fusion_act=dict(type='ReLU'),
        fuse_block_num=3,
        embed_dim_p=768,
        embed_dim_n=704,
        key_dim=8,
        num_heads=4,
        mlp_ratios=1,
        attn_ratios=2,
        c2t_stride=2,
        drop_path_rate=0.1,
        trans_channels=[128, 64, 128, 256],
        pool_mode='torch'
    )
)"""


class RepGDNeck_config:
    def __init__(self):
        self.type = 'RepGDNeck'
        self.num_repeats = [12, 12, 12, 12, 12, 12, 12, 12, 12]
        self.out_channels = [256, 128, 128, 256, 256, 512]
        self.channels_list = [64, 64, 64, 256, 512, 128, 64, 64, 128, 128, 256]
        self.extra_cfg = self.ExtraCfg()

    class ExtraCfg:
        def __init__(self):
            self.norm_cfg = dict(type='BN', requires_grad=True)
            self.depths = 2
            self.fusion_in = 960
            self.fusion_act = dict(type='ReLU')
            self.fuse_block_num = 3
            self.embed_dim_p = 768
            self.embed_dim_n = 704
            self.key_dim = 8
            self.num_heads = 4
            self.mlp_ratios = 1
            self.attn_ratios = 2
            self.c2t_stride = 2
            self.drop_path_rate = 0.1
            self.trans_channels = [128, 64, 128, 256]
            self.pool_mode = 'torch'

