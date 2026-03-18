import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath

from .gcn_lib import act_layer, Grapher, HSD_Block


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'dhs_vig_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'dhs_vig_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'dhs_vig_det': _cfg(
        crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """

    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Block(nn.Module):
    def __init__(self, channel, num_knn, knn_group, dilation, conv, act, norm, bias, stochastic,
                 epsilon, reduce_ratios, n, drop_path, mode='normal'):
        super().__init__()

        self.mode = mode

        # GCN&Attention
        if mode == 'normal':
            self.GraphProcessFlow = Grapher(channel, num_knn, dilation, conv=conv, act=act, norm=norm, bias=bias,
                                            stochastic=stochastic, epsilon=epsilon, r=reduce_ratios, n=n,
                                            drop_path=drop_path, relative_pos=True)
        elif mode == 'hds':
            self.GraphProcessFlow = HSD_Block(channel, knn_group, dilation, conv=conv, act=act, norm=norm,
                                              bias=bias, stochastic=stochastic, epsilon=epsilon, r=reduce_ratios,
                                              n=n, drop_path=drop_path, relative_pos=True)
        else:
            raise NotImplementedError('mode:{} is not supported'.format(mode))

        self.FFN = FFN(in_features=channel, hidden_features=channel*4, out_features=channel, act=act, drop_path=drop_path)

    def forward(self, x):
        score_map = self.GraphProcessFlow(x)
        ffn_map = self.FFN(score_map)

        return ffn_map


class dhs_vig(torch.nn.Module):
    def __init__(self, opt):
        super(dhs_vig, self).__init__()
        print(opt)
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        emb_dims = opt.emb_dims
        drop_path = opt.drop_path
        self.use_pos_embed = opt.use_pos_embed
        self.use_feature_extractor = opt.use_feature_extractor

        mode = opt.mode

        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        num_knn_MHGPB = [9, 7, 5, 3]
        max_dilation = 49 // max(num_knn)

        self.stem = Stem(out_dim=channels[0], act=act)

        if self.use_pos_embed:
            print("using pos embed")
            self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224 // 4, 224 // 4))
        HW = 224 // 4 * 224 // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
                HW = HW // 4
            if i < 2:
                print('HDS Block')
                for j in range(blocks[i]):
                    self.backbone += nn.Sequential(Block(channels[i], num_knn[idx], num_knn_MHGPB,
                                                                        min(idx // 4 + 1, max_dilation), conv, act,
                                                                        norm, bias, stochastic, epsilon, reduce_ratios[i],
                                                                        n=HW, drop_path=dpr[idx], mode=mode))
                    idx += 1
            else:
                print('GCN Block')
                for j in range(blocks[i]):
                    self.backbone += nn.Sequential(Block(channels[i], num_knn[idx], num_knn_MHGPB,
                                                                       min(idx // 4 + 1, max_dilation), conv, act,
                                                                       norm, bias, stochastic, epsilon, reduce_ratios[i],
                                                                       n=HW, drop_path=dpr[idx], mode="normal"))
                    idx += 1
        self.backbone = Seq(*self.backbone)
        
        if self.use_feature_extractor:
            print("using feature extractor")
            self.feature_extractor = Stem(in_dim=channels[1], out_dim=channels[-1], act=act)
        
        self.forzen = opt.forzen
        self.from_pretrained = opt.from_pretrained
        self.init_weights()
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def init_weights(self):
        self.load_state_dict(torch.load(self.from_pretrained), strict=False)
        print("pretrained weight loaded")
        
        if self.forzen:
            for name, param in self.named_parameters():
                print(name, "is forzen")
                param.requires_grad = False
            print("model forzen")

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        print("BN eval mode")

    def interpolate_pos_encoding(self, x):
        w, h = x.shape[2], x.shape[3]
        p_w, p_h = self.pos_embed.shape[2], self.pos_embed.shape[3]

        if w * h == p_w * p_h and w == h:
            return self.pos_embed

        w0 = w
        h0 = h
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            self.pos_embed,
            scale_factor=(w0 / p_w, h0 / p_h),
            mode='bicubic',
            align_corners=True     # set align_corners = True to fit mmdet
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        return patch_pos_embed

    def forward(self, inputs):    
        x = self.stem(inputs)
        if self.use_pos_embed:
            x = x + self.interpolate_pos_encoding(x)

        outs = []
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

            if i == 1 or i == 4 or i == 11:
                outs.append(x)

            if self.use_feature_extractor and i == 4:
                hidden_map = self.feature_extractor(x)
        
        if self.use_feature_extractor:
            x = x + hidden_map
        outs.append(x)
        return outs


# mmdetection regis
from mmdet.registry import MODELS


@MODELS.register_module()
def dhs_vig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.1, mode='hds', **kwargs):
            self.k = 9  # neighbor num (default:9)
            self.conv = 'mr'  # graph conv layer {edge, mr}
            self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch'  # batch or instance normalization {batch, instance}
            self.bias = True  # bias of conv layer True or False
            self.dropout = 0.0  # dropout rate
            self.use_dilation = True  # use dilated knn or not
            self.epsilon = 0.2  # stochastic epsilon for gcn
            self.use_stochastic = False  # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2, 6, 2]  # number of basic blocks in the backbone
            self.channels = [80, 160, 400, 640]  # number of channels of deep features
            self.n_classes = num_classes  # Dimension of out_channels
            self.emb_dims = 1024  # Dimension of embeddings
            self.mode = mode  # Diff type GvT blocks
            self.use_pos_embed = True
            self.use_feature_extractor = True
            self.from_pretrained = "YourPath"
            self.forzen = False  # Do not forzen model, cause the drop of acc.

    opt = OptInit(**kwargs)
    model = dhs_vig(opt)
    model.default_cfg = default_cfgs['dhs_vig_det']
    return model