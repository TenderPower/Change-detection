from typing import Dict, Optional, Sequence, Tuple, Union, List
from mmflow.models.decoders.pwcnet_decoder import PWCNetDecoder
import torch
import torch.nn as nn
from models.coattention import MyLayer
from einops import rearrange
import models.CCL_pytorch as ccl
import models.net as net


class PostModule(nn.Module):
    def __init__(self, channels=2048):
        super().__init__()
        self.layer = CCLayer()
        self.homolayer = HomoLayer()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=1)

    def forward(self, left_features, right_features, left2right_features, right2left_features):
        weighted_r_f = self.layer(left_features, right_features)
        weighted_l_f = self.layer(right_features, left_features)
        # 对weight进行CNN，方便与homo融合
        weighted_r_f_c = self.cnn(weighted_r_f)
        weighted_l_f_c = self.cnn(weighted_l_f)
        # homo
        weighted_r_h = self.homolayer(right_features, left2right_features)
        weighted_l_h = self.homolayer(left_features, right2left_features)
        # homo+weight_cnn -- 新思路中的思路一 --效果比*好
        weighted_l = weighted_l_h + weighted_r_f_c
        weighted_r = weighted_r_h + weighted_l_f_c
        left_attended_features = torch.cat((left_features, weighted_l), 1)
        right_attended_features = torch.cat((right_features, weighted_r), 1)
        # 不进行上采样
        return left_attended_features, right_attended_features, weighted_r_f, weighted_l_f


class HomoLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature1, feature2):
        correlation = feature1 - feature2
        return correlation


class CCLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_features, reference_features):
        c_probability = ccl.CCL(query_features, reference_features)
        return c_probability


# 新模型，输入：对齐后的图片对，输出differ 和 flow

class FModule(nn.Module):
    def __init__(self, in_channels, corr_cfg, warp_cfg, act_cfg, scaled, densefeat_channels, post_processor):
        super().__init__()
        self.getDiff = CustomizedFlow(in_channels=in_channels, corr_cfg=corr_cfg, warp_cfg=warp_cfg, act_cfg=act_cfg,
                                      scaled=scaled, densefeat_channels=densefeat_channels,
                                      post_processor=post_processor)

    def forward(self, feat_1: List[torch.Tensor], feat_2: List[torch.Tensor]):
        feat1 = {'level' + str(i): feat_1[i - 1] for i in range(1, 7)}
        feat2 = {'level' + str(i): feat_2[i - 1] for i in range(1, 7)}

        flow_pred, diffs = self.getDiff(feat1, feat2)
        level_keys = list(feat1.keys())
        level_keys.sort()
        level_keys= level_keys[1:]  # level2-level6
        for level in level_keys[::-1]:
            if level == 'level6':
                diff = feat1[level] - feat2[level]
                feat1[level] = torch.cat((feat1[level], diff), 1)
            else:
                feat1[level] = torch.cat((feat1[level], diffs[level]), 1)

        # 将Dict 转为 List
        image_feat = list(feat1.values())
        return image_feat


class CustomizedFlow(PWCNetDecoder):
    def __init__(self,
                 in_channels: Dict[str, int],
                 densefeat_channels: Sequence[int] = (128, 128, 96, 64, 32),
                 flow_div: float = 20.,
                 corr_cfg: dict = dict(type='Correlation', max_displacement=4),
                 scaled: bool = False,
                 warp_cfg: dict = dict(type='Warp', align_corners=True),
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 post_processor: dict = None,
                 flow_loss: Optional[dict] = None,
                 init_cfg: Optional[Union[list, dict]] = None) -> None:

        super().__init__(in_channels, densefeat_channels, flow_div, corr_cfg, scaled, warp_cfg, conv_cfg, norm_cfg,
                         act_cfg, post_processor, flow_loss, init_cfg)
        self.upflow = nn.ConvTranspose2d(
            2, 2, kernel_size=4, stride=2, padding=1
        )
        self.finalevel = "level2"

    def forward(self, feat1: Dict[str, torch.Tensor],
                feat2: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward function for PWCNetDecoder.

         Args:
             feat1 (Dict[str, Tensor]): The feature pyramid from the first
                 image.
             feat2 (Dict[str, Tensor]): The feature pyramid from the second
                 image.

         Returns:
             Dict[str, Tensor]: The predicted multi-levels optical flow.
         """

        flow_pred = dict()
        flow = None
        upflow = None
        upfeat = None
        diffs = dict()
        for level in self.flow_levels[::-1]:
            _feat1, _feat2 = feat1[level], feat2[level]

            if level == self.start_level:
                corr_feat = self.corr_block(_feat1, _feat2)
            else:
                warp_feat = self.warp(_feat2, upflow * self.multiplier[level])
                diff = _feat1 - warp_feat
                diffs[level] = diff
                corr_feat_ = self.corr_block(_feat1, warp_feat)
                corr_feat = torch.cat((corr_feat_, _feat1, upflow, upfeat),
                                      dim=1)

            flow, feat, upflow, upfeat = self.decoders[level](corr_feat)

            flow_pred[level] = flow

        if self.post_processor is not None:
            post_flow = self.post_processor(feat)
            flow_pred[self.end_level] = flow_pred[self.end_level] + post_flow

        # 最终的flow进行up，然后减
        _feat1, _feat2 = feat1[self.finalevel], feat2[self.finalevel]
        final_level_upflow = self.upflow(flow_pred[self.end_level])
        warp_feat = self.warp(_feat2, final_level_upflow)
        diff = _feat1 - warp_feat
        diffs[self.finalevel] = diff
        return flow_pred, diffs


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 下面的都是resnet50的
    a1 = torch.randn(14, 3, 256, 256).cuda()
    a2 = torch.randn(14, 64, 128, 128).cuda()
    a3 = torch.randn(14, 256, 64, 64).cuda()
    a4 = torch.randn(14, 512, 32, 32).cuda()
    a5 = torch.randn(14, 1024, 16, 16).cuda()
    a6 = torch.randn(14, 2048, 8, 8).cuda()
    a = [a1, a2, a3, a4, a5, a6]
    in_channels = dict(level6=81, level5=1109, level4=597, level3=341)
    corr_cfg = dict(type='Correlation', max_displacement=4, padding=0)
    warp_cfg = dict(type='Warp', align_corners=True, use_mask=True)
    act_cfg = dict(type='LeakyReLU', negative_slope=0.1)
    scaled = False
    densefeat_channels = (1024, 1024, 512, 256)
    post_processor = dict(type='ContextNet', in_channels=341 + sum(densefeat_channels))
    model = FModule(in_channels=in_channels, corr_cfg=corr_cfg, warp_cfg=warp_cfg, act_cfg=act_cfg,
                    scaled=scaled,
                    densefeat_channels=densefeat_channels, post_processor=post_processor)

    model = model.to(device)
    flow = model(a, a)
    print("over")
