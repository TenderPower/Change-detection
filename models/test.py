from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmflow.models.decoders.pwcnet_decoder import PWCNetDecoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# feat1: Dict[str, torch.Tensor], feat2: Dict[str, torch.Tensor]
a1 = torch.randn(14, 3, 256, 256).cuda()
a2 = torch.randn(14, 64, 128, 128).cuda()
a3 = torch.randn(14, 256, 64, 64).cuda()
a4 = torch.randn(14, 512, 32, 32).cuda()
a5 = torch.randn(14, 1024, 16, 16).cuda()
a6 = torch.randn(14, 2048, 8, 8).cuda()
a = [a1, a2, a3, a4, a5, a6]
feat1 = {'level' + str(i): a[i - 1] for i in range(1, 7)}
feat2 = {'level' + str(i): a[i - 1] for i in range(1, 7)}

in_channels = dict(level6=81, level5=1109, level4=597, level3=341)
corr_cfg = dict(type='Correlation', max_displacement=4, padding=0)
warp_cfg = dict(type='Warp', align_corners=True, use_mask=True)
act_cfg = dict(type='LeakyReLU', negative_slope=0.1)
scaled = False
densefeat_channels = (1024, 1024, 512, 256)
post_processor = dict(type='ContextNet', in_channels=341 + sum(densefeat_channels))


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


model = CustomizedFlow(in_channels=in_channels, corr_cfg=corr_cfg, warp_cfg=warp_cfg, act_cfg=act_cfg, scaled=scaled,
                       densefeat_channels=densefeat_channels, post_processor=post_processor)

model = model.to(device)
flow = model(feat1, feat2)
