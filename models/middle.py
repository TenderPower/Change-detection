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
        # 单单使用概率进行上采样
        return left_attended_features, right_attended_features, weighted_l, weighted_r


class FuseMoudle(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = MyLayer()
        self.homolayer = HomoLayer()

    def forward(self, left_features, right_features, left2right_features, right2left_features, weighted_r, weighted_l):
        # 向上采样 上一层的权重
        up = nn.Upsample(scale_factor=2, mode='bilinear')
        weighted_r_ = up(weighted_r)
        weighted_l_ = up(weighted_l)
        # 这一层的权重
        weighted_r = self.layer(left_features, right2left_features)
        weighted_l = self.layer(right_features, left2right_features)
        # 上一层和这一层进行融合(flow)
        weighted_r_f = torch.cat((weighted_r, weighted_r_), 1)
        weighted_l_f = torch.cat((weighted_l, weighted_l_), 1)
        # (homo)
        weighted_r_h = self.homolayer(right_features, left2right_features)
        weighted_l_h = self.homolayer(left_features, right2left_features)
        # 合并
        weighted_r = torch.cat((weighted_r_h, weighted_l_f), 1)
        weighted_l = torch.cat((weighted_l_h, weighted_r_f), 1)

        left_attended_features = torch.cat((left_features, weighted_l), 1)
        right_attended_features = torch.cat((right_features, weighted_r), 1)

        return left_attended_features, right_attended_features, weighted_l, weighted_r


class MiddleModule(nn.Module):
    def __init__(self, channels=1024, previous_channels=512):
        super().__init__()
        self.layer = CCLayer()
        self.homolayer = HomoLayer()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1x1 = nn.Conv2d(in_channels=previous_channels, out_channels=channels, kernel_size=1)
    def forward(self, left_features, right_features, left2right_features, right2left_features, weighted_l,
                weighted_r):
        # 对概率进行上采样
        weighted_l_ = self.upsample(self.conv1x1(weighted_l))
        weighted_r_ = self.upsample(self.conv1x1(weighted_r))
        # 这一层的权重
        weighted_r_f = self.layer(left_features, right_features)
        weighted_l_f = self.layer(right_features, left_features)

        # 将上一层与这一层进行相加结合 == 单单只将ccl上采样 融合这一层的ccl，效果不佳
        # weighted_r_f_all = weighted_r_f + weighted_r_f_u
        # weighted_l_f_all = weighted_l_f + weighted_l_f_u

        # 对weight进行CNN，方便与homo融合
        weighted_r_f = self.cnn(weighted_r_f)
        weighted_l_f = self.cnn(weighted_l_f)
        # (homo)
        weighted_r_h = self.homolayer(right_features, left2right_features)
        weighted_l_h = self.homolayer(left_features, right2left_features)
        # homo+weight_cnn -- 新思路中的思路一 --效果比*好
        weighted_l = weighted_l_h + weighted_r_f
        weighted_r = weighted_r_h + weighted_l_f

        # 上一层合并这一层信息--- 新思路，将上一层的所有信息进行一层上采样 然后融合，不是单纯用反卷积进行采样
        weighted_r = weighted_r_ + weighted_r
        weighted_l = weighted_l_ + weighted_l

        left_attended_features = torch.cat((left_features, weighted_l), 1)
        right_attended_features = torch.cat((right_features, weighted_r), 1)

        return left_attended_features, right_attended_features, weighted_l, weighted_r


class FrontModule(nn.Module):
    def __init__(self, channels=1024, previous_channels=512):
        super().__init__()
        self.layer = CCLayer()
        self.homolayer = HomoLayer()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1x1 = nn.Conv2d(in_channels=previous_channels, out_channels=channels, kernel_size=1)

    def forward(self, left_features, right_features, left2right_features, right2left_features,weighted_l ,
                weighted_r):
        # 对概率进行上采样
        weighted_l_ = self.upsample(self.conv1x1(weighted_l))
        weighted_r_ = self.upsample(self.conv1x1(weighted_r))
        # 这一层的权重
        weighted_r_f = self.layer(left_features, right_features)
        weighted_l_f = self.layer(right_features, left_features)

        # 将上一层与这一层进行相加结合 == 单单只将ccl上采样 融合这一层的ccl，效果不佳
        # weighted_r_f_all = weighted_r_f + weighted_r_f_u
        # weighted_l_f_all = weighted_l_f + weighted_l_f_u

        # 对weight进行CNN，方便与homo融合
        weighted_r_f = self.cnn(weighted_r_f)
        weighted_l_f = self.cnn(weighted_l_f)
        # (homo)
        weighted_r_h = self.homolayer(right_features, left2right_features)
        weighted_l_h = self.homolayer(left_features, right2left_features)

        # homo+weight_cnn -- 新思路中的思路一 --效果比*好
        weighted_l = weighted_l_h + weighted_r_f
        weighted_r = weighted_r_h + weighted_l_f
        # 上一层合并这一层信息--- 新思路，将上一层的所有信息进行一层上采样 然后融合，不是单纯用反卷积进行采样
        weighted_r = weighted_r_ + weighted_r
        weighted_l = weighted_l_ + weighted_l

        left_attended_features = torch.cat((left_features, weighted_l), 1)
        right_attended_features = torch.cat((right_features, weighted_r), 1)

        return left_attended_features, right_attended_features


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
