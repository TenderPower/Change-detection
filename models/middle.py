from typing import Dict, Optional, Sequence, Tuple, Union, List
from mmflow.models.decoders.pwcnet_decoder import PWCNetDecoder
import torch
import torch.nn as nn
from models.coattention import MyLayer
from einops import rearrange
import models.CCL_pytorch as ccl
import models.net as net


class FuseChannelsModuleAtten(nn.Module):
    def __init__(self, input_dim, criterion):
        super().__init__()
        if criterion == 4:
            self.getDiffer = getDifference()
        else:
            self.getDiffer = getBothDifferences(input_dim)

    def forward(self, left_features, right_features):
        leftAttendedFeatures = self.getDiffer(left_features, right_features)
        rightAttendedFeatures = self.getDiffer(right_features, left_features)
        return leftAttendedFeatures, rightAttendedFeatures


class getBothDifferences(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.homo = HomoLayer()
        self.crossAttentionLayer = CoAttentionLayer(input_dim, input_dim // 4)

    def forward(self, query_features, reference_features):
        B, C, h, _ = query_features.size()
        C = C // 2
        # 原图1
        queryOriginFeature = query_features[:, :C, :, :]
        queryTransFeature = query_features[:, C:, :, :]
        # 原图2
        referenceOriginFeature = reference_features[:, :C, :, :]
        # 进行cross attention
        crossDiff = self.crossAttentionLayer(queryOriginFeature, referenceOriginFeature)
        # 进行- 获取differ
        reduceDiff = self.homo(queryOriginFeature, queryTransFeature)
        # Differ 合并
        diff = crossDiff + reduceDiff
        # 原图和Differ拼接
        return torch.cat((queryOriginFeature, diff), 1)


class getDifference(nn.Module):
    def __init__(self):
        super().__init__()
        self.homo = HomoLayer()

    def forward(self, query_features, *features):
        B, C, h, _ = query_features.size()
        C = int(C / 2)
        queryOriginFeature = query_features[:, :C, :, :]
        queryTransFeature = query_features[:, C:, :, :];
        # 进行 - 获取differ
        reduceDiff = self.homo(queryOriginFeature, queryTransFeature)
        # 原图和Differ拼接
        return torch.cat((queryOriginFeature, reduceDiff), 1)


class HomoLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_features, reference_features):
        correlation = query_features - reference_features
        weights = nn.Softmax(dim=1)(correlation)
        output = torch.einsum('bcij,bcij->bcij', reference_features, weights)  # [batch_size, c, height, width]
        return output


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # 一个可学习的参数 gamma，它用于控制自注意力的强度
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


# Cross attention
class CoAttentionLayer(nn.Module):
    def __init__(self, input_channels=2048, hidden_channels=256):
        super().__init__()
        self.reference_dimensionality_reduction = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.query_dimensionality_reduction = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, query_features, reference_features):
        Q = self.query_dimensionality_reduction(query_features)
        K = self.reference_dimensionality_reduction(reference_features)
        V = rearrange(reference_features, "b c h w -> b c (h w)")
        attention_map = torch.einsum("bcij,bckl->bijkl", Q, K)
        attention_map = rearrange(attention_map, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")
        attention_map = nn.Softmax(dim=3)(attention_map)
        attended_features = torch.einsum("bijp,bcp->bcij", attention_map, V)
        return attended_features
