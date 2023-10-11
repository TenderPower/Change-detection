import torch
import torch.nn as nn
import torch.nn.functional as F


# A pytorch implement of CCL as described in [1]
# [1] Nie et al. Depth-Aware Multi-Grid Deep Homography Estimation with Contextual Correlation. TCSVT, 2021.

############ usage ############
# feature_1: bs, c, h, w      --- feature maps encoded from img1
# feature_2: bs, c, h, w      --- feature maps encoded from img2
# correlation: bs, 2, h, w    --- the correlation flow
# correlation = CCL(feature_1, feature_2)
###############################

def extract_patches(x, kernel=3, stride=1):
    if kernel != 1:
        x = nn.ZeroPad2d(1)(x)
    x = x.permute(0, 2, 3, 1)
    all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
    return all_patches


def CCL(feature_1, feature_2):
    bs, c, h, w = feature_1.size()

    norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
    norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
    # print(norm_feature_2.size())

    patches = extract_patches(norm_feature_2)
    if torch.cuda.is_available():
        patches = patches.cuda()
    matching_filters = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))

    match_vol = []
    for i in range(bs):
        single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1)
        match_vol.append(single_match)

    match_vol = torch.cat(match_vol, 0)
    # print(match_vol .size())

    # scale softmax
    softmax_scale = 10
    match_vol = F.softmax(match_vol * softmax_scale, 1)  # (bs, w*h, w, h)

    # match_vol 不就是弱相关被抑制，而最强相关被增强
    # 那我就让他1-match_vol 让他相反，变成弱相关被增强，强相关被抑制
    # optical
    # a = torch.ones(match_vol.shape).cuda()
    # match_vol_ = a - match_vol

    # 将(bs,w*h,w,h) 转为 (bs,1,w,h)
    match_vol = torch.max(match_vol, 1).values.unsqueeze(1)
    return match_vol


if __name__ == '__main__':
    f1 = torch.randn(4, 3, 5, 5).cuda()
    f2 = torch.randn(4, 3, 5, 5).cuda()
    c = CCL(f1, f2)
