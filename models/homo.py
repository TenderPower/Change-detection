import torch

from utils.general import tensor_to_PIL
from utils.alignment import get_keypoints
import cv2
import numpy
from kornia.geometry.transform import HomographyWarper
from kornia.geometry.transform.imgwarp import warp_perspective


def alignIm2getH(imgs1, imgs2):
    # 直接求出H，应用到各自的feature上
    def getcv(imgs):
        a = [i for i in imgs]
        b = [tensor_to_PIL(i) for i in a]
        c = [cv2.cvtColor(numpy.asarray(i), cv2.COLOR_RGB2BGR) for i in b]
        return c

    imgs1_cv = getcv(imgs1)
    imgs2_cv = getcv(imgs2)
    # 求H
    h = [getH(i1, i2) for i1, i2 in zip(imgs1_cv, imgs2_cv)]
    h_inv = [numpy.linalg.inv(i) for i in h]
    # 目前还是在cpu下，是否放进到gpu中
    h_tensor = torch.stack([torch.from_numpy(i).to(dtype=torch.float32).cuda() for i in h], 0)
    h_inv_tensor = torch.stack([torch.from_numpy(i).to(dtype=torch.float32).cuda() for i in h_inv], 0)
    return h_tensor, h_inv_tensor


def alignfea(inputs1, h_tensor):
    h, w = inputs1.shape[-2:]
    # 将得到H进行warp--使用的是HomographyWarper
    # warper = HomographyWarper(h, w)
    # outputs1 = warper(inputs1, h_tensor)  # inputs1->inputs2
    outputs1 = warp_perspective(inputs1, h_tensor, (h, w), )
    return outputs1


def alignIm(inputs1, inputs2):
    num = inputs1.shape[1]

    def getcv(input):
        # input都在cuda里，先进行转换
        input = input.cpu().clone()
        # tensor -> cv2
        a = [i for i in input]
        b = [[i[j, :, :] for j in range(0, num)] for i in a]
        c = [sum(one for one in i) for i in b]
        d = [tensor_to_PIL(i) for i in c]
        e = [cv2.cvtColor(numpy.asarray(i), cv2.COLOR_RGB2BGR) for i in d]
        return e

    inputs1_cv = getcv(inputs1)
    inputs2_cv = getcv(inputs2)
    # 求H
    h = [getH(i1, i2) for i1, i2 in zip(inputs1_cv, inputs2_cv)]
    # 目前还是在cpu下，是否放进到gpu中
    h_tensor = torch.stack([torch.from_numpy(i).to(dtype=torch.float32).cuda() for i in h], 0)
    h, w = inputs1.shape[-2:]
    # 将得到H进行warp --使用的是HomographyWarper
    # warper = HomographyWarper(h, w)
    # outputs1 = warper(inputs1, h_tensor)  # inputs1->inputs2
    outputs1 = warp_perspective(inputs1, h_tensor, (h, w), )
    return outputs1


def getH(input1, input2):
    H_default = numpy.array([[1., 0, 0], [0, 1, 0], [0, 0, 1]])
    keypoints_dict, criterion_perspect = get_keypoints(input1, input2)
    if criterion_perspect:
        return H_default
    points1 = keypoints_dict['points1']
    points2 = keypoints_dict['points2']
    if len(points1) > 5 or len(points2) > 5:
        # Homography
        M_H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
        if M_H is not None:
            return M_H
        else:
            return H_default
    else:
        return H_default
