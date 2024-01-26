import torch
import torch.nn as nn
import kornia as K
import numpy as np
from SuperGluePretrainedNetwork.models.matching import Matching
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import random
import cv2
import utils.general as general

import os

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def alignImage(batch):
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': "indoor",
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    matching = Matching(config).eval()

    # 使用Matching
    left2right = []
    right2left = []
    for i in range(batch['left_image'].shape[0]):
        image1 = batch['left_image'][i]
        image2 = batch['right_image'][i]
        inp1 = K.color.rgb_to_grayscale(image1).unsqueeze(0)
        inp2 = K.color.rgb_to_grayscale(image2).unsqueeze(0)
        pred = matching({'image0': inp1, 'image1': inp2})
        kpts1, kpts2 = pred['keypoints0'][0], pred['keypoints1'][0]
        matches, conf = pred['matches0'][0], pred['matching_scores0'][0]

        valid = matches != -1
        conf = conf[valid]
        kpts1 = kpts1[valid]
        kpts2 = kpts2[matches[valid]]

        conf, sort_idx = conf.sort(descending=True)
        kpts1 = kpts1[sort_idx]
        kpts2 = kpts2[sort_idx]


        if kpts1.shape[0] < 4:
            left2right.append(image1)
            right2left.append(image2)

        else:
            # 提取匹配的关键点
            kpts1 = kpts1.numpy()
            kpts2 = kpts2.numpy()
            # 计算单应性矩阵
            H, _ = cv2.findHomography(kpts1, kpts2, cv2.RANSAC, 3)
            inverH = torch.pinverse(torch.Tensor(H)).numpy()
            # 使用单应性矩阵对img1进行变换，使其与img2对齐
            image1_cv = tensor2cv(image1)
            image2_cv = tensor2cv(image2)
            aligned_img1 = cv2.warpPerspective(image1_cv, H, (256, 256))
            aligned_img2 = cv2.warpPerspective(image2_cv, inverH, (256, 256))
            # 将cv转为tensor
            image1_to_image2 = pil_to_tensor(
                Image.fromarray(cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2RGB))).float() / 255.0
            image2_to_image1 = pil_to_tensor(
                Image.fromarray(cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2RGB))).float() / 255.0

            left2right.append(image1_to_image2)
            right2left.append(image2_to_image1)

    left2right = torch.stack(left2right, dim=0)
    right2left = torch.stack(right2left, dim=0)
    left_images = batch['left_image']
    right_images = batch['right_image']
    imag_one = torch.concat((left_images, right2left), 1)
    imag_two = torch.concat((right_images, left2right), 1)

    batch['imag_one'] = imag_one
    batch['imag_two'] = imag_two
    return batch


def read_image_as_tensor(path_to_image):
    """
    Returms a normalised RGB image as tensor.
    """
    pil_image = Image.open(path_to_image).convert("RGB")
    image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
    return image_as_tensor


def tensor2cv(image):
    image_plt = general.tensor_to_PIL(image)
    image_cv = cv2.cvtColor(np.asarray(image_plt), cv2.COLOR_RGB2BGR)
    return image_cv


if __name__ == '__main__':
    for i in [44, 57, 61, 70, 71, 72, 94, 146, 148, 150, 156, 179, 180, 181, 210, 215, 216, 217, 232, 264, 265, 328,
              329, 330, 351, 395, 397, 414, 482, 483, 569, 570, 571, 646, 689, 692, 693, 694, 725, 727, 732, 742, 743,
              744, 807, 819, 843, 860, 878, 975, 1000, 1003, 1004, 1028, 1029, 1036, 1051, 1053, 1072, 1088, 1089, 1090,
              1091, 1092, 1108, 1133, 1134, 1149, 1170, 1202, 1208, 1255, 1258, 1312, 1313, 1314, 1315, 1316, 1328,
              1330, 1331, 1372, 1374, 1404, 1457, 1458, 1511, 1514, 1515, 1528, 1529, 1577, 1579, 1602]:
        path_1 = f'/home/ygk/disk/pycharm_project/The-Change-You-Want-to-See-main/imgs/kubric/_{i}/left_{i}.png'
        path_2 = f'/home/ygk/disk/pycharm_project/The-Change-You-Want-to-See-main/imgs/kubric/_{i}/right_{i}.png'
        image_tensor_scan = read_image_as_tensor(path_to_image=path_1)
        image_tensor_refer = read_image_as_tensor(path_to_image=path_2)
        # batch = {'left_image': image_tensor_scan, 'right_image': image_tensor_refer}
        aI, _ = alignImage(image_tensor_scan, image_tensor_refer)

        # ----------------------显示对齐后的图像---------------------
        save_path = f'/home/ygk/disk/pycharm_project/The-Change-You-Want-to-See-main/imgs/aaa/{i}.png'
        print(save_path)
        # 如果目录不存在，创建目录
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        cv2.imwrite(save_path, aI)
#     # -----------------------END----------------------------
#     pass
#     def forward(self, batch):
#         """
#         batch:List
#         """
#         for i in range(len(batch)):
#             image1 = batch[i]['left_image']
#             image2 = batch[i]['right_image']
#             inp1 = K.color.rgb_to_grayscale(image1).unsqueeze(0)
#             inp2 = K.color.rgb_to_grayscale(image2).unsqueeze(0)
#             inp1 = self._resize(inp1)
#             inp2 = self._resize(inp2)
#             pred = self._matching({'image0': inp1, 'image1': inp2})
#             kpts1, kpts2 = pred['keypoints0'][0], pred['keypoints1'][0]
#             matches, conf = pred['matches0'][0], pred['matching_scores0'][0]
#
#             valid = matches != -1
#             conf = conf[valid]
#             kpts1 = kpts1[valid]
#             kpts2 = kpts2[matches[valid]]
#
#             conf, sort_idx = conf.sort(descending=True)
#             kpts1 = kpts1[sort_idx]
#             kpts2 = kpts2[sort_idx]
#
#             if kpts1.shape[0] < 4:
#                 batch[i]['left2right'] = image1
#                 batch[i]['right2left'] = image2
#
#             else:
#                 # 提取匹配的关键点
#                 kpts1 = kpts1.numpy()
#                 kpts2 = kpts2.numpy()
#                 # 计算单应性矩阵
#                 H, _ = cv2.findHomography(kpts1, kpts2, cv2.RANSAC, 3)
#                 inverH = torch.pinverse(torch.Tensor(H)).numpy()
#                 # 使用单应性矩阵对img1进行变换，使其与img2对齐
#                 image1_cv = tensor2cv(image1)
#                 image2_cv = tensor2cv(image2)
#                 aligned_img1 = cv2.warpPerspective(image1_cv, H, (256, 256))
#                 aligned_img2 = cv2.warpPerspective(image2_cv, inverH, (256, 256))
#                 # 将cv转为tensor
#                 image1_to_image2 = pil_to_tensor(
#                     Image.fromarray(cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2RGB))).float() / 255.0
#                 image2_to_image1 = pil_to_tensor(
#                     Image.fromarray(cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2RGB))).float() / 255.0
#
#                 batch[i]['left2right'] = image1_to_image2
#                 batch[i]['right2left'] = image2_to_image1
#
#         return batch


# def forward(self, batch):
#     """
#     batch:List
#     """
#     left2right = []
#     right2left = []
#     for i in range(batch['left_image'].shape[0]):
#         image1 = batch['left_image'][i]
#         image2 = batch['right_image'][i]
#         inp1 = K.color.rgb_to_grayscale(image1).unsqueeze(0)
#         inp2 = K.color.rgb_to_grayscale(image2).unsqueeze(0)
#         inp1 = self._resize(inp1)
#         inp2 = self._resize(inp2)
#         pred = self._matching({'image0': inp1, 'image1': inp2})
#         kpts1, kpts2 = pred['keypoints0'][0], pred['keypoints1'][0]
#         matches, conf = pred['matches0'][0], pred['matching_scores0'][0]
#
#         valid = matches != -1
#         conf = conf[valid]
#         kpts1 = kpts1[valid]
#         kpts2 = kpts2[matches[valid]]
#
#         conf, sort_idx = conf.sort(descending=True)
#         kpts1 = kpts1[sort_idx]
#         kpts2 = kpts2[sort_idx]
#
#         if kpts1.shape[0] < 4:
#             left2right.append(image1)
#             right2left.append(image2)
#
#         else:
#             # 提取匹配的关键点
#             kpts1 = kpts1.numpy()
#             kpts2 = kpts2.numpy()
#             # 计算单应性矩阵
#             H, _ = cv2.findHomography(kpts1, kpts2, cv2.RANSAC, 3)
#             inverH = torch.pinverse(torch.Tensor(H)).numpy()
#             # 使用单应性矩阵对img1进行变换，使其与img2对齐
#             image1_cv = tensor2cv(image1)
#             image2_cv = tensor2cv(image2)
#             aligned_img1 = cv2.warpPerspective(image1_cv, H, (256, 256))
#             aligned_img2 = cv2.warpPerspective(image2_cv, inverH, (256, 256))
#             # 将cv转为tensor
#             image1_to_image2 = pil_to_tensor(
#                 Image.fromarray(cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2RGB))).float() / 255.0
#             image2_to_image1 = pil_to_tensor(
#                 Image.fromarray(cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2RGB))).float() / 255.0
#
#             left2right.append(image1_to_image2)
#             right2left.append(image2_to_image1)
#
#     left2right = torch.stack(left2right, dim=0)
#     right2left = torch.stack(right2left, dim=0)
#
#     batch['left2right'] = left2right
#     batch['right2left'] = right2left
#
#     return batch
