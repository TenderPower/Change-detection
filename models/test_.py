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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_grad_enabled(False)

class Test(nn.Module):
    def __init__(self, nms_radius=4, keypoint_threshold=0.005, max_keypoints=1024, superglue="indoor",
                 sinkhorn_iterations=20, match_threshold=0.2, resize=256):
        super().__init__()
        config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
        self._matching = Matching(config).eval()
        self._resize = K.augmentation.Resize(resize, side="long")


    def forward(self, batch):
        """
        batch:List
        """
        left2right = []
        right2left = []
        for i in range(batch['left_image'].shape[0]):
            image1 = batch['left_image'][i]
            image2 = batch['right_image'][i]
            inp1 = K.color.rgb_to_grayscale(image1).unsqueeze(0)
            inp2 = K.color.rgb_to_grayscale(image2).unsqueeze(0)
            inp1 = self._resize(inp1)
            inp2 = self._resize(inp2)
            pred = self._matching({'image0': inp1, 'image1': inp2})
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
                kpts1 = kpts1.cpu().numpy()
                kpts2 = kpts2.cpu().numpy()
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

        left2right = torch.stack(left2right, dim=0).cuda()
        right2left = torch.stack(right2left, dim=0).cuda()

        batch['left2right'] = left2right
        batch['right2left'] = right2left

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


def alignImage(batch):
    model = Test()
    model = model.to(device)
    batch = model(batch)

    return batch


# if __name__ == '__main__':
#     i = 57
#     path_1 = f'/home/ygk/disk/pycharm_project/The-Change-You-Want-to-See-main/imgs/kubric/_{i}/left_{i}.png'
#     path_2 = f'/home/ygk/disk/pycharm_project/The-Change-You-Want-to-See-main/imgs/kubric/_{i}/right_{i}.png'
#     image_tensor_scan = read_image_as_tensor(path_to_image=path_1).unsqueeze(0).to(device)
#     image_tensor_refer = read_image_as_tensor(path_to_image=path_2).unsqueeze(0).to(device)
#     batch = {'left_image': image_tensor_scan, 'right_image': image_tensor_refer}
#     aI = alignImage(batch)
#
#     # ----------------------显示对齐后的图像---------------------
#     save_path = f'/home/ygk/disk/pycharm_project/The-Change-You-Want-to-See-main/imgs/aaa/{i}.png'
#     print(save_path)
#     # 如果目录不存在，创建目录
#     if not os.path.exists(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path))
#
#     cv2.imwrite(save_path, aI)
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