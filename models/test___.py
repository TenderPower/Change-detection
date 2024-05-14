import numpy
import torch
import os
import numpy as np
import kornia as K
import torch.nn as nn
from einops import rearrange
from pytorch3d.renderer import PointsRasterizationSettings, PerspectiveCameras, PointsRasterizer, PointsRenderer, \
    AlphaCompositor
from pytorch3d.structures import Pointclouds

import utilssss.general as general
import torch.nn.functional as F
from SuperGluePretrainedNetwork.models.matching import Matching
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from models.geometry import transform_points, convert_image_coordinates_to_world, sample_depth_for_given_points
import cv2
import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tensor2cv(image):
    image_plt = general.tensor_to_PIL(image)
    image_cv = cv2.cvtColor(np.asarray(image_plt), cv2.COLOR_RGB2BGR)
    return image_cv


def read_image_as_tensor(path_to_image):
    """
    Returms a normalised RGB image as tensor.
    """
    pil_image = Image.open(path_to_image).convert("RGB")
    image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
    return image_as_tensor


def get_Homo_Images(pred, image1, image2):
    kpts1, kpts2 = pred['keypoints0'][0], pred['keypoints1'][0]

    if kpts1.shape[0] < 4 or kpts2.shape[0] < 4:
        return image1, image2

    else:
        matches, conf = pred['matches0'][0], pred['matching_scores0'][0]

        valid = matches != -1
        conf = conf[valid]
        kpts1 = kpts1[valid]
        kpts2 = kpts2[matches[valid]]

        conf, sort_idx = conf.sort(descending=True)
        kpts1 = kpts1[sort_idx]
        kpts2 = kpts2[sort_idx]

        # 干脆不管3d2d都先进行这样的处理（利用上述过滤的kpts 进行映射变化）
        if kpts1.shape[0] < 4 or kpts2.shape[0] < 4:
            return image1, image2

        # 提取匹配的关键点
        kpts1_ = kpts1.numpy()
        kpts2_ = kpts2.numpy()

        # 计算单应性矩阵
        H, _ = cv2.findHomography(kpts1_, kpts2_, cv2.RANSAC, 3)
        if H is None:
            return image1, image2
        inverH = torch.pinverse(torch.Tensor(H)).numpy()
        # 使用单应性矩阵对img1进行变换，使其与img2对齐
        image1_cv = tensor2cv(image1)
        image2_cv = tensor2cv(image2)

        # 我直接先原图变化
        # 之后再统一resize 可以吗?
        _, h, w = image1.shape
        # w = 224
        # h = 224
        aligned_img1 = cv2.warpPerspective(image1_cv, H, (w, h))
        aligned_img2 = cv2.warpPerspective(image2_cv, inverH, (w, h))
        # 将cv转为tensor
        aligned_img1 = pil_to_tensor(
            Image.fromarray(cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2RGB))).float() / 255.0
        aligned_img2 = pil_to_tensor(
            Image.fromarray(cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2RGB))).float() / 255.0

        return aligned_img1, aligned_img2


def get_points(pred_3d, inp1, inp2, depth1, depth2, rs):
    kpts1, kpts2 = pred_3d['keypoints0'][0], pred_3d['keypoints1'][0]
    scale_1 = torch.tensor(inp1.shape[-2:]).flip(dims=(0,))
    scale_2 = torch.tensor(inp2.shape[-2:]).flip(dims=(0,))
    kpts1 /= scale_1
    kpts2 /= scale_2
    if kpts1.shape[0] < 4 or kpts2.shape[0] < 4:
        l = kpts1.shape[0] if kpts1.shape[0] < kpts2.shape[0] else kpts2.shape[0]
        return kpts1[:l], kpts2[:l]
    else:
        matches, conf = pred_3d['matches0'][0], pred_3d['matching_scores0'][0]

        valid = matches != -1
        conf = conf[valid]
        kpts1 = kpts1[valid]
        kpts2 = kpts2[matches[valid]]

        conf, sort_idx = conf.sort(descending=True)
        kpts1 = kpts1[sort_idx]
        kpts2 = kpts2[sort_idx]

        if kpts1.shape[0] < 4 or kpts2.shape[0] < 4:
            l = kpts1.shape[0] if kpts1.shape[0] < kpts2.shape[0] else kpts2.shape[0]
            return kpts1[:l], kpts2[:l]

        kpts1, kpts2 = filter_out_bad_correspondences_using_ransac(kpts1, kpts2, depth1, depth2, rs, )
        return kpts1, kpts2


class Test:
    def __init__(self):
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
        self.matching = Matching(config).eval()
        self._resize = K.augmentation.Resize(640, side="long")
        self._resize_ = K.augmentation.Resize((224, 224))

    def alignImage(self, batch):
        batch_points1 = []
        batch_points2 = []
        left2right = []
        right2left = []
        for i in range(len(batch['left_image'])):
            # 不管是2d或3d都需要进行查找keypoints 和 matching
            image1 = batch['left_image'][i]
            image2 = batch['right_image'][i]

            inp1 = K.color.rgb_to_grayscale(image1).unsqueeze(0)
            inp2 = K.color.rgb_to_grayscale(image2).unsqueeze(0)

            # # 先判断本地是否存放了对应的数据
            # transimagepath = os.path.join(batch["path"][i], "transformImages")
            # pointspath = os.path.join(batch["path"][i], "points")
            # if not os.path.exists(transimagepath):
            #     os.makedirs(transimagepath)
            # if not os.path.exists(pointspath):
            #     os.makedirs(pointspath)

            # index = batch["index"][i]
            # l2r_file = os.path.join(transimagepath, f"{index}_l2r.pt")
            # r2l_file = os.path.join(transimagepath, f"{index}_r2l.pt")
            # if os.path.isfile(l2r_file) and os.path.isfile(r2l_file):
            #     with open(l2r_file, "rb") as f:
            #         l2r = torch.load(f)
            #     with open(r2l_file, "rb") as f:
            #         r2l = torch.load(f)
            # else:
            #     with open(l2r_file, "wb") as f:
            #         torch.save(l2r, f)
            #     with open(r2l_file, "wb") as f:
            #         torch.save(r2l, f)
            # ------------------------------END----------------------------------

            # 判断单应性对齐图片
            with torch.no_grad():
                # -------------------这个地方耗时------------
                pred = self.matching({'image0': inp1, 'image1': inp2})
            # 单应性变化
            l2r, r2l = get_Homo_Images(pred, image1, image2)

            # # ---------------画图-----------------------
            # image1 = cv2.cvtColor(numpy.asarray(general.tensor_to_PIL(inp1_2d)), cv2.COLOR_RGB2BGR)
            # image2 = cv2.cvtColor(numpy.asarray(general.tensor_to_PIL(inp2_2d)), cv2.COLOR_RGB2BGR)
            #
            # img1_2 = cv2.cvtColor(np.asarray(general.tensor_to_PIL(l2r)), cv2.COLOR_RGB2BGR)
            # img2_1 = cv2.cvtColor(np.asarray(general.tensor_to_PIL(r2l)), cv2.COLOR_RGB2BGR)
            #
            # imags = [image1, img2_1, image2, img1_2]
            # ploting_image(imags, "test")
            # print("Test")
            # # -----------------END--------------------

            bicubic_resize = K.augmentation.Resize((224, 224), resample=2, keepdim=True)
            l2r = bicubic_resize(l2r)
            r2l = bicubic_resize(r2l)

            left2right.append(l2r)
            right2left.append(r2l)

            # 问题点：(他对单映射变化还有影响)
            inp1_3d = self._resize(inp1)
            inp2_3d = self._resize(inp2)

            with torch.no_grad():
                # -------------这个地方耗时------------------------
                pred_3d = self.matching({'image0': inp1_3d, 'image1': inp2_3d})

            # 获取图片3d相关信息
            kpts1, kpts2 = get_points(pred_3d, inp1_3d, inp2_3d, batch["depth1"][i], batch["depth2"][i],
                                      batch["registration_strategy"][i])

            batch_points1.append(kpts1)
            batch_points2.append(kpts2)

        # 向batch里面加入 points 和 warp后的
        batch["points1"] = batch_points1  # List
        batch["points2"] = batch_points2
        batch['left2right'] = left2right
        batch['right2left'] = right2left
        return batch


def ploting_image(imagelist, index):
    images = imagelist
    img_horizontal = cv2.hconcat(images)
    # 指定保存的路径
    # /disk/ygk/pycharm_project/The-Change-You-Want-to-See-main/imgs
    save_path = f'/disk/ygk/pycharm_project/The_Change_You_Want_to_See_main/imgs/kc3d_2/{index}.png'
    print(save_path)
    # 如果目录不存在，创建目录
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    cv2.imwrite(save_path, img_horizontal)
    pass


def inliers_using_ransac(X, Y, n_iters=500):
    best_inliers = None
    best_fit_error = None
    threshold = torch.median(torch.abs(Y - torch.median(Y)))
    for _ in range(n_iters):
        # estimate transformation
        sample_indices = np.random.choice(np.arange(X.shape[0]), size=min(50, X.shape[0]), replace=False)
        sample_X = X[sample_indices]
        sample_Y = Y[sample_indices]
        X_ = F.pad(sample_X, (0, 1), value=1)
        Y_ = F.pad(sample_Y, (0, 1), value=1)
        X_pinv = torch.linalg.pinv(X_)
        M = torch.einsum("ij,jk->ki", X_pinv, Y_)
        # find inliers
        X_warped = transform_points(M, X)
        fit_error = torch.sum(torch.abs(X_warped - Y), dim=1)
        inliers = (fit_error < threshold).nonzero().squeeze()
        fit_error_of_inliers = fit_error[inliers].sum()
        if (best_fit_error is None or fit_error_of_inliers < best_fit_error) and torch.numel(inliers) >= 10:
            best_fit_error = fit_error_of_inliers
            best_inliers = (fit_error < threshold).nonzero().squeeze()
    if best_inliers is None:
        return torch.ones(X.shape[0]).bool()
    return best_inliers


def filter_out_bad_correspondences_using_ransac(points1, points2, depth1=None, depth2=None, registration_strategy="", ):
    if registration_strategy == "3d":
        assert depth1 is not None and depth2 is not None
        X = convert_image_coordinates_to_world(
            image_coords=points1.unsqueeze(0),
            depth=sample_depth_for_given_points(depth1.unsqueeze(0), points1.unsqueeze(0)),
            K_inv=torch.eye(3).type_as(points1).unsqueeze(0),
            Rt=torch.eye(4).type_as(points1).unsqueeze(0),
        ).squeeze(0)
        Y = convert_image_coordinates_to_world(
            image_coords=points2.unsqueeze(0),
            depth=sample_depth_for_given_points(depth2.unsqueeze(0), points2.unsqueeze(0)),
            K_inv=torch.eye(3).type_as(points2).unsqueeze(0),
            Rt=torch.eye(4).type_as(points2).unsqueeze(0),
        ).squeeze(0)

    elif registration_strategy != "3d":
        X = points1
        Y = points2
    else:
        raise NotImplementedError()

    inliers = inliers_using_ransac(X, Y)
    points1 = points1[inliers]
    points2 = points2[inliers]
    return points1, points2


# ------------------------------END------------------------------------------

# 可以在这里使用 matches 和 confidence 进行后续处理，例如可视化匹配结果等
if __name__ == '__main__':
    img1 = []
    img2 = []
    for i in [57, 61]:
        path_1 = f'/home/ygk/disk/pycharm_project/The-Change-You-Want-to-See-main/imgs/kubric/_{i}/left_{i}.png'
        path_2 = f'/home/ygk/disk/pycharm_project/The-Change-You-Want-to-See-main/imgs/kubric/_{i}/right_{i}.png'
        image_tensor_scan = read_image_as_tensor(path_to_image=path_1)
        image_tensor_refer = read_image_as_tensor(path_to_image=path_2)
        img1.append(image_tensor_scan)
        img2.append(image_tensor_refer)

    img1_ = torch.stack(img1, dim=0)
    img2_ = torch.stack(img2, dim=0)

    batch = {'left_image': img1_, 'right_image': img2_, 'index': [57, 61]}
    test = Test()
    test.alignImage(batch)
    # ----------------------显示对齐后的图像---------------------
    save_path = f'/home/ygk/disk/pycharm_project/The-Change-You-Want-to-See-main/imgs/aaa/{i}.png'
    print(save_path)
    # 如果目录不存在，创建目录
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # cv2.imwrite(save_path, aI)
