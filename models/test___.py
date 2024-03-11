import numpy
import torch
import os
import numpy as np
import kornia as K
import torch.nn as nn
import utils.general as general
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import AlphaCompositor, PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer, PointsRenderer
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.matching import SuperGlue
from SuperGluePretrainedNetwork.models.matching import Matching
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import cv2

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

    def alignImage(self, batch):
        # if 'registration_strategy' in batch:
        #     batch_points1 = []
        #     batch_points2 = []
        # else:
        left2right = []
        right2left = []
        for i in range(batch['left_image'].shape[0]):
            #     不管是2d或3d都需要进行查找keypoints 和 matching
            image1 = batch['left_image'][i]
            image2 = batch['right_image'][i]
            inp1 = K.color.rgb_to_grayscale(image1).unsqueeze(0)
            inp2 = K.color.rgb_to_grayscale(image2).unsqueeze(0)
            with torch.no_grad():
                pred = self.matching({'image0': inp1, 'image1': inp2})
            kpts1, kpts2 = pred['keypoints0'][0], pred['keypoints1'][0]

            if kpts1.shape[0] < 4 or kpts2.shape[0] < 4:
                left2right.append(image1)
                right2left.append(image2)

            else:
                matches, conf = pred['matches0'][0], pred['matching_scores0'][0]
                valid = matches != -1
                conf = conf[valid]
                kpts1 = kpts1[valid]
                kpts2 = kpts2[matches[valid]]

                conf, sort_idx = conf.sort(descending=True)
                kpts1 = kpts1[sort_idx]
                kpts2 = kpts2[sort_idx]
                # 3d--对齐
                if 'registration_strategy' in batch:
                    kpts1, kpts2 = filter_out_bad_correspondences_using_ransac(batch["registration_strategy"][i], kpts1,
                                                                           kpts2, batch["depth1"][i],
                                                                           batch["depth2"][i])

                    _K_inv_1, _K_inv_2, _Rt_1_to_2, _Rt_2_to_1 = estimate_Rt_using_points(kpts1,kpts2,batch["depth1"][i],batch["depth2"][i])  # 利用点 计算出相应的变换矩阵
                    # _K_inv_1, _K_inv_2, _Rt_1_to_2, _Rt_2_to_1 = estimate_Rt_using_camera_parameters(batch["intrinsics1"][i].unsqueeze(0),batch["intrinsics2"][i].unsqueeze(0),batch["rotation1"][i].unsqueeze(0),batch["rotation2"][i].unsqueeze(0),batch["position1"][i].unsqueeze(0),batch["position2"][i].unsqueeze(0))  # 利用点 计算出相应的变换矩阵

                    aligned_img1 = warp(image1.unsqueeze(0), batch["depth1"][i].unsqueeze(0), _K_inv_1, _K_inv_2, _Rt_1_to_2.unsqueeze(0))
                    aligned_img2 = warp(image2.unsqueeze(0), batch["depth2"][i].unsqueeze(0), _K_inv_2, _K_inv_1, _Rt_2_to_1.unsqueeze(0))

                else:
                    # 2d---对齐
                    if kpts1.shape[0] < 4 or kpts2.shape[0] < 4:
                        left2right.append(image1)
                        right2left.append(image2)
                        continue

                    # 提取匹配的关键点
                    kpts1 = kpts1.numpy()
                    kpts2 = kpts2.numpy()

                    # 计算单应性矩阵
                    H, _ = cv2.findHomography(kpts1, kpts2, cv2.RANSAC, 3)
                    if H is None:
                        left2right.append(image1)
                        right2left.append(image2)
                        continue
                    inverH = torch.pinverse(torch.Tensor(H)).numpy()
                    # 使用单应性矩阵对img1进行变换，使其与img2对齐
                    image1_cv = tensor2cv(image1)
                    image2_cv = tensor2cv(image2)
                    aligned_img1 = cv2.warpPerspective(image1_cv, H, (256, 256))
                    aligned_img2 = cv2.warpPerspective(image2_cv, inverH, (256, 256))

                    # 将cv转为tensor
                    aligned_img1 = pil_to_tensor(
                        Image.fromarray(cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2RGB))).float() / 255.0
                    aligned_img2 = pil_to_tensor(
                        Image.fromarray(cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2RGB))).float() / 255.0

                # 画图
                image1 = cv2.cvtColor(numpy.asarray(general.tensor_to_PIL(image1)), cv2.COLOR_RGB2BGR)
                image2 = cv2.cvtColor(numpy.asarray(general.tensor_to_PIL(image2)), cv2.COLOR_RGB2BGR)

                img1_2 = cv2.cvtColor(np.asarray(general.tensor_to_PIL(aligned_img1)), cv2.COLOR_RGB2BGR)
                img2_1 = cv2.cvtColor(np.asarray(general.tensor_to_PIL(aligned_img2)), cv2.COLOR_RGB2BGR)

                imags = [image1, img2_1, image2, img1_2]
                ploting_image(imags, batch["index"][i])


                left2right.append(aligned_img1)
                right2left.append(aligned_img2)

        # if 'registration_strategy' in batch:
        #     batch["points1"] = batch_points1
        #     batch["points2"] = batch_points2
        # else:
        left2right = torch.stack(left2right, dim=0)
        right2left = torch.stack(right2left, dim=0)

        batch['left2right'] = left2right
        batch['right2left'] = right2left
        return batch


def ploting_image(imagelist, index):
    images = imagelist
    img_horizontal = cv2.hconcat(images)
    # 指定保存的路径
    # /disk/ygk/pycharm_project/The-Change-You-Want-to-See-main/imgs
    save_path = f'/disk/ygk/pycharm_project/The-Change-You-Want-to-See-main/imgs/kc3d_2/{index}.png'
    print(save_path)
    # 如果目录不存在，创建目录
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    cv2.imwrite(save_path, img_horizontal)
    pass


# -------------------一些基础方法----------------------------------------
def setup_canonical_cameras(batch_size, tensor_to_infer_type_from):
    b = batch_size
    K_inv = torch.eye(3).unsqueeze(0).repeat(b, 1, 1).type_as(tensor_to_infer_type_from)  # [2,3,3]
    Rt = torch.eye(4).unsqueeze(0).repeat(b, 1, 1).type_as(tensor_to_infer_type_from)  # [2,4,4]
    return K_inv, Rt  # 先虚拟构成出与相机参数相符的矩阵

def convert_image_coordinates_to_world(image_coords, depth, K_inv, Rt):
    """
    Returns a point cloud of image coords projected into world.
    Note: image_coords must be of shape (b x n x 2), depth must be of shape (b x n)
    Output shape: (b x n x 3)
    """
    # convert to homogenous coordinates
    # 扩展一个维度，为后面乘深度做准备
    homogenous_coords = F.pad(image_coords, (0, 1), value=1)
    # multiply by the inverse of camera intrinsics
    # 将像素平面 转为 归一化平面
    camera_ref_coords = torch.einsum("bij,bnj->bni", K_inv, homogenous_coords)
    # introduce depth information
    # 获取物体在相机坐标系下的平面
    camera_ref_coords_with_depth = torch.einsum("bni,bn->bni", camera_ref_coords, depth)
    # convert 3d coordinates to 4d homogenous coordinates
    points_in_4d = F.pad(camera_ref_coords_with_depth, (0, 1), value=1)
    # multiply by the inverse of camera extrinsics
    points_in_world = torch.einsum("bij,bnj->bni", Rt, points_in_4d)
    # convert 4d -> 3d
    points_in_world = safe_division(points_in_world[:, :, :3], repeat(points_in_world[:, :, 3], "... -> ... n", n=3))
    return points_in_world

def sample_depth_for_given_points(depth_map, points):  # "depth_map" 是一个深度图; "points" 是一组二维坐标，表示在深度图中的某些点。
    depth_of_points = F.grid_sample(  # 根据输入的坐标从输入的深度图中提取对应的深度值
        rearrange(depth_map, "b h w -> b 1 h w"),
        rearrange(
            convert_to_grid_sample_coordinate_system(points),  # 将点的坐标转换成适合于 grid_sample 函数的格式
            "b n two -> b 1 n two",
        ),
    )
    return rearrange(depth_of_points, "b 1 1 n -> b n")

def estimate_linear_warp(X, Y):
    """
    Given X, Y, estimate a warp (rotation, translation) from X to Y using least squares.
    Note: shape of X, Y must be (b x n x 3)

    Returns: R (shape: b x 3 x 3), T (shape: b x 3).
    For inference: torch.einsum("bij,bnj->bni", R, X) + T
    """
    X_ = F.pad(X, (0, 1), value=1)
    Y_ = F.pad(Y, (0, 1), value=1)
    X_pinv = torch.linalg.pinv(X_)
    return torch.einsum("ij,jk->ki", X_pinv, Y_)

def construct_Rt_matrix(rotation, translation):
    Rt = torch.eye(4).type_as(rotation)
    Rt = Rt.unsqueeze(0).repeat(rotation.shape[0], 1, 1)
    Rt[:, :3, :3] = rotation
    Rt[:, :3, 3] = translation
    return Rt

def get_relative_pose(rotation_before, rotation_after, position_before, position_after, as_single_matrix=False):
    Rt_before = construct_Rt_matrix(rotation_before, position_before)  # 将rotation 和 position 结合 构成[4,4]矩阵
    Rt_after = construct_Rt_matrix(rotation_after, position_after)
    Rt_1_to_2 = torch.einsum("bij,bjk->bik", torch.linalg.inv(Rt_after),
                             Rt_before)  # 计算从一个相机姿态到另一个相机姿态的转换矩阵 表示了从第一个相机姿态到第二个相机姿态的转换。
    Rt_2_to_1 = torch.einsum("bij,bjk->bik", torch.linalg.inv(Rt_before), Rt_after)

    if as_single_matrix:
        return Rt_1_to_2, Rt_2_to_1

    rotation_from_1_to_2 = matrix_to_quaternion(Rt_1_to_2[:, :3, :3])
    rotation_from_2_to_1 = matrix_to_quaternion(Rt_2_to_1[:, :3, :3])
    translation_from_1_to_2 = Rt_1_to_2[:, :3, 3]
    translation_from_2_to_1 = Rt_2_to_1[:, :3, 3]
    return (
        rotation_from_1_to_2,
        rotation_from_2_to_1,
        translation_from_1_to_2,
        translation_from_2_to_1,
    )


def estimate_Rt_using_camera_parameters(intrinsics1, intrinsics2, rotation1, rotation2, position1, position2):
    K_inv_1 = intrinsics1.inverse()
    K_inv_2 = intrinsics2.inverse()
    Rt_1_to_2, Rt_2_to_1 = get_relative_pose(  # 通过提供的相机参数,变换成相关的转换矩阵
        rotation1,
        rotation2,
        position1,
        position2,
        as_single_matrix=True,
    )
    return K_inv_1, K_inv_2, Rt_1_to_2, Rt_2_to_1
def estimate_Rt_using_points(points1, points2, depth1, depth2):
    K_inv, Rt = setup_canonical_cameras(1, tensor_to_infer_type_from=points1)  # 先虚拟构成出与相机参数相符的矩阵

    points1_in_world_coordinates = convert_image_coordinates_to_world(# 获取现实世界坐标系下的点坐标
        image_coords=points1.unsqueeze(0),
        depth=sample_depth_for_given_points(depth1.unsqueeze(0), points1.unsqueeze(0)),
        # 获取坐标对应的深度值
        K_inv=K_inv,
        Rt=Rt,
    ).squeeze(0)  # [n,3]
    points2_in_world_coordinates = convert_image_coordinates_to_world(
        image_coords=points2.unsqueeze(0),
        depth=sample_depth_for_given_points(depth2.unsqueeze(0), points2.unsqueeze(0)),
        K_inv=K_inv,
        Rt=Rt,
    ).squeeze(0)  # [n,3]

    Rt_1_to_2 = estimate_linear_warp(points1_in_world_coordinates,points2_in_world_coordinates)  # 通过上述获得坐标点,计算出图像对变换矩阵
    Rt_2_to_1 = estimate_linear_warp(points2_in_world_coordinates,points1_in_world_coordinates) #[4,4]
    return K_inv, K_inv, Rt_1_to_2, Rt_2_to_1


def get_index_grid(height, width, batch=None, type_as=None):
    y, x = torch.linspace(0, 1, height), torch.linspace(0, 1, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    index_grid = rearrange([yy, xx], "two y x -> y x two")
    index_grid[:, :, [0, 1]] = index_grid[:, :, [1, 0]]
    if batch is not None:
        index_grid = repeat(index_grid, "y x two -> b y x two", b=batch)
    if type_as is None:
        return index_grid
    return index_grid.type_as(type_as)

def convert_world_to_image_coordinates(world_points, K_inv, Rt, keep_depth):
    """
    Given 3d world coordinates, convert them into image coordinates.
    Note: world_points must be of shape (b x n x 3)
    Output shape: (b x n x 2) if keep_depth=False, else (b x n x 3)
    """
    b = K_inv.shape[0]
    # compute camera projection matrix
    homogenous_intrinsics = torch.zeros((b, 3, 4)).type_as(K_inv)  # 先进行初始化
    homogenous_intrinsics[:, :, :3] = torch.linalg.inv(K_inv)  # 将K_inv进行逆运算 变成了 K
    camera_projection_matrix = torch.einsum("bij,bjk->bik", homogenous_intrinsics, torch.linalg.inv(Rt))  # shape: bx3x4 #结合相机内参和相机的外参逆矩阵,得到相机投影矩阵
    # project world points onto the image plane
    homogenous_coords = F.pad(world_points, (0, 1), value=1) #(b,_,4)
    camera_ref_coords_with_depth = torch.einsum("bij,bnj->bni", camera_projection_matrix, homogenous_coords)# 使用相机投影矩阵将齐次坐标系下的世界坐标转换为相机参考坐标系下的坐标 #(b,_,3)
    if keep_depth:
        return camera_ref_coords_with_depth # 包含深度信息的相机参考坐标系下的坐标
    # convert 3d -> 2d 将包含深度信息的相机参考坐标系下的坐标转换为图像坐标 即将 x 和 y 分量分别除以 z 分量
    image_coords = safe_division(
        camera_ref_coords_with_depth[:, :, :2],
        repeat(camera_ref_coords_with_depth[:, :, 2], "... -> ... n", n=2),
    )
    return image_coords

def convert_to_pytorch3d_coordinate_system(points):
    xy = points[:, :, :2]
    z = points[:, :, 2]
    xy = safe_division(xy, repeat(z, "... -> ... n", n=2))
    xy = 1 - (2 * xy)
    xyz = torch.einsum("bni,bn->bni", F.pad(xy, (0, 1), value=1), z)
    return xyz


def render(point_cloud, device, image_hw): # 定义了一个渲染函数，用于将点云渲染成特征图像
    raster_settings = PointsRasterizationSettings(#配置点云渲染时的栅格化设置; 栅格化是将连续的点云数据转换为离散的像素表示的过程;
        image_size=image_hw,
        radius=float(1.5) / min(image_hw) * 2.0,
        bin_size=0,
        points_per_pixel=8,
    )
    canonical_cameras = PerspectiveCameras(#用于指定相机的参数。使用了单位旋转矩阵和零平移向量来定义一个基准相机
        R=rearrange(torch.eye(3), "r c -> 1 r c"),
        T=rearrange(torch.zeros(3), "n -> 1 n"),
    )
    canonical_rasterizer = PointsRasterizer(cameras=canonical_cameras, raster_settings=raster_settings) #利用上述初始化的设置,创建一个用户执行点云的栅格化操作 (栅格化器)
    canonical_renderer = PointsRenderer(rasterizer=canonical_rasterizer, compositor=AlphaCompositor()) #创建一个组合器(渲染器),用于执行点云的渲染操作
    canonical_renderer.to(device)
    # 这里太费时间了
    rendered_features = rearrange(canonical_renderer(point_cloud, eps=1e-5), "b h w c -> b c h w")#利用渲染器对点云进行渲染;通过 eps 参数来指定渲染的精度
    return rendered_features.squeeze() #(b,c,w,h)(和之前的feature shape是一样的)

def render_features_from_points(points_in_3d, features): #从三维点云数据中渲染出特征图像
    b, _, h, w = features.shape
    src_point_cloud = Pointclouds(#将转换后的三维点云和特征数据封装成点云对象
        points=convert_to_pytorch3d_coordinate_system(points_in_3d),#将输入的三维点云坐标转换为 PyTorch3D 坐标系
        features=rearrange(features, "b c h w -> b (h w) c"),#其中 (h w) 表示图像的所有像素点
    )
    return render(src_point_cloud, features.device, (h, w))
def warp(features_src, depth_src, src_camera_K_inv, dst_camera_K_inv,Rt_src_to_dst):
    # 从一个相机视角中的特征和深度图，将它们转换到另一个相机视角的过程
    b, _, h, w = features_src.shape
    image_coords = rearrange(  # 生成图像坐标,一个网格坐标的张量，该网格覆盖了整个图像
        get_index_grid(h, w, b,  type_as=features_src),
        "b h w t -> b (h w) t",
    )
    src_points_in_dst_camera_coords = convert_world_to_image_coordinates(
        # 将src下图片变换成dst之后的三维坐标, 在dst的环境下进行变换,转换成图像坐标
        convert_image_coordinates_to_world(  # 将图像坐标转换为dst相机坐标系下的三维坐标。
            image_coords=image_coords,
            depth=rearrange(depth_src, "b h w -> b (h w)"),
            K_inv=src_camera_K_inv,
            Rt=Rt_src_to_dst,
        ),
        dst_camera_K_inv,
        torch.eye(4).unsqueeze(0).repeat(b, 1, 1).type_as(Rt_src_to_dst),
        keep_depth=True,
    )
    return render_features_from_points(src_points_in_dst_camera_coords, features_src)

def safe_division(numerator, denominator):
    sign = torch.sign(denominator)
    sign[sign == 0] = 1
    return numerator / (
            sign
            * torch.maximum(
        torch.abs(denominator),
        1e-5 * torch.ones(denominator.shape).type_as(denominator),
    )
    )


def convert_image_coordinates_to_world(image_coords, depth, K_inv, Rt):
    """
    Returns a point cloud of image coords projected into world.
    Note: image_coords must be of shape (b x n x 2), depth must be of shape (b x n)
    Output shape: (b x n x 3)
    """
    # convert to homogenous coordinates
    # 扩展一个维度，为后面乘深度做准备
    homogenous_coords = F.pad(image_coords, (0, 1), value=1)
    # multiply by the inverse of camera intrinsics
    # 将像素平面 转为 归一化平面
    camera_ref_coords = torch.einsum("bij,bnj->bni", K_inv, homogenous_coords)
    # introduce depth information
    # 获取物体在相机坐标系下的平面
    camera_ref_coords_with_depth = torch.einsum("bni,bn->bni", camera_ref_coords, depth)
    # convert 3d coordinates to 4d homogenous coordinates
    points_in_4d = F.pad(camera_ref_coords_with_depth, (0, 1), value=1)
    # multiply by the inverse of camera extrinsics
    points_in_world = torch.einsum("bij,bnj->bni", Rt, points_in_4d)
    # convert 4d -> 3d
    points_in_world = safe_division(points_in_world[:, :, :3], repeat(points_in_world[:, :, 3], "... -> ... n", n=3))
    return points_in_world


def convert_to_grid_sample_coordinate_system(points):
    return 2 * points - 1


def sample_depth_for_given_points(depth_map, points):  # "depth_map" 是一个深度图; "points" 是一组二维坐标，表示在深度图中的某些点。
    depth_of_points = F.grid_sample(  # 根据输入的坐标从输入的深度图中提取对应的深度值
        rearrange(depth_map, "b h w -> b 1 h w"),
        rearrange(
            convert_to_grid_sample_coordinate_system(points),  # 将点的坐标转换成适合于 grid_sample 函数的格式
            "b n two -> b 1 n two",
        ),
    )
    return rearrange(depth_of_points, "b 1 1 n -> b n")


def transform_points(transformation_matrix, points, keep_depth=False):
    """
    Transforms points with a transformation matrix.
    """
    shape = points.shape
    if len(shape) == 2:
        transformation_matrix = transformation_matrix.unsqueeze(0)
        points = points.unsqueeze(0)
    points = F.pad(points, (0, 1), value=1)
    points = torch.einsum("bij,bnj->bni", transformation_matrix, points)
    if keep_depth:
        if len(shape) == 2:
            points = points.squeeze(0)
        return points
    points = safe_division(
        points[:, :, :-1],
        repeat(points[:, :, -1], "... -> ... n", n=points.shape[-1] - 1),
    )
    if len(shape) == 2:
        points = points.squeeze(0)
    return points


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


def filter_out_bad_correspondences_using_ransac(registration_strategy, points1, points2, depth1=None, depth2=None):
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
    for i in [44, 57, 61, 70, 71, 72, 94, 146, 148, 150, 156, 179, 180, 181, 210, 215, 216, 217, 232, 264, 265, 328,
              329, 330, 351, 395, 397, 414, 482, 483, 569, 570, 571, 646, 689, 692, 693, 694, 725, 727, 732, 742, 743,
              744, 807, 819, 843, 860, 878, 975, 1000, 1003, 1004, 1028, 1029, 1036, 1051, 1053, 1072, 1088, 1089, 1090,
              1091, 1092, 1108, 1133, 1134, 1149, 1170, 1202, 1208, 1255, 1258, 1312, 1313, 1314, 1315, 1316, 1328,
              1330, 1331, 1372, 1374, 1404, 1457, 1458, 1511, 1514, 1515, 1528, 1529, 1577, 1579, 1602]:
        path_1 = f'/home/ygk/disk/pycharm_project/The-Change-You-Want-to-See-main/imgs/kubric/_{i}/left_{i}.png'
        path_2 = f'/home/ygk/disk/pycharm_project/The-Change-You-Want-to-See-main/imgs/kubric/_{i}/right_{i}.png'
        image_tensor_scan = read_image_as_tensor(path_to_image=path_1)
        image_tensor_refer = read_image_as_tensor(path_to_image=path_2)
        img1.append(image_tensor_scan)
        img2.append(image_tensor_refer)

    img1_ = torch.stack(img1, dim=0)
    img2_ = torch.stack(img2, dim=0)

    batch = {'left_image': img1_, 'right_image': img2_}
    test = Test()
    test.alignImage(batch)
    # ----------------------显示对齐后的图像---------------------
    save_path = f'/home/ygk/disk/pycharm_project/The-Change-You-Want-to-See-main/imgs/aaa/{i}.png'
    print(save_path)
    # 如果目录不存在，创建目录
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # cv2.imwrite(save_path, aI)
