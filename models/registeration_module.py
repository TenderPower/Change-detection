import kornia as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pytorch3d.renderer import AlphaCompositor, PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer, \
    PointsRenderer
from pytorch3d.structures import Pointclouds

import models.geometry
from utilssss.utils import fill_in_the_missing_information_


class FeatureRegisterationModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_warper = DifferentiableFeatureWarper()
        self.combine = CombinedModel(feature_dim)

    def register_3d_features(
            self,
            batch,
            features1,
            features2,
    ):
        using_camera_parameters = [x is not None for x in batch["intrinsics1"]]  # 3d的数据集中有自带相机参数的
        K_inv_1 = torch.zeros(len(batch["intrinsics1"]), 3, 3).type_as(features1)
        K_inv_2 = torch.zeros(len(batch["intrinsics2"]), 3, 3).type_as(features2)
        Rt_1_to_2 = torch.zeros(len(batch["intrinsics1"]), 4, 4).type_as(features1)
        Rt_2_to_1 = torch.zeros(len(batch["intrinsics2"]), 4, 4).type_as(features2)
        if sum(using_camera_parameters) > 0:
            _K_inv_1, _K_inv_2, _Rt_1_to_2, _Rt_2_to_1 = estimate_Rt_using_camera_parameters(
                torch.stack([batch["intrinsics1"][i] for i, x in enumerate(using_camera_parameters) if x]),
                torch.stack([batch["intrinsics2"][i] for i, x in enumerate(using_camera_parameters) if x]),
                torch.stack([batch["rotation1"][i] for i, x in enumerate(using_camera_parameters) if x]),
                torch.stack([batch["rotation2"][i] for i, x in enumerate(using_camera_parameters) if x]),
                torch.stack([batch["position1"][i] for i, x in enumerate(using_camera_parameters) if x]),
                torch.stack([batch["position2"][i] for i, x in enumerate(using_camera_parameters) if x]),
            )  # 参数:相机参数不为空的图像对 函数返回通过相机参数构成图像对的转换矩阵
            K_inv_1[using_camera_parameters] = _K_inv_1
            K_inv_2[using_camera_parameters] = _K_inv_2
            Rt_1_to_2[using_camera_parameters] = _Rt_1_to_2
            Rt_2_to_1[using_camera_parameters] = _Rt_2_to_1

        using_points = [x is None for x in batch["intrinsics1"]]  # 3d数据集中虽然有的图像对没有自带的相机参数,也可以通过points来找到图像对的之间的关系
        if sum(using_points) > 0:
            _K_inv_1, _K_inv_2, _Rt_1_to_2, _Rt_2_to_1 = estimate_Rt_using_points(
                [batch["points1"][i] for i, x in enumerate(using_points) if x],
                [batch["points2"][i] for i, x in enumerate(using_points) if x],
                batch["depth1"][using_points],
                batch["depth2"][using_points],
            )  # 利用点 计算出相应的变换矩阵

            K_inv_1[using_points] = _K_inv_1
            K_inv_2[using_points] = _K_inv_2
            Rt_1_to_2[using_points] = _Rt_1_to_2
            Rt_2_to_1[using_points] = _Rt_2_to_1

        nearest_resize = K.augmentation.Resize(features1.shape[-2:], resample=0, align_corners=None, keepdim=True)
        depth1 = nearest_resize(batch["depth1"])
        depth2 = nearest_resize(batch["depth2"])
        image1_warped_onto_image2 = self.feature_warper.warp(features1, depth1, K_inv_1, K_inv_2, Rt_1_to_2)  # 进行warp
        image2_warped_onto_image1 = self.feature_warper.warp(features2, depth2, K_inv_2, K_inv_1, Rt_2_to_1)

        return image1_warped_onto_image2, image2_warped_onto_image1

    def register_features(self, batch, image1, image2, strategy):
        # if strategy!="3d" and strategy!="other":
        #     # 先get depth
        #     # 后 3d 和 2d 融合
        #     pass
        if len(image1) == 0:
            return [], []
        b, c, h, w = image1.shape
        # 获取所有left坐标系下的feature
        half = c // 2
        Fleft = image1[:, :half, :, :]
        Fright2left = image1[:, half:, :, :]
        # 获取所有right坐标系下的feature
        Fright = image2[:, :half, :, :]
        Fleft2right = image2[:, half:, :, :]
        if strategy == "both":
            # 如果把3d和2d融为一体，那么 不管是2d还是3d都要求得该图的depth
            visibility = torch.ones((b, 1, h, w), requires_grad=False).type_as(image1)
            image1_warped_onto_image2, image2_warped_onto_image1 = self.register_3d_features(
                # 这个是重点
                batch, torch.cat([Fleft, visibility], dim=1), torch.cat([Fright, visibility], dim=1)
            )
            visibility1 = image1_warped_onto_image2[:, -1:, :, :]
            visibility2 = image2_warped_onto_image1[:, -1:, :, :]
            image1_warped_onto_image2 = image1_warped_onto_image2[:, :-1, :, :]  # 从warp之后的数据中分离出来
            image2_warped_onto_image1 = image2_warped_onto_image1[:, :-1, :, :]
            image1D1 = visibility2 * (Fleft - image2_warped_onto_image1)  # c/2
            image1D2 = Fleft - Fright2left  # c/2

            image2D1 = visibility1 * (Fright - image1_warped_onto_image2)
            image2D2 = Fright - Fleft2right

            # image1 = torch.cat((image1D1, image1D2), 1)  # c  # 重点!!!!!!!!! 实质就是对其融合
            # image2 = torch.cat((image2D1, image2D2), 1)
            #     先对D1，D2 分别进行注意力处理 后进行+ 得到总D
            combined_feature1 = self.combine(image1D1, image1D2)
            combined_feature2 = self.combine(image2D1, image2D2)
            #     用原特征拼接总D
            image1 = torch.cat((Fleft, combined_feature1), 1)  # c  # 重点!!!!!!!!! 实质就是对其融合
            image2 = torch.cat((Fright, combined_feature2), 1)

        return image1, image2

    def forward(self, batch, image1, image2):

        image1_, image2_ = self.register_features(batch, image1, image2, "both")

        return image1_, image2_

    # 实现原图的变换
    def test(self, batch, im1, im2):
        # reg_3d = [s == "3d" for s in batch["registration_strategy"]]
        # batch = slice_batch_given_bool_array(batch, reg_3d)
        self.register_features(batch, im1, im2, "both")
        pass


def estimate_Rt_using_camera_parameters(intrinsics1, intrinsics2, rotation1, rotation2, position1, position2):
    K_inv_1 = intrinsics1.inverse()
    K_inv_2 = intrinsics2.inverse()
    Rt_1_to_2, Rt_2_to_1 = models.geometry.get_relative_pose(  # 通过提供的相机参数,变换成相关的转换矩阵
        rotation1,
        rotation2,
        position1,
        position2,
        as_single_matrix=True,
    )
    return K_inv_1, K_inv_2, Rt_1_to_2, Rt_2_to_1


def estimate_Rt_using_points(points1, points2, depth1, depth2):
    K_inv, Rt = models.geometry.setup_canonical_cameras(len(points1),
                                                        tensor_to_infer_type_from=points1[0])  # 先虚拟构成出与相机参数相符的矩阵
    batch_points1_in_world_coordinates = []
    batch_points2_in_world_coordinates = []
    for i in range(len(points1)):
        points1_in_world_coordinates = models.geometry.convert_image_coordinates_to_world(  # 获取现实世界坐标系下的点坐标
            image_coords=points1[i].unsqueeze(0),
            depth=models.geometry.sample_depth_for_given_points(depth1[i].unsqueeze(0), points1[i].unsqueeze(0)),
            # 获取坐标对应的深度值
            K_inv=K_inv[i].unsqueeze(0),
            Rt=Rt[i].unsqueeze(0),
        ).squeeze(0)  # [n,3]
        points2_in_world_coordinates = models.geometry.convert_image_coordinates_to_world(
            image_coords=points2[i].unsqueeze(0),
            depth=models.geometry.sample_depth_for_given_points(depth2[i].unsqueeze(0), points2[i].unsqueeze(0)),
            K_inv=K_inv[i].unsqueeze(0),
            Rt=Rt[i].unsqueeze(0),
        ).squeeze(0)  # [n,3]
        batch_points1_in_world_coordinates.append(points1_in_world_coordinates)
        batch_points2_in_world_coordinates.append(points2_in_world_coordinates)
    Rt_1_to_2 = models.geometry.estimate_linear_warp(batch_points1_in_world_coordinates,
                                                     batch_points2_in_world_coordinates)  # 通过上述获得坐标点,计算出图像对变换矩阵
    Rt_2_to_1 = models.geometry.estimate_linear_warp(batch_points2_in_world_coordinates,
                                                     batch_points1_in_world_coordinates)
    return K_inv, K_inv, Rt_1_to_2, Rt_2_to_1


def slice_batch_given_bool_array(batch, mask):
    sliced_batch = {}
    for key in batch.keys():
        if "transform" in key:
            continue
        if isinstance(batch[key], list):
            sliced_batch[key] = [batch[key][i] for i in range(len(batch[key])) if mask[i]]
            if "bbox" in key or "point" in key:
                continue
            if len(sliced_batch[key]) > 0 and isinstance(sliced_batch[key][0], torch.Tensor):
                sliced_batch[key] = rearrange(sliced_batch[key], "... -> ...")  # 这句话 直接将list 转换成了Tensor
        else:
            sliced_batch[key] = batch[key][mask]
    return sliced_batch


class DifferentiableFeatureWarper(nn.Module):
    def __init__(self):
        super().__init__()

    def render(self, point_cloud, device, image_hw):  # 定义了一个渲染函数，用于将点云渲染成特征图像
        raster_settings = PointsRasterizationSettings(  # 配置点云渲染时的栅格化设置; 栅格化是将连续的点云数据转换为离散的像素表示的过程;
            image_size=image_hw,
            radius=float(1.5) / min(image_hw) * 2.0,
            bin_size=0,
            points_per_pixel=8,
        )
        canonical_cameras = PerspectiveCameras(  # 用于指定相机的参数。使用了单位旋转矩阵和零平移向量来定义一个基准相机
            R=rearrange(torch.eye(3), "r c -> 1 r c"),
            T=rearrange(torch.zeros(3), "n -> 1 n"),
        )
        canonical_rasterizer = PointsRasterizer(cameras=canonical_cameras,
                                                raster_settings=raster_settings)  # 利用上述初始化的设置,创建一个用户执行点云的栅格化操作 (栅格化器)
        canonical_renderer = PointsRenderer(rasterizer=canonical_rasterizer,
                                            compositor=AlphaCompositor())  # 创建一个组合器(渲染器),用于执行点云的渲染操作
        canonical_renderer.to(device)
        rendered_features = rearrange(canonical_renderer(point_cloud, eps=1e-5),
                                      "b h w c -> b c h w")  # 利用渲染器对点云进行渲染;通过 eps 参数来指定渲染的精度
        return rendered_features  # (b,c,w,h)(和之前的feature shape是一样的)

    def setup_given_cameras(self, batch):
        src_camera_K_inv = torch.linalg.inv(batch["intrinsics1"])
        dst_camera_K_inv = torch.linalg.inv(batch["intrinsics2"])
        src_camera_Rt = models.geometry.construct_Rt_matrix(batch["rotation1"], batch["position1"])
        dst_camera_Rt = models.geometry.construct_Rt_matrix(batch["rotation2"], batch["position2"])
        return src_camera_K_inv, dst_camera_K_inv, src_camera_Rt, dst_camera_Rt

    def warp(self, features_src, depth_src, src_camera_K_inv, dst_camera_K_inv,
             Rt_src_to_dst):  # 从一个相机视角中的特征和深度图，将它们转换到另一个相机视角的过程
        b, _, h, w = features_src.shape
        image_coords = rearrange(  # 生成图像坐标,一个网格坐标的张量，该网格覆盖了整个图像
            models.geometry.get_index_grid(h, w, batch=b, type_as=features_src),
            "b h w t -> b (h w) t",
        )
        src_points_in_dst_camera_coords = models.geometry.convert_world_to_image_coordinates(
            # 将src下图片变换成dst之后的三维坐标, 在dst的环境下进行变换,转换成图像坐标
            models.geometry.convert_image_coordinates_to_world(  # 将图像坐标转换为dst相机坐标系下的三维坐标。
                image_coords=image_coords,
                depth=rearrange(depth_src, "b h w -> b (h w)"),
                K_inv=src_camera_K_inv,
                Rt=Rt_src_to_dst,
            ),
            dst_camera_K_inv,
            torch.eye(4).unsqueeze(0).repeat(b, 1, 1).type_as(Rt_src_to_dst),
            keep_depth=True,
        )
        return self.render_features_from_points(src_points_in_dst_camera_coords, features_src)

    def render_features_from_points(self, points_in_3d, features):  # 从三维点云数据中渲染出特征图像
        b, _, h, w = features.shape
        src_point_cloud = Pointclouds(  # 将转换后的三维点云和特征数据封装成点云对象
            points=models.geometry.convert_to_pytorch3d_coordinate_system(points_in_3d),  # 将输入的三维点云坐标转换为 PyTorch3D 坐标系
            features=rearrange(features, "b c h w -> b (h w) c"),  # 其中 (h w) 表示图像的所有像素点
        )
        return self.render(src_point_cloud, features.device, (h, w))

    def render_warped_images_from_ground_truth_data(self, batch):
        (
            image1_camera_K_inv,
            image2_camera_K_inv,
            image1_camera_Rt,
            image2_camera_Rt,
        ) = self.setup_given_cameras(batch)
        Rt_1_to_2 = torch.einsum("bij,bjk->bik", torch.linalg.inv(image2_camera_Rt), image1_camera_Rt)
        Rt_2_to_1 = torch.einsum("bij,bjk->bik", torch.linalg.inv(image1_camera_Rt), image2_camera_Rt)
        image1_warped_onto_image2 = self.warp(batch["image1"], batch["depth1"], image1_camera_K_inv,
                                              image2_camera_K_inv, Rt_1_to_2)
        image2_warped_onto_image1 = self.warp(batch["image2"], batch["depth2"], image2_camera_K_inv,
                                              image1_camera_K_inv, Rt_2_to_1)
        return image1_warped_onto_image2, image2_warped_onto_image1


class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.fc1 = nn.Linear(feature_dim * 2, feature_dim // 2)
        self.fc2 = nn.Linear(feature_dim // 2, 2)

    def forward(self, feature2d, feature3d):
        # 全局平均池化
        pooled_feature2d = F.adaptive_avg_pool2d(feature2d, 1).view(feature2d.size(0), -1)
        pooled_feature3d = F.adaptive_avg_pool2d(feature3d, 1).view(feature3d.size(0), -1)

        # 连接2D和3D特征
        combined = torch.cat((pooled_feature2d, pooled_feature3d), dim=1)

        # 计算注意力权重
        attention_weights = F.softmax(self.fc2(F.relu(self.fc1(combined))), dim=1)
        return attention_weights


class CombinedModel(nn.Module):
    def __init__(self, feature_dim=512):
        super(CombinedModel, self).__init__()
        self.attention_module = AttentionModule(feature_dim//2)

    def forward(self, feature2d, feature3d):
        attention_weights = self.attention_module(feature2d, feature3d)

        # 加权特征
        weighted_feature2d = attention_weights[:, 0].view(-1, 1, 1, 1) * feature2d
        weighted_feature3d = attention_weights[:, 1].view(-1, 1, 1, 1) * feature3d

        combined_feature = weighted_feature2d + weighted_feature3d
        return combined_feature


if __name__ == '__main__':
    model = AttentionModule(feature_dim=768)
    input_2d = torch.randn(3, 768, 64, 64)  # 示例2D特征
    input_3d = torch.randn(3, 768, 64, 64)  # 示例3D特征

    output = model(input_2d, input_3d)
    print(output.shape)
