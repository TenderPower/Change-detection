import kornia as K
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import copy
from detectron2.structures.image_list import ImageList
from easydict import EasyDict
from loguru import logger as L
from mmdet.models.dense_heads.centernet_head import CenterNetHead
from pytorch_lightning.utilities import rank_zero_only
from einops import rearrange
from data.datamodule import DataModule
from models.unet import Unet
from utilssss.utils import fill_in_the_missing_information
from utilssss.voc_eval import BoxList, eval_detection_voc
from models.middle import BackLayers, FrontLayers
from models.test__ import alignImage
import utilssss.geometry as geometry

plt.ioff()
import cv2
import utilssss.general as general

import utilssss.alignment as algin
from PIL import Image, PngImagePlugin
from torchvision.transforms.functional import pil_to_tensor


class CenterNetWithCoAttention(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.test_set_names = [test_set.name for test_set in args.datasets.test_datasets]
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        number_of_coam_layers, number_of_post_layers, number_of_middle_layers, number_of_front_layers, \
            coam_input_channels, coam_hidden_channels, sizes = args.coam_layer_data
        self.numberPost = number_of_post_layers
        self.unet_model = Unet(
            args.encoder,
            decoder_channels=(256, 256, 128, 128, 64),
            encoder_depth=5,
            in_channels=6,
            encoder_weights="imagenet",
            num_coam_layers=number_of_coam_layers,
            decoder_attention_type=args.decoder_attention,
            disable_segmentation_head=True,
            fix_dim=True,
            half_dim=False,
        )
        self.backLayers = nn.ModuleList(
            [
                BackLayers(coam_input_channels[i] // 2) for i in range(number_of_post_layers)
            ]
        )
        self.frontLayers = nn.ModuleList(
            [
                FrontLayers() for _ in range(number_of_post_layers, number_of_coam_layers)
            ]
        )

        self.centernet_head = CenterNetHead(
            in_channel=64,
            feat_channel=64,
            num_classes=1,
            test_cfg=EasyDict({"topk": 100, "local_maximum_kernel": 3, "max_per_img": 100}),
        )
        self.centernet_head.init_weights()
        if args.load_weights_from is not None:
            self.safely_load_state_dict(torch.load(args.load_weights_from)["state_dict"])

    def safely_load_state_dict(self, checkpoint_state_dict):
        model_state_dict = self.state_dict()
        for k in checkpoint_state_dict:
            if k in model_state_dict:
                if checkpoint_state_dict[k].shape != model_state_dict[k].shape:
                    L.log(
                        "INFO",
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {checkpoint_state_dict[k].shape}",
                    )
                    checkpoint_state_dict[k] = model_state_dict[k]
            else:
                L.log("INFO", f"Dropping parameter {k}")
        self.load_state_dict(checkpoint_state_dict, strict=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("CenterNetWithCoAttention")
        parser.add_argument("--lr", type=float)
        parser.add_argument("--weight_decay", type=float)
        parser.add_argument("--encoder", type=str, choices=["resnet50", "resnet18"])
        parser.add_argument("--coam_layer_data", nargs="+", type=int)
        parser.add_argument("--attention", type=str)
        parser.add_argument("--decoder_attention", type=str, default=None)
        return parent_parser

    def training_step(self, batch, batch_idx):
        left_image_outputs, right_image_outputs = self(batch)
        left_losses = self.centernet_head.loss(
            *left_image_outputs,
            batch["left_image_target_bboxes"],
            batch["target_bbox_labels"],
            img_metas=batch["query_metadata"],
        )
        right_losses = self.centernet_head.loss(
            *right_image_outputs,
            batch["right_image_target_bboxes"],
            batch["target_bbox_labels"],
            img_metas=batch["query_metadata"],
        )
        overall_loss = 0
        for key in left_losses:
            self.log(
                f"train/{key}", left_losses[key] + right_losses[key], on_step=True, on_epoch=True
            )
            overall_loss += left_losses[key] + right_losses[key]
        self.log("train/overall_loss", overall_loss, on_step=True, on_epoch=True)
        return overall_loss

    def validation_step(self, batch, batch_index):
        left_image_outputs, right_image_outputs = self(batch)
        left_losses = self.centernet_head.loss(
            *left_image_outputs,
            batch["left_image_target_bboxes"],
            batch["target_bbox_labels"],
            img_metas=batch["query_metadata"],
        )
        right_losses = self.centernet_head.loss(
            *right_image_outputs,
            batch["right_image_target_bboxes"],
            batch["target_bbox_labels"],
            img_metas=batch["query_metadata"],
        )
        overall_loss = 0
        for key in left_losses:
            self.log(f"val/{key}", left_losses[key] + right_losses[key], on_epoch=True)
            overall_loss += left_losses[key] + right_losses[key]
        self.log("val/overall_loss", overall_loss, on_epoch=True)
        left_predicted_bboxes = self.centernet_head.get_bboxes(
            *left_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        right_predicted_bboxes = self.centernet_head.get_bboxes(
            *right_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        return (
            [
                [bboxes.cpu(), classification.cpu()]
                for bboxes, classification in left_predicted_bboxes
            ],
            [
                [bboxes.cpu(), classification.cpu()]
                for bboxes, classification in right_predicted_bboxes
            ],
            [bboxes.cpu() for bboxes in batch["left_image_target_bboxes"]],
            [bboxes.cpu() for bboxes in batch["right_image_target_bboxes"]],
        )

    def validation_epoch_end(self, multiple_test_set_outputs):
        """
        Test set evaluation function.
        """

        multiple_test_set_outputs = [multiple_test_set_outputs]
        # iterate over all the test sets
        for test_set_name, test_set_batch_outputs in zip(
                ["cocoval"], multiple_test_set_outputs
        ):
            predicted_bboxes = []
            target_bboxes = []
            # iterate over all the batches for the current test set
            for test_set_outputs in test_set_batch_outputs:
                (
                    left_predicted_bboxes,
                    right_predicted_bboxes,
                    left_target_bboxes,
                    right_target_bboxes,
                ) = test_set_outputs
                # iterate over all predictions for images
                for bboxes_per_side in [left_predicted_bboxes, right_predicted_bboxes]:
                    for bboxes_per_image in bboxes_per_side:
                        # filter out background bboxes
                        bboxes_per_image = bboxes_per_image[0][bboxes_per_image[1] == 0]
                        bbox_list = BoxList(
                            bboxes_per_image[:, :4],
                            image_size=(256, 256),
                            mode="xyxy",
                        )
                        bbox_list.add_field("scores", bboxes_per_image[:, 4])
                        bbox_list.add_field("labels", torch.ones(bboxes_per_image.shape[0]))
                        predicted_bboxes.append(bbox_list)
                # iterate over all targets for images
                for bboxes_per_side in [left_target_bboxes, right_target_bboxes]:
                    for bboxes_per_image in bboxes_per_side:
                        bbox_list = BoxList(
                            bboxes_per_image,
                            image_size=(256, 256),
                            mode="xyxy",
                        )
                        bbox_list.add_field("labels", torch.ones(bboxes_per_image.shape[0]))
                        bbox_list.add_field("difficult", torch.zeros(bboxes_per_image.shape[0]))
                        target_bboxes.append(bbox_list)
            # compute metrics
            ap_map_precision_recall = eval_detection_voc(
                predicted_bboxes, target_bboxes, iou_thresh=0.5
            )
            self.log(f'{test_set_name}_AP', ap_map_precision_recall['ap'][1], on_epoch=True)
            L.log("INFO",
                  f"{test_set_name} AP: {ap_map_precision_recall['ap']}, mAP: {ap_map_precision_recall['map']}", )

    def test_step(self, batch, batch_index, dataloader_index=0):
        left_image_outputs, right_image_outputs = self(batch)
        left_predicted_bboxes = self.centernet_head.get_bboxes(
            *left_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        right_predicted_bboxes = self.centernet_head.get_bboxes(
            *right_image_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        return (
            [
                [bboxes.cpu(), classification.cpu()]
                for bboxes, classification in left_predicted_bboxes
            ],
            [
                [bboxes.cpu(), classification.cpu()]
                for bboxes, classification in right_predicted_bboxes
            ],
            [bboxes.cpu() for bboxes in batch["left_image_target_bboxes"]],
            [bboxes.cpu() for bboxes in batch["right_image_target_bboxes"]],
        )

    def test_epoch_end(self, multiple_test_set_outputs):
        """
        Test set evaluation function.
        """
        if len(self.test_set_names) == 1:
            multiple_test_set_outputs = [multiple_test_set_outputs]
        # iterate over all the test sets
        for test_set_name, test_set_batch_outputs in zip(
                self.test_set_names, multiple_test_set_outputs
        ):
            predicted_bboxes = []
            target_bboxes = []
            # iterate over all the batches for the current test set
            for test_set_outputs in test_set_batch_outputs:
                (
                    left_predicted_bboxes,
                    right_predicted_bboxes,
                    left_target_bboxes,
                    right_target_bboxes,
                ) = test_set_outputs
                # iterate over all predictions for images
                for bboxes_per_side in [left_predicted_bboxes, right_predicted_bboxes]:
                    for bboxes_per_image in bboxes_per_side:
                        # filter out background bboxes
                        bboxes_per_image = bboxes_per_image[0][bboxes_per_image[1] == 0]
                        bbox_list = BoxList(
                            bboxes_per_image[:, :4],
                            image_size=(256, 256),
                            mode="xyxy",
                        )
                        bbox_list.add_field("scores", bboxes_per_image[:, 4])
                        bbox_list.add_field("labels", torch.ones(bboxes_per_image.shape[0]))
                        predicted_bboxes.append(bbox_list)
                # iterate over all targets for images
                for bboxes_per_side in [left_target_bboxes, right_target_bboxes]:
                    for bboxes_per_image in bboxes_per_side:
                        bbox_list = BoxList(
                            bboxes_per_image,
                            image_size=(256, 256),
                            mode="xyxy",
                        )
                        bbox_list.add_field("labels", torch.ones(bboxes_per_image.shape[0]))
                        bbox_list.add_field("difficult", torch.zeros(bboxes_per_image.shape[0]))
                        target_bboxes.append(bbox_list)
            # compute metrics
            ap_map_precision_recall = eval_detection_voc(
                predicted_bboxes, target_bboxes, iou_thresh=0.5
            )
            L.log(
                "INFO",
                f"{test_set_name} AP: {ap_map_precision_recall['ap']}, mAP: {ap_map_precision_recall['map']}",
            )

    def configure_optimizers(self):
        optimizer_params = [
            {"params": [parameter for parameter in self.parameters() if parameter.requires_grad]}
        ]
        optimizer = torch.optim.Adam(optimizer_params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, batch):
        left_images = batch['left_image']
        right_images = batch['right_image']
        l2r = batch['left2right']
        r2l = batch['right2left']

        # 将图片1与对齐后的图片2 进行拼接
        imag_one = torch.concat((left_images, r2l), 1)
        imag_two = torch.concat((right_images, l2r), 1)

        # 将通道数为6的图片，放入到encoder里面
        imag_one_encoded_features = self.unet_model.encoder(imag_one)
        imag_two_encoded_features = self.unet_model.encoder(imag_two)

        # 将encoder后的features进行处理----backlayers
        for i in range(len(self.backLayers)):
            (imag_one_encoded_features[-(i + 1)],
             imag_two_encoded_features[-(i + 1)]) = self.backLayers[i](
                imag_one_encoded_features[-(i + 1)], imag_two_encoded_features[-(i + 1)])

        # 将encoder后的features进行处理----frontlayers
        for i in range(len(self.frontLayers)):
            (imag_one_encoded_features[-(i + 1 + self.numberPost)],
             imag_two_encoded_features[-(i + 1 + self.numberPost)]) = self.frontLayers[i](
                imag_one_encoded_features[-(i + 1 + self.numberPost)],
                imag_two_encoded_features[-(i + 1 + self.numberPost)])

        left_image_decoded_features = self.unet_model.decoder(*imag_one_encoded_features)
        right_image_decoded_features = self.unet_model.decoder(*imag_two_encoded_features)
        return (
            self.centernet_head([left_image_decoded_features]),
            self.centernet_head([right_image_decoded_features]),
        )


def resize_other(batch, reg_other):
    for i in range(len(batch['left_image'])):
        # 说明是2d相关的resize方式
        if reg_other[i]:
            (
                batch["left_image"][i],
                target_region_and_annotations,
            ) = geometry.resize_image_and_annotations(
                batch["left_image"][i],
                output_shape_as_hw=(224, 224),
                annotations=batch["image1_target_annotations"][i],
            )
            batch["image1_target_annotations"][i] = target_region_and_annotations
            (
                batch["right_image"][i],
                target_region_and_annotations,
            ) = geometry.resize_image_and_annotations(
                batch["right_image"][i],
                output_shape_as_hw=(224, 224),
                annotations=batch["image2_target_annotations"][i],
            )
            batch["image2_target_annotations"][i] = target_region_and_annotations
            image1_target_bboxes = torch.Tensor([x["bbox"] for x in batch["image1_target_annotations"][i]])
            image2_target_bboxes = torch.Tensor([x["bbox"] for x in batch["image2_target_annotations"][i]])
            batch["left_image_target_bboxes"][i] = image1_target_bboxes
            batch["right_image_target_bboxes"][i] = image2_target_bboxes
            batch["target_bbox_labels"][i] = torch.zeros(len(image1_target_bboxes)).long()

    return batch


def resize_3d(batch, reg_3d):
    nearest_resize = K.augmentation.Resize((224, 224), resample=0, align_corners=None, keepdim=True)
    bicubic_resize = K.augmentation.Resize((224, 224), resample=2, keepdim=True)
    for i in range(len(batch['left_image'])):
        # 说明是2d相关的resize方式
        if reg_3d[i]:
            original_hw1 = batch["left_image"][i].shape[-2:]
            original_hw2 = batch["right_image"][i].shape[-2:]
            # 我先不进行图像的正则化 （因为，对3d进行正则化，还需要对2d进行正则化）
            batch["image1_target_annotations"][i] = resize_bbox(batch["image1_target_annotations"][i], original_hw1,
                                                                (224, 224))
            batch["image2_target_annotations"][i] = resize_bbox(batch["image2_target_annotations"][i], original_hw2,
                                                                (224, 224))

            batch["left_image"] = bicubic_resize(batch["left_image"][i])
            batch["right_image"] = bicubic_resize(batch["right_image"][i])

            if batch["depth1"][i] is not None:
                original_depth_hw1 = batch["depth1"][i].shape[-2:]
                batch["depth1"][i] = nearest_resize(batch["depth1"][i])
            if batch["depth2"][i] is not None:
                original_depth_hw2 = batch["depth2"][i].shape[-2:]
                batch["depth2"][i] = nearest_resize(batch["depth2"][i])
            if batch["intrinsics1"][i] is not None:
                assert original_hw1 == original_depth_hw1
                transformation = nearest_resize.transform_matrix.squeeze()
                transformation = convert_kornia_transformation_matrix_to_normalised_coordinates(transformation,
                                                                                                original_hw1,
                                                                                                (224, 224))
                batch["intrinsics1"][i] = transformation @ batch["intrinsics1"][i]
            if batch["intrinsics2"][i] is not None:
                assert original_hw2 == original_depth_hw2
                transformation = nearest_resize.transform_matrix.squeeze()
                transformation = convert_kornia_transformation_matrix_to_normalised_coordinates(transformation,
                                                                                                original_hw2,
                                                                                                (224, 224))
                batch["intrinsics2"][i] = transformation @ batch["intrinsics2"][i]
    return batch


def prepare_batch_for_model_all(batch):
    # 前提已经统一了标准
    # batch标准的
    reg_3d = [s == "3d" for s in batch["registration_strategy"]]
    reg_other = [s != "3d" for s in batch["registration_strategy"]]
    batch = resize_other(batch, reg_other)
    batch = resize_3d(batch, reg_3d)
    return batch


def marshal_getitem_data(data, split):
    # 对不同数据集中的数据进行统一格式
    """
    The data field above is returned by the individual datasets.
    This function marshals that data into the format expected by this
    model/method.
    """

    def create_sanity_data(data):
        data_ = {}
        all_keys = [
            "image1",
            "image2",
            "depth1",
            "depth2",
            "image1_target_annotations",
            "image2_target_annotations",
            "intrinsics1",
            "intrinsics2",
            "position1",
            "position2",
            "rotation1",
            "rotation2",
            "registration_strategy",
        ]
        for key in all_keys:
            value = data.get(key, None)
            if value is None:
                data_[key] = None
                continue
            data_[key] = value
        return data_

    # 统一格式
    data = create_sanity_data(data)

    return {
        "left_image": data["image1"],
        "right_image": data["image2"],
        "left_image_target_bboxes": None,
        "right_image_target_bboxes": None,
        "target_bbox_labels": None,
        "image1_target_annotations": data["image1_target_annotations"],
        "image2_target_annotations": data["image2_target_annotations"],
        "registration_strategy": data['registration_strategy'],
        "depth1": data["depth1"],
        "depth2": data["depth2"],
        "intrinsics1": data["intrinsics1"],
        "intrinsics2": data["intrinsics2"],
        "position1": data["position1"],
        "position2": data["position2"],
        "rotation1": data["rotation1"],
        "rotation2": data["rotation2"],
        "query_metadata": {
            "pad_shape": data["image1"].shape[-2:],
            "border": np.array([0, 0, 0, 0]),
            "batch_input_shape": data["image1"].shape[-2:],
        },
    }
    # -------------------END---------------------------
    # if split in ["train", "val", "test"]:
    #     # 不管是3d还是2d  都进行batch的参数补全
    #
    #     # Resize
    #     if data['registration_strategy'] == "3d":
    #         #     进行resize--貌似不需要
    #         data = prepare_batch_for_model(data)
    #     # 对2d图片进行resize
    #     else:
    #         # 对已有的depth进行resize
    #         data = resize_depth(data)
    #         (
    #             data["image1"],
    #             target_region_and_annotations,
    #         ) = geometry.resize_image_and_annotations(
    #             data["image1"],
    #             output_shape_as_hw=(224, 224),
    #             annotations=data["image1_target_annotations"],
    #         )
    #         data["image1_target_annotations"] = target_region_and_annotations
    #         (
    #             data["image2"],
    #             target_region_and_annotations,
    #         ) = geometry.resize_image_and_annotations(
    #             data["image2"],
    #             output_shape_as_hw=(224, 224),
    #             annotations=data["image2_target_annotations"],
    #         )
    #         data["image2_target_annotations"] = target_region_and_annotations
    #
    # assert data["image1"].shape == data["image2"].shape
    # if data['registration_strategy'] == "3d":
    #     image1_target_bboxes = data["image1_target_annotations"]
    #     image2_target_bboxes = data["image2_target_annotations"]
    # else:
    #     image1_target_bboxes = torch.Tensor([x["bbox"] for x in data["image1_target_annotations"]])
    #     image2_target_bboxes = torch.Tensor([x["bbox"] for x in data["image2_target_annotations"]])
    #
    # if len(image1_target_bboxes) != len(image2_target_bboxes) or len(image1_target_bboxes) == 0:
    #     return None
    #
    # # 图片变化--使用的是传统对齐
    # # image1_to_image2 = alignimage(data["image1"], data["image2"])
    # # image2_to_image1 = alignimage(data["image2"], data["image1"])
    # # 使用模型
    # # image1_to_image2, image2_to_image1 = alignImage(data["image1"], data["image2"])
    #
    # # ----------------------既然开头已经把key都补全了，直接返回补全的就行--------------------------
    # return {
    #     "left_image": data["image1"],
    #     "right_image": data["image2"],
    #     "left_image_target_bboxes": image1_target_bboxes,
    #     "right_image_target_bboxes": image2_target_bboxes,
    #     "target_bbox_labels": torch.zeros(len(image1_target_bboxes)).long(),
    #     "registration_strategy": data['registration_strategy'],
    #     "depth1": data["depth1"],
    #     "depth2": data["depth2"],
    #     "intrinsics1": data["intrinsics1"],
    #     "intrinsics2": data["intrinsics2"],
    #     "position1": data["position1"],
    #     "position2": data["position2"],
    #     "rotation1": data["rotation1"],
    #     "rotation2": data["rotation2"],
    #     "query_metadata": {
    #         "pad_shape": data["image1"].shape[-2:],
    #         "border": np.array([0, 0, 0, 0]),
    #         "batch_input_shape": data["image1"].shape[-2:],
    #     },
    #     # "index": data["index"]
    # }

    # 使用的是传统对齐


def alignimage(image1_tensor, image2_tensor):
    # plt
    image1_plt = general.tensor_to_PIL(image1_tensor)
    image2_plt = general.tensor_to_PIL(image2_tensor)
    # cv
    image1_cv = cv2.cvtColor(np.asarray(image1_plt), cv2.COLOR_RGB2BGR)
    image2_cv = cv2.cvtColor(np.asarray(image2_plt), cv2.COLOR_RGB2BGR)
    # align the images
    # scan -> reference
    s2r, _, H = algin.alignImages(image1_cv, image2_cv)
    # inverH = torch.pinverse(torch.Tensor(H)).numpy()
    # h, w, channels = image2_cv.shape
    # # reference -> scan
    # r2s = cv2.warpPerspective(image2_cv, inverH, (w, h))

    # 将cv转为tensor

    image1_to_image2 = pil_to_tensor(Image.fromarray(cv2.cvtColor(s2r, cv2.COLOR_BGR2RGB))).float() / 255.0
    # image2_to_image1 = pil_to_tensor(Image.fromarray(cv2.cvtColor(r2s, cv2.COLOR_BGR2RGB))).float() / 255.0

    return image1_to_image2


def normalise_image(img_as_tensor):
    imagenet_normalisation = K.enhance.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = rearrange(img_as_tensor, "c h w -> 1 c h w")
    img = imagenet_normalisation(img)
    return img.squeeze()


def convert_kornia_transformation_matrix_to_normalised_coordinates(matrix, original_hw, new_hw):
    scale_up = torch.Tensor([[original_hw[1], 0, 0], [0, original_hw[0], 0], [0, 0, 1]])
    scale_down = torch.Tensor([[1 / new_hw[1], 0, 0], [0, 1 / new_hw[0], 0], [0, 0, 1]])
    return scale_down @ matrix @ scale_up


def resize_bbox(bboxes, original_size, new_size):
    """
    Args:
        bboxes: tensor of shape (N, 4) representing N bounding boxes.
                Each bounding box is represented as (x1, y1, x2, y2)
        original_size: tuple or list containing original image size (width, height)
        new_size: tuple or list containing new image size (width, height)
    Returns:
        resized_bboxes: tensor of shape (N, 4) representing resized bounding boxes.
    """
    # 计算缩放比例
    w_ratio = new_size[0] / original_size[0]
    h_ratio = new_size[1] / original_size[1]

    # 重塑bbox
    resized_bboxes = torch.zeros_like(bboxes)
    resized_bboxes[:, 0] = bboxes[:, 0] * w_ratio  # x1
    resized_bboxes[:, 1] = bboxes[:, 1] * h_ratio  # y1
    resized_bboxes[:, 2] = bboxes[:, 2] * w_ratio  # x2
    resized_bboxes[:, 3] = bboxes[:, 3] * h_ratio  # y2

    return resized_bboxes


def resize_depth(data):
    nearest_resize = K.augmentation.Resize((224, 224), resample=0, align_corners=None, keepdim=True)
    if data["depth1"] is not None:
        data["depth1"] = nearest_resize(data["depth1"])
    if data["depth2"] is not None:
        data["depth2"] = nearest_resize(data["depth2"])
    return data


# 这个是处理单个数据的，不是处理batch的
def prepare_batch_for_model(data):
    nearest_resize = K.augmentation.Resize((224, 224), resample=0, align_corners=None, keepdim=True)
    bicubic_resize = K.augmentation.Resize((224, 224), resample=2, keepdim=True)

    original_hw1 = data["image1"].shape[-2:]
    original_hw2 = data["image2"].shape[-2:]

    data["image1_target_annotations"] = resize_bbox(data["image1_target_annotations"], original_hw1, (224, 224))
    data["image2_target_annotations"] = resize_bbox(data["image2_target_annotations"], original_hw2, (224, 224))

    data["image1"] = bicubic_resize(data["image1"])
    data["image2"] = bicubic_resize(data["image2"])
    if data["depth1"] is not None:
        original_depth_hw1 = data["depth1"].shape[-2:]
        data["depth1"] = nearest_resize(data["depth1"])
    if data["depth2"] is not None:
        original_depth_hw2 = data["depth2"].shape[-2:]
        data["depth2"] = nearest_resize(data["depth2"])
    if data["intrinsics1"] is not None:
        assert original_hw1 == original_depth_hw1
        transformation = nearest_resize.transform_matrix.squeeze()
        transformation = convert_kornia_transformation_matrix_to_normalised_coordinates(transformation,
                                                                                        original_hw1, (224, 224))
        data["intrinsics1"] = transformation @ data["intrinsics1"]
    if data["intrinsics2"] is not None:
        assert original_hw2 == original_depth_hw2
        transformation = nearest_resize.transform_matrix.squeeze()
        transformation = convert_kornia_transformation_matrix_to_normalised_coordinates(transformation,
                                                                                        original_hw2, (224, 224))
        data["intrinsics2"] = transformation @ data["intrinsics2"]
    return data


def create_batch_from_metadata(metadata):
    batch = {}
    all_keys = list(metadata[0].keys())
    for key in all_keys:
        batch[key] = []
    for item in metadata:
        for key in all_keys:
            value = item.get(key, None)
            if value is None:
                batch[key].append(None)
                continue
            batch[key].append(value)

    return batch


def dataloader_collate_fn(batch, test, depth_predictor):
    """
    Defines the collate function for the dataloader specific to this
    method/model.
    """
    # 需要进行List转batch
    batch = create_batch_from_metadata(batch)

    # 先统一格式（在单个data时已经完成了处理，只是没有进行resize 和 gt赋值）
    # 再进行物理对齐 + 获取keypoints（以batch标准进行）
    #     进行对齐操作+ 获取每个图的depth信息
    batch = fill_in_the_missing_information(batch, test, depth_predictor)
    # 最后再resize
    batch = prepare_batch_for_model_all(batch)

    # 这个好像单纯是保证放入模型的数据
    keys = batch[0].keys()
    collated_dictionary = {}
    for key in keys:
        collated_dictionary[key] = []
        for batch_item in batch:
            collated_dictionary[key].append(batch_item[key])
        if key in [
            "left_image_target_bboxes",
            "right_image_target_bboxes",
            "query_metadata",
            "target_bbox_labels",
            "depth1",
            "depth2",
            "intrinsics1",
            "intrinsics2",
            "position1",
            "position2",
            "rotation1",
            "rotation2",
            "registration_strategy",
            # "index",
        ]:
            continue
        collated_dictionary[key] = ImageList.from_tensors(
            collated_dictionary[key], size_divisibility=32
        ).tensor

    # collated_dictionary = test.alignImage(collated_dictionary)
    return collated_dictionary


################################################
## The callback manager below handles logging ##
## to Weights And Biases.                     ##
################################################

class WandbCallbackManager(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        datamodule = DataModule(args)
        datamodule.setup()
        self.test_set_names = datamodule.test_dataset_names
        self.index = 0

    @rank_zero_only
    def on_fit_start(self, trainer, model):
        if self.args.no_logging:
            return
        trainer.logger.experiment.config.update(self.args, allow_val_change=True)

    @rank_zero_only
    def on_test_start(self, trainer, model):
        self.test_batches = [[] for _ in range(len(self.test_set_names))]
        self.test_set_predicted_bboxes = [[] for _ in range(len(self.test_set_names))]
        self.test_set_target_bboxes = [[] for _ in range(len(self.test_set_names))]
