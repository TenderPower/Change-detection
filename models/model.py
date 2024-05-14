import math
import os
import pickle
import types
from typing import Tuple
import pytorch_lightning as pl
import kornia as K
import numpy as np
import timm
from timm.models import vision_transformer
import torch
import torch.nn as nn
import torch.nn.modules.utils as nn_utils
from easydict import EasyDict
from loguru import logger as L
from mmdet.models.dense_heads.centernet_head import CenterNetHead
from segmentation_models_pytorch.unet.model import UnetDecoder
from einops import rearrange

from utilssss.voc_eval import BoxList, eval_detection_voc
from models.building_blocks import DownSamplingBlock, FeatureFusionBlock, Sequence2SpatialBlock
from models.registeration_module import FeatureRegisterationModule


class Model(pl.LightningModule):
    def __init__(self, args, load_weights_from=None):
        super().__init__()
        self.args = args
        self.test_set_names = [test_set.name for test_set in args.datasets.test_datasets]
        # self.val_set_names = [val_set.name for val_set in args.datasets.val_datasets]

        model = build_model(args)
        self.feature_backbone = FeatureBackbone(args, model)
        self.registeration_module = FeatureRegisterationModule()
        # keepdim=True
        self.bicubic_resize = K.augmentation.Resize((64, 64), resample=2)
        self.unet_encoder = nn.ModuleList([DownSamplingBlock(i, j) for i, j in args.decoder.downsampling_blocks])
        self.unet_decoder = UnetDecoder(
            encoder_channels=args.decoder.encoder_channels,
            decoder_channels=args.decoder.decoder_channels,
            n_blocks=len(args.decoder.decoder_channels),
            use_batchnorm=True,
            center=False,
            attention_type="scse",
            num_coam_layers=0,
            return_features=False,
        )
        self.feature_fusion_block = FeatureFusionBlock(input_dims=64 + 768, hidden_dims=256, output_dims=64,
                                                       output_resolution=[224, 224])
        self.centernet_head = CenterNetHead(
            in_channel=64,
            feat_channel=64,
            num_classes=1,
            test_cfg=EasyDict({"topk": 100, "local_maximum_kernel": 3, "max_per_img": 100}),
        )
        self.centernet_head.init_weights()
        if load_weights_from is not None:
            self.safely_load_state_dict(torch.load(load_weights_from))

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
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--lr", type=float)
        parser.add_argument("--weight_decay", type=float)
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
        # dataloader_idx 用来分辨验证数据集的
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

        return {'loss': overall_loss}

    # def validation_epoch_end(self, outputs):
    #     a = 1
    #     avg_loss = torch.stack([x[0]['loss'] for x in outputs]).mean()
    #     self.log('val_loss', avg_loss, on_epoch=True)

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
                            image_size=(224, 224),
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
                            image_size=(224, 224),
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
        optimizer = torch.optim.Adam(optimizer_params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer

        # 写一个方法用于验证对齐的方式

    def test(self, batch, im1, im2):
        self.registeration_module.test(batch, im1, im2)
        pass

    def forward(self, batch):
        # 在这里测试变化

        left_images = batch['left_image']
        right_images = batch['right_image']
        l2r = batch['left2right']
        r2l = batch['right2left']
        # 将图片1与对齐后的图片2 进行拼接
        imag_one = torch.concat((left_images, r2l), 1)
        imag_two = torch.concat((right_images, l2r), 1)
        # 测试3d是否能对齐的
        # self.test(batch, imag_one, imag_two)
        image1_dino_features = self.feature_backbone(imag_one)
        image2_dino_features = self.feature_backbone(imag_two)
        image1_last_layer = self.bicubic_resize(image1_dino_features[-1])
        image2_last_layer = self.bicubic_resize(image2_dino_features[-1])
        image1_encoded_features = [[], image1_last_layer]
        image2_encoded_features = [[], image2_last_layer]
        # 进行U-Net Encoder
        for layer in self.unet_encoder:
            image1_encoded_features.append(layer(image1_encoded_features[-1]))
            image2_encoded_features.append(layer(image2_encoded_features[-1]))
        # 进行Feature Registration and Difference
        for i in range(len(self.unet_encoder) + 1):
            image1_encoded_features[i + 1], image2_encoded_features[i + 1] = self.registeration_module(
                batch, image1_encoded_features[i + 1], image2_encoded_features[i + 1]
            )
        image1_decoded_features = self.unet_decoder(*image1_encoded_features)
        image2_decoded_features = self.unet_decoder(*image2_encoded_features)
        image1_decoded_features = self.feature_fusion_block(image1_dino_features[0], image1_decoded_features)
        image2_decoded_features = self.feature_fusion_block(image2_dino_features[0], image2_decoded_features)
        return (
            self.centernet_head([image1_decoded_features]),
            self.centernet_head([image2_decoded_features]),
        )

    def get_bboxes_from_logits(self, image1_outputs, image2_outputs, batch):
        image1_predicted_bboxes = self.centernet_head.get_bboxes(
            *image1_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        image2_predicted_bboxes = self.centernet_head.get_bboxes(
            *image2_outputs,
            img_metas=batch["query_metadata"],
            rescale=False,
        )
        return image1_predicted_bboxes, image2_predicted_bboxes

    @torch.no_grad()
    def predict(self, batch):
        image1_outputs, image2_outputs = self(batch)
        batch_image1_predicted_bboxes, batch_image2_predicted_bboxes = self.get_bboxes_from_logits(image1_outputs,
                                                                                                   image2_outputs,
                                                                                                   batch)
        return batch_image1_predicted_bboxes, batch_image2_predicted_bboxes


class FeatureBackbone(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.sequence_to_spatial = nn.ModuleList([Sequence2SpatialBlock(args) for _ in args.vit_feature_layers])
        self._features = []
        self.register_hooks(args.vit_feature_layers)

    def register_hooks(self, hook_layers):
        for index in hook_layers:
            def _hook(module, input, output):
                qkv = rearrange(output, "b n (t c) -> t b n c", t=3)
                self._features.append(qkv[1])

            self.model.blocks[index].attn.qkv.register_forward_hook(_hook)

    def forward(self, x):
        self.model.forward_features(x)  # desired features will get stored in self._features
        output = [self.sequence_to_spatial[i](feature) for i, feature in enumerate(self._features)]
        self._features.clear()  # clear for next forward pass
        return output


def build_model(args, frozen=True):
    # pretrained_cfg_overlay = {'file': r"/home/ygk/.cache/huggingface/hub/models--timm--vit_base_patch8_224.dino/pytorch_model.bin"}
    model = timm.create_model("vit_base_patch8_224_dino", pretrained=True, in_chans=6)
    model = patch_vit_resolution(model, image_hw=[224, 224], stride=args.encoder.stride)
    if frozen:
        for _, value in model.named_parameters():
            value.requires_grad = False
    return model


def patch_vit_resolution(model: nn.Module, image_hw, stride: int) -> nn.Module:
    """
    change resolution of model output by changing the stride of the patch extraction.
    :param model: the model to change resolution for.
    :param stride: the new stride parameter.
    :return: the adjusted model
    """
    patch_size = model.patch_embed.patch_size
    if stride == patch_size:  # nothing to do
        return model

    stride = nn_utils._pair(stride)
    assert all([(p // s_) * s_ == p for p, s_ in
                zip(patch_size, stride)]), f"stride {stride} should divide patch_size {patch_size}"

    # fix the stride
    model.patch_embed.proj.stride = stride
    # fix the positional encoding code
    model._pos_embed = types.MethodType(fix_pos_enc(patch_size, image_hw, stride), model)
    return model


def fix_pos_enc(patch_size: Tuple[int, int], image_hw, stride_hw: Tuple[int, int]):
    """
    Creates a method for position encoding interpolation.
    :param patch_size: patch size of the model.
    :param stride_hw: A tuple containing the new height and width stride respectively.
    :return: the interpolation method
    """

    def interpolate_pos_encoding(self, x) -> torch.Tensor:
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        h, w = image_hw
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # compute number of tokens taking stride into account
        w0 = 1 + (w - patch_size[1]) // stride_hw[1]
        h0 = 1 + (h - patch_size[1]) // stride_hw[0]
        assert (
                w0 * h0 == npatch
        ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
                                        stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=False,
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return x + torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    return interpolate_pos_encoding
