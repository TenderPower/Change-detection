import os
from pathlib import Path

import kornia as K
import shapely
import torch
import torch.nn as nn
from einops import rearrange

import utilssss.geometry as geometry


class AugmentationPipeline(nn.Module):
    def __init__(self, mode, path_to_dataset, image_transformation) -> None:
        super(AugmentationPipeline, self).__init__()
        self.jit = K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0, keepdim=True)
        self.aff = K.augmentation.RandomAffine(
            degrees=30,
            translate=(0.2, 0.2),
            scale=(0.8, 1.5, 0.8, 1.5),
            padding_mode="border",
            p=1.0,
            keepdim=True,
        )
        self.mode = mode
        self.path_to_dataset = path_to_dataset
        self.image_transformation = image_transformation

    def kornia_augmentation_function(self, input, depth, type_of_image, image_index):
        if len(input.shape) == 3:
            input = torch.unsqueeze(input, dim=0)

        if self.mode in ["train", "val"]:
            return self.apply_train_augmentations(input, depth)
        if self.mode == "test":
            return self.apply_test_augmentations(input, depth, type_of_image, image_index)
        raise NotImplementedError(f"Unsupported mode {self.mode}")

    def apply_train_augmentations(self, input, depth):
        input = self.jit(input)
        if self.image_transformation == "identity":
            return input, torch.eye(3)

        input = torch.cat([input, depth.unsqueeze(0).unsqueeze(0)], 1)
        input_ = self.aff(input)
        input = input_[:, :3, :, :]
        depth = input_[:, 3:, :, :]
        return input, self.aff.transform_matrix, depth

    def apply_test_augmentations(self, input, depth, type_of_image, image_index):
        # 这里也要改 但先暂定
        precomputed_augmentation_path = os.path.join(
            self.path_to_dataset, f"test_augmentations/{type_of_image}/{image_index}.params"
        )
        if os.path.exists(precomputed_augmentation_path):
            augmentation_params = torch.load(precomputed_augmentation_path)
        else:
            jit_params = self.jit.generate_parameters(input.shape)
            aff_params = self.aff.generate_parameters(input.shape)
            augmentation_params = {"aff": aff_params, "jit": jit_params}
            Path(precomputed_augmentation_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(augmentation_params, precomputed_augmentation_path)
        input = self.jit(input, params=augmentation_params["jit"])
        if self.image_transformation == "identity":
            return input, torch.eye(3)

        input = torch.cat([input, depth.unsqueeze(0).unsqueeze(0)], 1)
        input_ = self.aff(input, params=augmentation_params["aff"])
        input = input_[:, :3, :, :]
        depth = input_[:, 3:, :, :]
        return input, self.aff.transform_matrix, depth

    def forward(self, image1_image_as_tensor, image2_image_as_tensor, depth1, depth2, annotations, image_index):
        if self.image_transformation == "registered":
            return self.forward_registered(
                image1_image_as_tensor, image2_image_as_tensor, annotations, image_index
            )
        if self.image_transformation == "identity":
            return self.forward_identity(
                image1_image_as_tensor, image2_image_as_tensor, annotations, image_index
            )
        # 只改affine，其他的部分不动
        return self.forward_transformed(
            image1_image_as_tensor, image2_image_as_tensor, depth1, depth2, annotations, image_index
        )

    def forward_transformed(
            self,
            image1_image_as_tensor,
            image2_image_as_tensor,
            depth1,
            depth2,
            annotations,
            image_index,
    ):
        """
        The is a hairy piece code. It performs the following steps:
        1. apply a random transformation to both image1
            and image2. We store this transformation.
            Due to geometric transformations such as
            random crops, the labels may not be completely valid e.g.
            the changed part in augmented image 1 may not be visible
            in augmented image 2.
        2. transform the segmentations twice with the same transformations
            as in step 1.
        3. invert the transformation in step 2 and take the
            intersecton of the segmentations. These segmentations should now
            reflect the valid changed area that is visible in both
            the augmented images.
        4. reapply the appropriate transformation to the segmentations in step 3
            and compute the axis aligned bounding boxes.
        """
        image_shape_as_hw = image1_image_as_tensor.shape[-2:]
        segmentations_as_shapely_objects = (
            geometry.get_segmentation_as_shapely_polygons_from_coco_annotations(annotations)
        )
        ############
        ## Step 1 ##
        ############
        (
            image1_image_transformed,
            image1_image_transformation,
            depth1,
        ) = self.kornia_augmentation_function(image1_image_as_tensor, depth1, "original", image_index)
        (
            image2_image_transformed,
            image2_image_transformation,
            depth2,
        ) = self.kornia_augmentation_function(image2_image_as_tensor, depth2, "inpainted", image_index)

        ############
        ## Step 2 ##
        ############
        image1_segmentations_transformed = (
            geometry.apply_kornia_transformation_to_shapely_objects(
                segmentations_as_shapely_objects,
                image1_image_transformation,
                image_shape_as_hw,
                keep_empty=True,
            )
        )
        image2_segmentations_transformed = (
            geometry.apply_kornia_transformation_to_shapely_objects(
                segmentations_as_shapely_objects,
                image2_image_transformation,
                image_shape_as_hw,
                keep_empty=True,
            )
        )
        ############
        ## Step 3 ##
        ############
        image1_image_inverse_transformation = torch.inverse(image1_image_transformation)
        image2_image_inverse_transformation = torch.inverse(image2_image_transformation)

        image1_segmentations_transformed_inverted = (
            geometry.apply_kornia_transformation_to_shapely_objects(
                image1_segmentations_transformed,
                image1_image_inverse_transformation,
                image_shape_as_hw,
                keep_empty=True,
            )
        )
        image2_segmentations_transformed_inverted = (
            geometry.apply_kornia_transformation_to_shapely_objects(
                image2_segmentations_transformed,
                image2_image_inverse_transformation,
                image_shape_as_hw,
                keep_empty=True,
            )
        )
        intersection_of_segmentations = []
        for image1, image2 in zip(
                image1_segmentations_transformed_inverted, image2_segmentations_transformed_inverted
        ):
            intersection_segmentation = geometry.make_valid_polygon(
                image1.intersection(image2)
            )
            intersection_of_segmentations.append(intersection_segmentation)
        ############
        ## Step 4 ##
        ############
        target_image1_segmentations = geometry.apply_kornia_transformation_to_shapely_objects(
            intersection_of_segmentations,
            image1_image_transformation,
            image_shape_as_hw,
            keep_empty=True,
        )
        target_image2_segmentations = geometry.apply_kornia_transformation_to_shapely_objects(
            intersection_of_segmentations,
            image2_image_transformation,
            image_shape_as_hw,
            keep_empty=True,
        )
        transformed_image1_annotations = geometry.merge_shapely_polygons_into_annotations(
            target_image1_segmentations, annotations
        )
        transformed_image2_annotations = geometry.merge_shapely_polygons_into_annotations(
            target_image2_segmentations, annotations
        )

        return (
            image1_image_transformed,
            image2_image_transformed,
            depth1,
            depth2,
            transformed_image1_annotations,
            transformed_image2_annotations,
        )

    def forward_identity(
            self,
            image1_image_as_tensor,
            image2_image_as_tensor,
            annotations,
            image_index,
    ):
        """
        Unlike forward_transformed(), here we just return the original images with
        colour jittering and annotations in the correct format.
        """
        segmentations_as_shapely_objects = (
            geometry.get_segmentation_as_shapely_polygons_from_coco_annotations(annotations)
        )
        (image1_image_transformed, _) = self.kornia_augmentation_function(
            image1_image_as_tensor, "original", image_index
        )
        (image2_image_transformed, _) = self.kornia_augmentation_function(
            image2_image_as_tensor, "inpainted", image_index
        )
        transformed_image1_annotations = geometry.merge_shapely_polygons_into_annotations(
            segmentations_as_shapely_objects, annotations
        )
        transformed_image2_annotations = geometry.merge_shapely_polygons_into_annotations(
            segmentations_as_shapely_objects, annotations
        )
        return (
            image1_image_transformed,
            image2_image_transformed,
            transformed_image1_annotations,
            transformed_image2_annotations,
        )

    def forward_registered(
            self, image1_image_as_tensor, image2_image_as_tensor, annotations, image_index
    ):
        """
        Like forward_transformed(), we first transform both the images but then
        we invert the transformation to "register" the images. This is different
        from forward_identity() in that these images may have boundary effects due
        to cropping.
        """
        image_shape_as_hw = image1_image_as_tensor.shape[-2:]
        segmentations_as_shapely_objects = (
            geometry.get_segmentation_as_shapely_polygons_from_coco_annotations(annotations)
        )
        ############
        ## Step 1 ##
        ############
        (
            image1_image_transformed,
            image1_image_transformation,
        ) = self.kornia_augmentation_function(image1_image_as_tensor, "original", image_index)
        (
            image2_image_transformed,
            image2_image_transformation,
        ) = self.kornia_augmentation_function(image2_image_as_tensor, "inpainted", image_index)

        ############
        ## Step 2 ##
        ############
        image1_segmentations_transformed = (
            geometry.apply_kornia_transformation_to_shapely_objects(
                segmentations_as_shapely_objects,
                image1_image_transformation,
                image_shape_as_hw,
                keep_empty=True,
            )
        )
        image2_segmentations_transformed = (
            geometry.apply_kornia_transformation_to_shapely_objects(
                segmentations_as_shapely_objects,
                image2_image_transformation,
                image_shape_as_hw,
                keep_empty=True,
            )
        )
        ############
        ## Step 3 ##
        ############
        image1_image_inverse_transformation = torch.inverse(image1_image_transformation)
        image2_image_inverse_transformation = torch.inverse(image2_image_transformation)
        image1_image_transformed = K.geometry.warp_perspective(
            image1_image_transformed,
            rearrange(image1_image_inverse_transformation, "r c -> 1 r c"),
            mode="nearest",
            dsize=image_shape_as_hw,
            padding_mode="border",
        )

        image2_image_transformed = K.geometry.warp_perspective(
            image2_image_transformed,
            rearrange(image2_image_inverse_transformation, "r c -> 1 r c"),
            mode="nearest",
            dsize=image_shape_as_hw,
            padding_mode="border",
        )

        image1_segmentations_transformed_inverted = (
            geometry.apply_kornia_transformation_to_shapely_objects(
                image1_segmentations_transformed,
                image1_image_inverse_transformation,
                image_shape_as_hw,
                keep_empty=True,
            )
        )
        image2_segmentations_transformed_inverted = (
            geometry.apply_kornia_transformation_to_shapely_objects(
                image2_segmentations_transformed,
                image2_image_inverse_transformation,
                image_shape_as_hw,
                keep_empty=True,
            )
        )
        intersection_of_segmentations = []
        for image1, image2 in zip(
                image1_segmentations_transformed_inverted, image2_segmentations_transformed_inverted
        ):
            intersection_segmentation = geometry.make_valid_polygon(
                image1.intersection(image2)
            )
            intersection_of_segmentations.append(intersection_segmentation)
        ############
        ## Step 4 ##
        ############
        transformed_image1_annotations = geometry.merge_shapely_polygons_into_annotations(
            intersection_of_segmentations, annotations
        )
        transformed_image2_annotations = geometry.merge_shapely_polygons_into_annotations(
            intersection_of_segmentations, annotations
        )

        return (
            image1_image_transformed,
            image2_image_transformed,
            transformed_image1_annotations,
            transformed_image2_annotations,
        )
