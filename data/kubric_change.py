from __future__ import annotations

import os

import kornia as K
import numpy as np
from einops import rearrange
from PIL import Image
from scipy.ndimage import label as label_connected_components
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import pil_to_tensor

import torch


class KubricChange(Dataset):
    def __init__(self, depth_predictor, path_to_dataset, split, method):
        assert split == "test"
        self.split = split
        self.marshal_getitem_data = self.import_method_specific_functions(method)
        self.data = self.get_data_info(path_to_dataset)
        self.path_to_dataset = path_to_dataset
        self.depth_predictor = depth_predictor

    def get_data_info(self, path_to_dataset):
        if os.path.exists(os.path.join(path_to_dataset, "metadata.npy")):
            return np.load(os.path.join(path_to_dataset, "metadata.npy"), allow_pickle=True)
        image_1, image_2, mask_1, mask_2 = [], [], [], []
        for file in os.listdir(path_to_dataset):
            file_without_extension = file.split(".")[0]
            id = file_without_extension.split("_")[-1]
            if id == "00000":
                mask_1.append(os.path.join(path_to_dataset, file))
            elif id == "00001":
                mask_2.append(os.path.join(path_to_dataset, file))
            elif id == "0":
                image_1.append(os.path.join(path_to_dataset, file))
            elif id == "1":
                image_2.append(os.path.join(path_to_dataset, file))
            else:
                continue
        assert len(image_1) == len(image_2) == len(mask_1) == len(mask_2)
        image_1, image_2, mask_1, mask_2 = (
            sorted(image_1),
            sorted(image_2),
            sorted(mask_1),
            sorted(mask_2),
        )
        data = np.array(list(zip(image_1, image_2, mask_1, mask_2)))
        np.save(os.path.join(path_to_dataset, "metadata"), data)
        return data

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.centernet_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def read_image_as_tensor(self, path_to_image):
        """
        Returms a normalised RGB image as tensor.
        """
        pil_image = Image.open(path_to_image).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor

    def __len__(self):
        """
        Returns the number of testing images.
        """
        return len(self.data)

    def get_target_annotations_in_coco_format(self, mask_path):
        pil_image = Image.open(mask_path)
        mask_as_np_array = np.array(pil_image)
        (
            connected_components,
            number_of_components,
        ) = label_connected_components(mask_as_np_array)
        masks = []
        for i in range(number_of_components):
            masks.append(connected_components == i + 1)
        masks = rearrange(masks, "c h w -> h w c")
        masks_as_tensor = K.image_to_tensor(masks)
        bboxes = masks_to_boxes(masks_as_tensor)
        coco_annotations = []
        for bbox in bboxes:
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            four_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            coco_annotation = {
                "bbox": [*four_corners[0], *four_corners[2]],
                "segmentation": [four_corners.reshape(-1)],
            }
            coco_annotations.append(coco_annotation)
        return coco_annotations

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)
    @torch.no_grad()
    def __base_getitem__(self, item_index):
        """
        Returns 3 things:
        a reference image (3xRxC),
        a query image (3xRxC), and
        a list of coco-format change annotations corresponding to the query image.
        """
        image1_path, image2_path, mask1_path, mask2_path = self.data[item_index]
        image1_as_tensor = self.read_image_as_tensor(image1_path)
        image2_as_tensor = self.read_image_as_tensor(image2_path)
        image1_target_annotation = self.get_target_annotations_in_coco_format(mask1_path)
        image2_target_annotation = self.get_target_annotations_in_coco_format(mask2_path)

        part = image1_path.split('/')[-1].split('_')
        index = '_'.join(part[:3])


        depth_path = os.path.join(self.path_to_dataset, "depth")
        if not os.path.exists(depth_path):
            os.makedirs(depth_path)

        depth1_file = os.path.join(depth_path, f"{index}_1.pt")
        depth2_file = os.path.join(depth_path, f"{index}_2.pt")
        # depth1
        if os.path.isfile(depth1_file):
            with open(depth1_file, "rb") as f:
                depth1 = torch.load(f)
        else:
            depth1 = self.depth_predictor.infer(image1_as_tensor.unsqueeze(0)).squeeze()
            # depth1 = self.depth_predictor.eval(image1_image_as_tensor)
            with open(depth1_file, "wb") as f:
                torch.save(depth1, f)

        # depth2
        if os.path.isfile(depth2_file):
            with open(depth2_file, "rb") as f:
                depth2 = torch.load(f)
        else:
            # 对train 和 val 数据集进行测试depth并保存， 方便后续继续使用
            depth2 = self.depth_predictor.infer(image2_as_tensor.unsqueeze(0)).squeeze()
            # depth2 = self.depth_predictor.eval(image2_image_as_tensor)
            with open(depth2_file, "wb") as f:
                torch.save(depth2, f)

        return {
            "image1": image1_as_tensor.squeeze(),
            "image2": image2_as_tensor.squeeze(),
            "image1_target_annotations": image1_target_annotation,
            "image2_target_annotations": image2_target_annotation,
            "depth1": depth1,
            "depth2": depth2,
            # "index": index,
            # "path": self.path_to_dataset
            # "registration_strategy": None
        }
