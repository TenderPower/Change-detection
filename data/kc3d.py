import os
import pickle
from loguru import logger as L
# import imageio.v2 as imageio
import imageio
import kornia as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from etils import epath
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
from torchvision.ops import masks_to_boxes


class KC3D(Dataset):
    def __init__(self, depth_predictor, path_to_dataset, split, method, use_ground_truth_registration=True):
        self.path_to_dataset = path_to_dataset
        self.data = self.get_data_info(path_to_dataset)
        self.indicies = self.data[split]
        self.split = split

        len_train = int(len(self.data["train"]) * 0.2)
        len_val = len(self.data["val"])
        len_test = len(self.data["test"])
        self.use_ground_truth_registration = use_ground_truth_registration
        self.annotation_indices = np.arange(len_train + len_val, len_train + len_val + len_test)
        self.data = np.array(self.indicies)

        self.marshal_getitem_data = self.import_method_specific_functions(method)

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.centernet_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def get_data_info(self, path_to_dataset):
        train_val_test_split_file_path = os.path.join(path_to_dataset, "data_split.pkl")
        if os.path.exists(train_val_test_split_file_path):
            with open(train_val_test_split_file_path, "rb") as file:
                return pickle.load(file)

    def read_image_as_tensor(self, path_to_image):
        """
        Returms a normalised RGB image as tensor.
        """
        with open(path_to_image, "rb") as file:
            pil_image = Image.open(file).convert("RGB")
            image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor.squeeze()

    def read_gt_depth_from_tiff(self, path_to_depth):
        """
        Returns the depth map as a float tensor.
        """
        filename = epath.Path(path_to_depth)
        img = imageio.imread(filename.read_bytes(), format="tiff")
        if img.ndim == 2:
            img = img[:, :, None]
        return K.image_to_tensor(img).float().squeeze()

    def __len__(self):
        """
        Returns the number of images.
        """
        return len(self.data)

    def get_target_annotation_mask(self, mask_path):
        with open(mask_path, "rb") as file:
            pil_image = Image.open(file)
            mask_as_np_array = np.array(pil_image)
        return K.image_to_tensor(mask_as_np_array).float()

    def get_target_bboxes_from_mask(self, mask_as_tensor):
        if len(mask_as_tensor.shape) == 2:
            mask_as_tensor = rearrange(mask_as_tensor, "h w -> 1 h w")
        bboxes = masks_to_boxes(mask_as_tensor)
        return bboxes

    def __getitem__(self, item_index):
        scene = self.data[item_index]
        image1_as_tensor = self.read_image_as_tensor(os.path.join(self.path_to_dataset, scene["image1"]))
        image2_as_tensor = self.read_image_as_tensor(os.path.join(self.path_to_dataset, scene["image2"]))
        depth1_as_tensor = self.read_gt_depth_from_tiff(os.path.join(self.path_to_dataset, scene["depth1"]))
        depth2_as_tensor = self.read_gt_depth_from_tiff(os.path.join(self.path_to_dataset, scene["depth2"]))

        target_mask_1 = self.get_target_annotation_mask(os.path.join(self.path_to_dataset, scene["mask1"]))
        target_mask_2 = self.get_target_annotation_mask(os.path.join(self.path_to_dataset, scene["mask2"]))
        target_bbox_1 = self.get_target_bboxes_from_mask(target_mask_1)
        target_bbox_2 = self.get_target_bboxes_from_mask(target_mask_2)

        data = {
            "image1": image1_as_tensor,
            "image2": image2_as_tensor,
            "depth1": depth1_as_tensor,
            "depth2": depth2_as_tensor,
            "registration_strategy": "3d",
            "image1_target_annotations": target_bbox_1,
            "image2_target_annotations": target_bbox_2,
            # "index": "_".join(scene["image1"].split("_")[:-1])

        }

        if self.use_ground_truth_registration:
            metadata = np.load(
                os.path.join(
                    self.path_to_dataset,
                    "_".join(scene["image1"].split(".")[0].split("_")[:3]) + ".npy",
                ),
                allow_pickle=True,
            ).item()

            for key, value in metadata.items():
                if key == "intrinsics":
                    data["intrinsics1"] = torch.Tensor(value)
                    data["intrinsics2"] = torch.Tensor(value)
                elif key == "position_before":
                    data["position1"] = torch.Tensor(value)
                elif key == "position_after":
                    data["position2"] = torch.Tensor(value)
                elif key == "rotation_before":
                    data["rotation1"] = torch.Tensor(value)
                elif key == "rotation_after":
                    data["rotation2"] = torch.Tensor(value)

        return self.marshal_getitem_data(data, self.split)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from models.centernet_with_coam import dataloader_collate_fn
    from models.test___ import Test

    dataset = KC3D(path_to_dataset="/home/ygk/disk/datas/data/kc3d", split="test", method="centernet",
                   use_ground_truth_registration=True)
    import matplotlib.pyplot as plt


    def collate_fn(batch, dataset):
        """
        A wrapper collate function that calls method-specific,
        data collation functions. It also takes care of filtering out
        any None batch items and if the batch ends up empty, it attempts
        to create a batch of a single non-None item.
        """
        batch = [x for x in batch if x is not None]
        tries = 0
        while len(batch) == 0:
            tries += 1
            random_item = dataset[np.random.randint(0, len(dataset))]
            if random_item is not None:
                batch = [random_item]
            if tries % 50 == 0:
                L.log(
                    "DEBUG",
                    f"Made {tries} attempts to construct a non-None batch.\
                        If this happens too often, maybe it's not a good workaround.",
                )
        return dataloader_collate_fn(batch, Test())


    def collate_fn_wrapper(batch):
        return collate_fn(batch, dataset)


    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_wrapper)
    for batch in dataloader:
        pass
