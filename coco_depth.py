import os
import pickle
import random
from copy import deepcopy

import kornia as K
import numpy as np
import torch
from einops import rearrange
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from utilssss import general
from utilssss import geometry
from utilssss.general import cache_data_triton
from data.augmentation import AugmentationPipeline

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)


class InpatinedCocoDataset(Dataset):
    def __init__(self, depth_predictor, path_to_dataset, split, method, image_transformation, machine="local"):
        self.path_to_dataset = path_to_dataset
        train_val_test_split = self.get_train_val_test_split(split)
        self.indicies = train_val_test_split[split]
        self.indicies = self.indicies[int(len(self.indicies) * 0.52):]
        self.split = split
        self.machine = machine
        self.inpainted_image_names = self.get_inpainted_image_names()
        self.image_augmentations = AugmentationPipeline(
            mode=split, path_to_dataset=path_to_dataset, image_transformation=image_transformation
        )
        self.marshal_getitem_data = self.import_method_specific_functions(method)
        self.depth_predictor = depth_predictor
        self.depth_path = os.path.join(self.path_to_dataset, "depth")
        if not os.path.exists(self.depth_path):
            os.makedirs(self.depth_path)

    def import_method_specific_functions(self, method):
        if method == "centernet":
            from models.centernet_with_coam import marshal_getitem_data
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return marshal_getitem_data

    def get_train_val_test_split(self, split):
        train_val_test_split_file_path = os.path.join(self.path_to_dataset, "data_split.pkl")
        if os.path.exists(train_val_test_split_file_path):
            with open(train_val_test_split_file_path, "rb") as file:
                return pickle.load(file)
        indices_of_coco_images = np.load(os.path.join(self.path_to_dataset, "list_of_indices.npy"))
        # number_of_images = int(len(indices_of_coco_images) * 0.5)
        # indices_of_coco_images = indices_of_coco_images[:number_of_images]
        np.random.shuffle(indices_of_coco_images)
        # indices_of_coco_images = indices_of_coco_images[:number_of_images]
        if split == "test":
            train_val_test_split = {
                "test": indices_of_coco_images,
            }
        else:
            number_of_images = len(indices_of_coco_images)  # 6000
            number_of_train_images = int(0.95 * number_of_images)  # 5700
            train_val_test_split = {
                "train": indices_of_coco_images[:number_of_train_images],
                "val": indices_of_coco_images[number_of_train_images:],
            }
        with open(train_val_test_split_file_path, "wb") as file:
            pickle.dump(train_val_test_split, file)
        return train_val_test_split

    def get_inpainted_image_names(self):
        filenames_as_list = list(os.listdir(os.path.join(self.path_to_dataset, "inpainted")))
        inpainted_image_names = dict()
        for filename in filenames_as_list:
            index = int(filename.split("_")[0])
            if index in inpainted_image_names.keys():
                inpainted_image_names[index].append(filename)
            else:
                inpainted_image_names[index] = [filename]
        return inpainted_image_names

    def read_image_as_tensor(self, path_to_image):
        """
        Returms a normalised RGB image as tensor.
        """
        pil_image = Image.open(path_to_image).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
        return image_as_tensor

    def get_inpainted_objects_bitmap_from_image_path(self, image_path, bit_length):
        if "inpainted" not in image_path:
            return 0
        bitmap_string = image_path.split("mask")[1].split(".")[0]
        if bitmap_string == "":
            return (2 ** bit_length) - 1
        return int(bitmap_string)

    def add_random_objects(self, image_as_tensor, item_index, depth):
        all_indices_except_current = list(range(item_index)) + list(
            range(item_index + 1, len(self.indicies))
        )
        random_image_index = random.choice(all_indices_except_current)
        index = self.indicies[random_image_index]
        original_image = self.read_image_as_tensor(
            cache_data_triton(self.path_to_dataset, f"images_and_masks/{index}.png", self.machine)
        )
        original_depth = self.get_depth(self.depth_path + f"/{index}.pt", original_image)
        annotation_path = cache_data_triton(
            self.path_to_dataset, f"metadata/{index}.npy", self.machine
        )
        annotations = np.load(annotation_path, allow_pickle=True)
        (
            original_image_resized_to_current,
            original_depth_resized_to_current,
            annotations_resized,
        ) = geometry.resize_image_and_annotations(
            original_image, original_depth, image_as_tensor.shape[-2:], annotations
        )
        annotation_mask = general.coco_annotations_to_mask_np_array(
            annotations_resized, image_as_tensor.shape[-2:]
        )
        image_as_tensor = rearrange(image_as_tensor, "c h w -> h w c")
        original_image_resized_to_current = rearrange(
            original_image_resized_to_current, "c h w -> h w c"
        )
        original_depth_resized_to_current = rearrange(
            original_depth_resized_to_current, "c h w -> h w c"
        )
        depth = rearrange(depth, "c h w -> h w c")
        image_as_tensor[annotation_mask] = original_image_resized_to_current[annotation_mask]
        depth[annotation_mask] = original_depth_resized_to_current[annotation_mask]
        return rearrange(image_as_tensor, "h w c -> c h w"), annotations_resized, index, rearrange(depth,
                                                                                                   "h w c -> c h w")

    def get_depth(self, depth_file, tensor):
        if os.path.isfile(depth_file):
            with open(depth_file, "rb") as f:
                depth = torch.load(f)
                print(f"已经--保存了图片depth信息：{depth_file}")
        else:
            # 对train 和 val 数据集进行测试depth并保存， 方便后续继续使用
            depth = self.depth_predictor.infer(tensor.unsqueeze(0)).squeeze().unsqueeze(0)  # [1,w,h]
            with open(depth_file, "wb") as f:
                torch.save(depth, f)
                print(f"保存图片depth信息：{depth_file}")

        return depth

    def __len__(self):
        """
        Returns the number of training/validation/testing images.
        """
        return len(self.indicies)

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return item_data

    @torch.no_grad()
    def __base_getitem__(self, item_index):
        index = self.indicies[item_index]
        image_filenames = [f"images_and_masks/{index}.png"]
        for inpainted_image_name in self.inpainted_image_names[index]:
            image_filenames.append("inpainted/" + inpainted_image_name)
        for image_path in image_filenames:
            filename = image_path.split('/')[-1][:-4]
            depth_file = os.path.join(self.depth_path, f"{filename}.pt")
            image_as_tensor = self.read_image_as_tensor(
                cache_data_triton(self.path_to_dataset, image_path, self.machine)
            )
            self.get_depth(depth_file, image_as_tensor)
        return {
            'registration_strategy': "2d",
        }


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config

    conf = get_config("zoedepth_nk", "infer")
    depth_predictor = build_model(conf)
    dataset = InpatinedCocoDataset(depth_predictor=depth_predictor,
                                   path_to_dataset="/home/ygk/disk/datas/data/coco-inpainted/train", split="train",
                                   method="centernet", image_transformation="affine")

    dataloader = DataLoader(dataset, batch_size=2048)
    for batch in dataloader:
        print()
