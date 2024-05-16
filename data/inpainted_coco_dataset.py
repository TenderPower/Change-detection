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

import utilssss.general as general
import utilssss.geometry as geometry
from data.augmentation import AugmentationPipeline
from utilssss.general import cache_data_triton

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)


class InpatinedCocoDataset(Dataset):
    def __init__(self, depth_predictor, path_to_dataset, split, method, image_transformation, machine="local"):
        self.path_to_dataset = path_to_dataset
        train_val_test_split = self.get_train_val_test_split(split)
        self.indicies = train_val_test_split[split]
        self.split = split
        self.machine = machine
        self.inpainted_image_names = self.get_inpainted_image_names()
        self.image_augmentations = AugmentationPipeline(
            mode=split, path_to_dataset=path_to_dataset, image_transformation=image_transformation
        )
        self.marshal_getitem_data = self.import_method_specific_functions(method)
        self.depth_predictor = depth_predictor

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
        number_of_images = int(len(indices_of_coco_images) * 0.5)
        # indices_of_coco_images = indices_of_coco_images[:number_of_images]
        np.random.shuffle(indices_of_coco_images)
        indices_of_coco_images = indices_of_coco_images[:number_of_images]
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

    def add_random_objects(self, image_as_tensor, item_index):
        all_indices_except_current = list(range(item_index)) + list(
            range(item_index + 1, len(self.indicies))
        )
        random_image_index = random.choice(all_indices_except_current)
        index = self.indicies[random_image_index]
        original_image = self.read_image_as_tensor(
            cache_data_triton(self.path_to_dataset, f"images_and_masks/{index}.png", self.machine)
        )
        annotation_path = cache_data_triton(
            self.path_to_dataset, f"metadata/{index}.npy", self.machine
        )
        annotations = np.load(annotation_path, allow_pickle=True)
        (
            original_image_resized_to_current,
            annotations_resized,
        ) = geometry.resize_image_and_annotations(
            original_image, image_as_tensor.shape[-2:], annotations
        )
        annotation_mask = general.coco_annotations_to_mask_np_array(
            annotations_resized, image_as_tensor.shape[-2:]
        )
        image_as_tensor = rearrange(image_as_tensor, "c h w -> h w c")
        original_image_resized_to_current = rearrange(
            original_image_resized_to_current, "c h w -> h w c"
        )
        image_as_tensor[annotation_mask] = original_image_resized_to_current[annotation_mask]
        return rearrange(image_as_tensor, "h w c -> c h w"), annotations_resized, index

    def __len__(self):
        """
        Returns the number of training/validation/testing images.
        """
        return len(self.indicies)

    def __getitem__(self, item_index):
        item_data = self.__base_getitem__(item_index)
        return self.marshal_getitem_data(item_data, self.split)

    @torch.no_grad()
    def __base_getitem__(self, item_index):
        index = self.indicies[item_index]
        image_filenames = [f"images_and_masks/{index}.png"]
        for inpainted_image_name in self.inpainted_image_names[index]:
            image_filenames.append("inpainted/" + inpainted_image_name)
        if self.split == "test":
            # this if condition is important to enforce fixed test set
            image1_image_path, image2_image_path = image_filenames
        else:
            image1_image_path, image2_image_path = random.sample(image_filenames, 2)
        if True or self.split == "test":
            flag = 1
            annotation_path = cache_data_triton(
                self.path_to_dataset, f"metadata/{index}.npy", self.machine
            )
            image1_image_as_tensor = self.read_image_as_tensor(
                cache_data_triton(self.path_to_dataset, image1_image_path, self.machine)
            )
            image2_image_as_tensor = self.read_image_as_tensor(
                cache_data_triton(self.path_to_dataset, image2_image_path, self.machine)
            )
            image2_image_as_tensor = K.geometry.transform.resize(
                image2_image_as_tensor, image1_image_as_tensor.shape[-2:]
            )
            annotations = np.load(annotation_path, allow_pickle=True)
            image1_image_inpainted_objects = self.get_inpainted_objects_bitmap_from_image_path(
                image1_image_path, len(annotations)
            )
            image2_image_inpainted_objects = self.get_inpainted_objects_bitmap_from_image_path(
                image2_image_path, len(annotations)
            )
            changed_objects = image1_image_inpainted_objects ^ image2_image_inpainted_objects
            change_objects_indices = np.array(
                [x == "1" for x in bin(changed_objects)[2:].zfill(len(annotations))]
            )
            annotations = annotations[change_objects_indices]
        else:
            flag = 2
            random_ = False
            image1_image_as_tensor = self.read_image_as_tensor(
                cache_data_triton(self.path_to_dataset, image1_image_path, self.machine)
            )
            image2_image_as_tensor = deepcopy(image1_image_as_tensor)
            image2_image_as_tensor, annotations, file_index = self.add_random_objects(
                image2_image_as_tensor, item_index
            )
            if random.random() < 0.5:
                random_ = True
                image1_image_as_tensor, image2_image_as_tensor = (
                    image2_image_as_tensor,
                    image1_image_as_tensor,
                )
        # 在这里对得出图像的depth信息
        # 然后将depth与图像拼接起来 放入到随机变换中
        # 看是否能实现depth变换

        depth_path = os.path.join(self.path_to_dataset, "depth")
        if not os.path.exists(depth_path):
            os.makedirs(depth_path)

        if flag == 2:
            if random_:
                filename1 = image1_image_path.split('/')[-1][:-4] + '_' + str(flag) + '_' + '1' + '_' + str(file_index)
                filename2 = image1_image_path.split('/')[-1][:-4] + '_' + str(flag) + '_' + '1'
            else:
                filename1 = image1_image_path.split('/')[-1][:-4] + '_' + str(flag) + '_' + '0'
                filename2 = image1_image_path.split('/')[-1][:-4] + '_' + str(flag) + '_' + '0' + '_' + str(file_index)
        else:
            filename1 = image1_image_path.split('/')[-1][:-4] + '_' + str(flag)
            filename2 = image2_image_path.split('/')[-1][:-4] + '_' + str(flag)

        depth1_file = os.path.join(depth_path, f"{filename1}.pt")
        depth2_file = os.path.join(depth_path, f"{filename2}.pt")
        # depth1
        if os.path.isfile(depth1_file):
            with open(depth1_file, "rb") as f:
                depth1 = torch.load(f)
            #     做个判断 是否读入的depth是我现在想要的
            if depth1.shape != image1_image_as_tensor.shape[-2:]:
                depth1 = self.depth_predictor.infer(image1_image_as_tensor.unsqueeze(0)).squeeze()  # [w,h]
                with open(depth1_file, "wb") as f:
                    torch.save(depth1, f)
        else:
            # 对train 和 val 数据集进行测试depth并保存， 方便后续继续使用
            depth1 = self.depth_predictor.infer(image1_image_as_tensor.unsqueeze(0)).squeeze()  # [w,h]
            with open(depth1_file, "wb") as f:
                torch.save(depth1, f)

        # depth2
        if os.path.isfile(depth2_file):
            with open(depth2_file, "rb") as f:
                depth2 = torch.load(f)

            #     做个判断 是否读入的depth是我现在想要的
            if depth2.shape != image2_image_as_tensor.shape[-2:]:
                depth2 = self.depth_predictor.infer(image2_image_as_tensor.unsqueeze(0)).squeeze()  # [w,h]
                with open(depth2_file, "wb") as f:
                    torch.save(depth2, f)
        else:

            # 对train 和 val 数据集进行测试depth并保存， 方便后续继续使用
            depth2 = self.depth_predictor.infer(image2_image_as_tensor.unsqueeze(0)).squeeze()
            with open(depth2_file, "wb") as f:
                torch.save(depth2, f)

        (
            image1_image_as_tensor,
            image2_image_as_tensor,
            depth1,
            depth2,
            transformed_image1_target_annotations,
            transformed_image2_target_annotations,
        ) = self.image_augmentations(
            image1_image_as_tensor,
            image2_image_as_tensor,
            depth1,
            depth2,
            annotations,
            index,
        )

        return {
            "image1": image1_image_as_tensor.squeeze(),
            "image2": image2_image_as_tensor.squeeze(),
            "image1_target_annotations": transformed_image1_target_annotations,
            "image2_target_annotations": transformed_image2_target_annotations,
            "depth1": depth1.squeeze(),
            "depth2": depth2.squeeze(),
            'registration_strategy': "2d",
            # "index": index,
            # "path": self.path_to_dataset
        }
