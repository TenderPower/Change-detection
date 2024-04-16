from etils import epath
import imageio
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor
import yaml
from easydict import EasyDict
from einops import rearrange
from PIL import Image
from pycocotools import mask as coco_mask_utils
from torchvision.transforms import transforms
import kornia as K


def get_easy_dict_from_yaml_file(path_to_yaml_file):
    """
    Reads a yaml and returns it as an easy dict.
    """
    with open(path_to_yaml_file, "r") as stream:
        yaml_file = yaml.safe_load(stream)
    return EasyDict(yaml_file)


def single_coco_annotation_to_mask_image(annotation, image_shape_as_hw):
    """
    Converts a single object annotation which can be polygons, uncompressed RLE,
    or RLE to binary mask.
    """
    h, w = image_shape_as_hw
    segm = annotation["segmentation"]
    if type(segm) == list:
        rles = coco_mask_utils.frPyObjects(segm, h, w)
        rle = coco_mask_utils.merge(rles)
    elif type(segm["counts"]) == list:
        rle = coco_mask_utils.frPyObjects(segm, h, w)
    else:
        rle = annotation["segmentation"]
    m = coco_mask_utils.decode(rle)
    return m


def coco_annotations_to_mask_np_array(list_of_annotations, image_shape_as_hw):
    """
    Given a list of object annotations, returns a single binary mask.
    """
    mask = np.zeros(image_shape_as_hw, dtype=bool)
    for annotation in list_of_annotations:
        object_mask = single_coco_annotation_to_mask_image(annotation, image_shape_as_hw)
        mask = np.maximum(object_mask, mask)
    return mask


def cache_data_triton(path_to_dataset, path_to_file, machine):
    """
    When running jobs on triton/slurm, automatically cache
    data in the temp folder.
    """
    if machine not in ["triton", "slurm"]:
        return os.path.join(path_to_dataset, path_to_file)
    caching_location = os.path.join("/tmp/", path_to_file)
    if not os.path.exists(caching_location):
        Path(caching_location).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(path_to_dataset, path_to_file), caching_location)
        os.sync()
    return caching_location


def tensor_to_PIL(tensor):
    image = tensor.cpu()
    image = image.squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    return image


def perspective_transform_masks(infor, transfor, img_sizes, proba_threshold=.55, criterion='masks'):
    """
    # change the masks according to the M
    :param infor: List[Dict{}]
    :param transfor: Tensor(N,3,3)
    :return: List[Tensor(N,h,w)]
    """
    tms = []

    def get_transform_mask(mask, H, w, h):
        mask = cv2.warpPerspective(mask, H, (w, h))
        return torch.from_numpy(mask).to("cuda")

    transfor = transfor.detach().cpu().numpy()
    if criterion == 'imasks':
        for mask, tf, img_size in zip(infor, transfor, img_sizes):
            h, w = img_size
            a = get_transform_mask(mask, tf, w, h)
            tms.append(a)
    else:
        for inf, trsf, img_size in zip(infor, transfor, img_sizes):
            h, w = img_size
            masks = inf['masks'] > proba_threshold
            numpy_masks = masks.squeeze().detach().cpu().numpy().astype(np.uint8)
            if numpy_masks.shape[0] != 0:
                a = [get_transform_mask(mask, trsf, w, h) for mask in numpy_masks]
                a = torch.stack(a, 0)
            else:
                a = torch.ones(0, h, w)
            tms.append(a)
    return tms


def get_easy_dict_from_yaml_file(path_to_yaml_file):
    """
    Reads a yaml and returns it as an easy dict.
    """
    import yaml
    from easydict import EasyDict

    with open(path_to_yaml_file, "r") as stream:
        yaml_file = yaml.safe_load(stream)
    return EasyDict(yaml_file)


def _read_depth_from_png(path_to_depth):
    with open(path_to_depth, "rb") as file:
        pil_image = Image.open(path_to_depth)
        image_as_tensor = pil_to_tensor(pil_image).float()
    if image_as_tensor.ndim == 3:
        image_as_tensor = image_as_tensor.squeeze(0)
    return image_as_tensor


def _read_depth_from_tiff(path_to_depth):
    filename = epath.Path(path_to_depth)
    img = imageio.imread(filename.read_bytes(), format="tiff")
    if img.ndim == 2:
        img = img[:, :, None]
    return K.image_to_tensor(img).float().squeeze()


def read_image_as_tensor(path_to_image):
    assert path_to_image is not None
    with open(path_to_image, "rb") as file:
        pil_image = Image.open(file).convert("RGB")
        image_as_tensor = pil_to_tensor(pil_image).float() / 255.0
    return image_as_tensor


def read_depth_as_tensor(path_to_depth):
    assert path_to_depth is not None
    if ".tiff" in path_to_depth:
        return _read_depth_from_tiff(path_to_depth)
    return _read_depth_from_png(path_to_depth)


def create_batch_from_metadata(metadata):
    list_of_items = metadata["batch"]
    batch = {}
    all_keys = [
        "image1",
        "image2",
        "depth1",
        "depth2",
        "intrinsics1",
        "intrinsics2",
        "position1",
        "position2",
        "rotation1",
        "rotation2",
        "transfm2d_1_to_2",
        "transfm2d_2_to_1",
        "registration_strategy"
    ]
    for key in all_keys:
        batch[key] = []
    for item in list_of_items:
        for key in all_keys:
            value = item.get(key, None)
            if value is None:
                batch[key].append(None)
                continue
            if "image" in key:
                value = read_image_as_tensor(value)
            if "depth" in key:
                value = read_depth_as_tensor(value)
            for k in ["position", "rotation", "intrinsics", "transfm2d"]:
                if k in key:
                    value = torch.tensor(np.load(value))
            batch[key].append(value)
    return batch
