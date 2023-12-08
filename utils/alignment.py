import cv2
import numpy
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy
import kornia as K
import shapely.affinity
import shapely.geometry
import torch
from shapely.validation import make_valid
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import pil_to_tensor
import math

from PIL import Image
import utils.general as general

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.2

def getDistance(point1, point2):
    distance = math.sqrt(math.pow((point1[0] - point2[0]), 2) + math.pow((point1[1] - point2[1]), 2))
    return distance

def getmedian(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2

def getaverg(data):
    return np.average(data)


def get_keypoints(image_scan, image_reference):
    record = {}
    no_descr = True
    # Convert images to grayscale
    # PS: The path does not contain Chinese.
    image1_Gray = cv2.cvtColor(image_scan, cv2.COLOR_BGR2GRAY)
    image2_Gray = cv2.cvtColor(image_reference, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(image1_Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2_Gray, None)

    # 创建BEBLID描述符
    beblid = cv2.xfeatures2d.BEBLID_create(0.75)

    # 使用BEBLID计算描述符
    descriptors1 = beblid.compute(image1_Gray, keypoints1)[1]
    descriptors2 = beblid.compute(image2_Gray, keypoints2)[1]

    # Match features.
    # matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    """
    matches为数据类型为list，包含了所匹配的特征点，list中每个元素的数据类型为DMatch。
    DMatch的数据结构包括：queryIdx、trainIdx、distance
    queryIdx：某一特征点在本帧图像的索引，即在img1特征点的索引；
    trainIdx：trainIdx是该特征点在另一张图像中相匹配的特征点的索引，即img2特征点的索引；
    distance：代表这一对匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近
    """
    if descriptors2 is None or descriptors1 is None:
        return record, no_descr
    else:
        no_descr = False
        # ---Using the  Brute-Force matcher---
        matches = matcher.match(descriptors1, descriptors2, None)
        # Sort matches by score 升
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)
        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        good_matches = matches[:numGoodMatches]
        # Extract location of good matches
        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
        for i, match in enumerate(good_matches):
            # queryIdx：某一特征点在本帧图像的索引，即在img1特征点的索引；
            # trainIdx：trainIdx是该特征点在另一张图像中相匹配的特征点的索引，即img2特征点的索引；
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        record['points1'] = points1
        record['points2'] = points2
        record['matches'] = good_matches
        record['kp1'] = keypoints1
        record['kp2'] = keypoints2

        return record, no_descr


def alignImages(image_scan, image_reference):
    '''
    Rotate and scale the images
    :param image_scan:
    :param image_reference:
    :return:
    '''
    H_default = np.array([[1., 0, 0], [0, 1, 0], [0, 0, 1]])
    keypoints_dict, criterion_perspect = get_keypoints(image_scan, image_reference)
    # Avoiding the occurrence of the phenomena that ORB can't find the descr
    if criterion_perspect:
        return image_scan, image_reference, H_default

    # Get keypoints of the two images
    # and perform perspective transformation
    points1 = keypoints_dict['points1']
    points2 = keypoints_dict['points2']
    if len(points1) > 5 or len(points2) > 5:
        # Homography
        # M_H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransacReprojThreshold=3.0)
        M_H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
        if M_H is not None:
            h, w, channels = image_reference.shape
            perspective_image_scan = cv2.warpPerspective(image_scan, M_H, (w, h))
            H_default = M_H
            finally_image_scan = perspective_image_scan
        else:
            finally_image_scan = image_scan
        # # -------------------------------------Showing----------------------------------
        # # Showing the transformed images
        # draw_params = dict(
        #     matchColor=(0, 255, 0),
        #     singlePointColor=None,
        #     matchesMask=mask.ravel().tolist(),
        #     flags=2
        # )
        # # Showing the keypoints on the images
        # img3 = cv2.drawMatches(image_scan, keypoints_dict['kp1'], image_reference, keypoints_dict['kp2'],
        #                        keypoints_dict["matches"], None, **draw_params)
        # plt.imshow(img3, 'gray')
        # plt.show()
        # # ------------------------------------------END---------------------------------
    else:
        finally_image_scan = image_scan
    # # -------------------------------------Ploting the image----------------------------------
    # images = [image_scan, image_reference, finally_image_scan, ]
    # for i in range(1, 4):
    #     plt.subplot(1, 3, i)
    #     plt.imshow(images[i - 1])
    # plt.show()
    # # -------------------------------------END----------------------------------
    return finally_image_scan, image_reference, H_default

def image_procession(images_scan, images_reference):
    '''
    Preprocess the images
    :param images_scan:Tensor[batchsize, 3, h, w,]
    :param images_reference:Tensor
    :return:
    '''
    # show the image
    """img_PIL = utils.general.tensor_to_PIL(images_scan[0])
    plt.imshow(img_PIL)
    plt.show()
    img_CV = cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    cv2.imshow('cv_im', img_CV)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    # Fasterrcnn
    # batchsize = images_scan.shape[0]
    # Maskrcnn
    batchsize = len(images_scan)
    im_scan2refer = []
    im_refer2refer = []
    im_refer2scan = []
    transfH = []
    invertransfH = []
    # im_or = []
    for i in range(batchsize):
        # Convert Tesnor to Cv
        img_scan_plt = general.tensor_to_PIL(images_scan[i])
        img_reference_plt = general.tensor_to_PIL(images_reference[i])
        image_scan_cv = cv2.cvtColor(numpy.asarray(img_scan_plt), cv2.COLOR_RGB2BGR)
        image_reference_cv = cv2.cvtColor(numpy.asarray(img_reference_plt), cv2.COLOR_RGB2BGR)
        # Align the images
        s2r, r2r, H = alignImages(image_scan_cv, image_reference_cv)
        inverH = torch.pinverse(torch.Tensor(H)).numpy()
        # add
        h, w, channels = image_scan_cv.shape
        r2s = cv2.warpPerspective(image_reference_cv, inverH, (w, h))
        im_scan2refer.append(
            (pil_to_tensor(Image.fromarray(cv2.cvtColor(s2r, cv2.COLOR_BGR2RGB))).float() / 255.0).cuda())
        im_refer2refer.append(
            (pil_to_tensor(Image.fromarray(cv2.cvtColor(r2r, cv2.COLOR_BGR2RGB))).float() / 255.0).cuda())
        im_refer2scan.append(
            (pil_to_tensor(Image.fromarray(cv2.cvtColor(r2s, cv2.COLOR_BGR2RGB))).float() / 255.0).cuda())
        transfH.append(torch.Tensor(H))
        invertransfH.append(torch.Tensor(inverH))

    transfH_tensor = torch.stack(transfH, dim=0).cuda()
    invertransfH_tensor = torch.stack(invertransfH, dim=0).cuda()

    return im_scan2refer, im_refer2refer, im_refer2scan, transfH_tensor, invertransfH_tensor


def resize_based_on_image_reference(im_scan, w, h):
    dim = (w, h)  # (w,h)
    resized_im_scan = cv2.resize(im_scan, dim, interpolation=cv2.INTER_CUBIC)
    # print('resized_im_scan：{}'.format(resized_im_scan.shape))  #(430, 640, 3)
    return resized_im_scan


def isAline(points: list[list[int]]):
    '''
    Determine whether three points are on the same line.
    :param points:
    :return: bool
    '''
    x1, y1 = points[0][0], points[0][1]
    x2, y2 = points[1][0], points[1][1]
    x3, y3 = points[2][0], points[2][1]
    return (x2 - x1) * (y3 - y2) == (y2 - y1) * (x3 - x2)


def show_imag_plt(images, length):
    for i in range(1, length + 1):
        plt.subplot(1, length, i)
        plt.imshow(images[i - 1])
    plt.show()
