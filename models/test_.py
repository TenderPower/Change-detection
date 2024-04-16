# from monodepth2 import monodepth2
from monodepth import Mono
import os
import numpy as np
import time
import PIL.Image as pil
import torch
from torchvision import transforms, datasets
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Webcam depth
    import cv2

    cap = cv2.VideoCapture(0)
    m = Mono()
    # m = monodepth2()
    # zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).eval()
    from PIL import Image

    image = Image.open("/disk/ygk/pycharm_project/The-Change-You-Want-to-See-main/imgs/test_image.jpg").convert("RGB")  # load
    # depth_numpy = zoe.infer_pil(image)  # as numpy
    from zoedepth.utils.misc import pil_to_batched_tensor
    X = pil_to_batched_tensor(image)
    # depth_tensor = zoe.infer(X) #[1,1,w,h]
    # depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

    plt.ion()
    # plt.draw()
    plt.show(block=False)

    ax1 = plt.subplot(1, 6, 1)
    ax2 = plt.subplot(1, 6, 2)
    ax3 = plt.subplot(1, 6, 3)
    # ax4 = plt.subplot(1, 6, 4)

    while (True):
        try:
            # Capture the video frame
            # by frame
            # ret, frame = cap.read()

            frame = cv2.imread('/disk/ygk/pycharm_project/The-Change-You-Want-to-See-main/imgs/test_image.jpg')

            # Display the resulting frame
            depth = m.eval(frame)
            # 将三通道的伪彩色深度图转换为单通道深度图
            depth_map = np.mean(depth, axis=2).astype(np.float32)
            depth_map_tensor = torch.from_numpy(depth_map) #[w,h]
            # 将单通道深度图转换为 float32 类型，并将其值缩放到 [0, 1] 范围内

            ax1.imshow(frame)
            ax2.imshow(depth)
            ax3.imshow(depth_map)
            # ax4.imshow(depth_numpy)

            # cv2.imwrite('tmps/frame.png', frame)
            # cv2.imwrite('tmps/depth.png', depth)
            # depth.save('depth.jpeg')

            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            # time.sleep(0.01)
            plt.pause(0.001)
        except:
            break

    plt.show()

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
