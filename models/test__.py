from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import torch
import numpy as np
from PIL import Image
from zoedepth.utils.misc import colorize
import matplotlib.pyplot as plt
import cv2
from zoedepth.utils.misc import pil_to_batched_tensor
import  time
# 记录开始时间
start_time = time.time()

# ZoeD_NK
# conf = get_config("zoedepth_nk", "infer")
# zoe = build_model(conf).eval()
zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).eval()


image = Image.open("/disk/ygk/pycharm_project/The-Change-You-Want-to-See-main/imgs/test_image.jpg").convert(
    "RGB")  # load
frame = cv2.imread('/disk/ygk/pycharm_project/The-Change-You-Want-to-See-main/imgs/test_image.jpg')

# depth = zoe.infer_pil(image)
X = pil_to_batched_tensor(image)
depth_tensor = zoe.infer(X)
# 记录结束时间
end_time = time.time()
# colored = colorize(depth)
# 计算运行时间
elapsed_time = end_time - start_time
# 打印运行时间
print("Program ran for {:.2f} seconds.".format(elapsed_time))
plt.ion()
# plt.draw()
plt.show(block=False)
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

ax1.imshow(frame)
ax2.imshow(depth_tensor.squeeze().detach().numpy())

plt.show()
cv2.destroyAllWindows()
