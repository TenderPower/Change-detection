# from monodepth2 import monodepth2
import torch
import torch.nn as nn
import torch
MODEL_NAMES = [
    "mono_640x192",
    "stereo_640x192",
    "mono+stereo_640x192",
    "mono_no_pt_640x192",
    "stereo_no_pt_640x192",
    "mono+stereo_no_pt_640x192",
    "mono_1024x320",
    "stereo_1024x320",
    "mono+stereo_1024x320"
]


class Mono(monodepth2):
    def __init__(self, model_name=MODEL_NAMES[2], no_cuda=False, pred_metric_depth=False) :
        super().__init__(model_name,no_cuda,pred_metric_depth)

        # self.encoder = nn.DataParallel(self.encoder).to(self.device)
        # self.depth_decoder= nn.DataParallel(self.depth_decoder).to(self.device)

    def eval(self, input_image):
        with torch.no_grad():
            # Load image and preprocess
            b, c, original_height, original_width = input_image.shape
            # resize
            input_image = torch.nn.functional.interpolate(
                input_image, (self.feed_width, self.feed_height), mode="bilinear", align_corners=False)



            # PREDICTION
            input_image = input_image.to(self.device)
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False).squeeze()

        return disp_resized
