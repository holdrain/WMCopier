import cv2
import numpy as np
import torch
from imwatermark import WatermarkDecoder, WatermarkEncoder



def cv2torch(cv2image):
    image_rgb = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
    image_tensor /= 255.0
    return image_tensor


def torch2cv(torchimage):
    """
    torchimage is shape of (3,h,w) and value from 0 to 1
    """
    image_tensor_bgr = torchimage[[2, 1, 0], :, :]
    image_tensor_bgr = image_tensor_bgr * 255
    image_np_bgr = image_tensor_bgr.byte().numpy()
    image_np_bgr = np.transpose(image_np_bgr, (1, 2, 0))
    return image_np_bgr


class Watermarker:
    def encode(self, img_path, output_path, prompt=""):
        raise NotImplementedError

    def decode(self, img_path):
        raise NotImplementedError


class InvisibleWatermarker(Watermarker):
    def __init__(self, wm_text, method, wmtype="bits"):
        if method == "rivaGan":
            WatermarkEncoder.loadModel()
        self.method = method
        self.encoder = WatermarkEncoder()
        self.wm_type = wmtype
        self.wm_text = wm_text
        if wmtype == "bytes":
            wm_length = len(wm_text) * 8
        else:
            wm_length = len(wm_text)
        self.decoder = WatermarkDecoder(self.wm_type, wm_length)

    def encode(self, image_tensor):
        """
        note: dwtdct method just is deal with the case of batchsize = 1
        input: image_tensor is a tensor with shape of (1,3,h,w) and value from 0-1, message is (1,message_length)
        """
        # transform to cvimread type
        img = image_tensor.squeeze(0)
        image_np_bgr = torch2cv(img)

        self.encoder.set_watermark(self.wm_type, self.wm_text.encode("utf-8"))
        out = self.encoder.encode(image_np_bgr, self.method)
        return cv2torch(out).unsqueeze(0)

    def decode(self, img_tensor):
        img = img_tensor.squeeze(0)
        image_np_bgr = torch2cv(img)
        wm_text_decode = self.decoder.decode(image_np_bgr, self.method)
        return torch.tensor(wm_text_decode, dtype=torch.float64)