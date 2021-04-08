import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from math import sqrt
import cv2
from config import MEANS, STD

# For Input
class Resize(object):
    @staticmethod
    def calc_size_preserve_ar(img_w, img_h, max_size):
        ratio = sqrt(img_w / img_h)
        w = max_size * ratio
        h = max_size / ratio

        return int(w), int(h)

    def __init__(self, resize_gt=True):
        self.resize_gt = resize_gt
        self.max_size = cfg.max_size
        self.preserve_aspect_ratio = cfg.preserve_aspect_ratio

    def __call__(self, image, masks, boxes, labels=None):
        img_h, img_w, _ = image.shape

        if self.preserve_aspect_ratio:
            width, height = Resize.calc_size_preserve_ar(img_w, img_h, self.max_size)
        else:
            width, height = self.max_size, self.max_size

        image = cv2.resize(image, (width, height))

        if self.resize_gt:
            masks = masks.transpose((1, 2, 0))
            masks = cv2.resize(masks, (width, height))

            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, 0)
            else:
                masks = masks.transpose((2, 0, 1))

            boxes[:, [0, 2]] *= (width / img_w)
            boxes[:, [1, 3]] *= (height / img_h)

        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        keep = (w > cfg.discard_box_width) * (h > cfg.discard_boxe_height)

        masks = masks[keep]
        boxes = boxes[keep]
        labels['labels'] = labels['labels'][keep]
        labels['num_crowds'] = (labels['labels'] < 0).sum()

        return image, masks, boxes, labels

class FastBaseTransform(nn.Module):

    def __init__(self):
        super().__init__()

        if torch.cuda.is_available():
            self.mean = torch.Tensor(MEANS).float().cuda()[None, :, None, None]
            self.std  = torch.Tensor( STD ).float().cuda()[None, :, None, None]
        else:
            self.mean = torch.Tensor(MEANS).float()[None, :, None, None]
            self.std  = torch.Tensor( STD ).float()[None, :, None, None]

        self.transform = cfg.backbone.transform

    def forward(self, img):

        print("image device:", img.device)
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)

        if cfg.preserve_aspect_ratio:
            _, h, w, _ = img.size()
            img_size = Resize.calc_size_preserve_ar(w, h, cfg.max_size)
            img_size = (img_size[1], img_size[0])
        else:
            img_size = (cfg.max_size, cfg.max_size)

        img = img.permute(0, 3, 1, 2).contiguous()

        img = F.interpolate(img, img_size, mode="bilinear", align_corners=False)


        if self.transform.normalize:
            img = (img - self.mean) / self.std
        
        elif self.transform.subtract_means:
            img = (img - self.mean)
        elif self.transform.to_float:
            img = img / 255

        if self.transform.channel_order != "RGB":
            raise NotImplementedError

        img = img[:, (2, 1, 0), :, :].contiguous()

        return img