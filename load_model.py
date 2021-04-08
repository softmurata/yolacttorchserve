import argparse
import cv2
import numpy as np
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from model import Yolact
from config import cfg, mask_type
from config import COLORS
from config import MEANS, STD
import timer

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--trained_model", default="yolact_base_54_800000.pth", type=str, help="weight path")
parser.add_argument("--cuda", default=False, type=str2bool)
parser.add_argument("--image_path", type=str, default="kitten_small.jpg")
parser.add_argument("--score_threshold", type=float, default=0.5)
parser.add_argument("--top_k", type=int, default=15)
parser.add_argument("--display_fps", type=str2bool, default=True)
parser.add_argument("--display_lincomb", type=str2bool, default=False)
parser.add_argument("--crop", type=str2bool, default=False)
parser.add_argument("--display_masks", type=str2bool, default=True)
parser.add_argument("--display_text", type=str2bool, default=True)
parser.add_argument("--display_bboxes", default=True, type=str2bool)
parser.add_argument("--display_scores", default=True, type=str2bool)
args = parser.parse_args()


# Global variables
gpu_flag = torch.cuda.is_available()

color_cache = defaultdict(lambda: {})


# helper function for evaluation
@torch.jit.script
def sanitize_coordinates(_x1, _x2, img_size: int, padding: int=0, cast: bool=True):

    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()

    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)

    x1 = torch.clamp(x1-padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2

# helper function for postprocess
@torch.jit.script
def crop(masks, boxes, padding:int=1):
    """[summary]

    Args:
        masks ([type]): [h, w, mask_dim] tensor of masks
        boxes ([type]): [mask_dim, 4] tensor of bbox coords
        padding (int, optional): [description]. Defaults to 1.
    """

    h, w, mask_dim = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, mask_dim)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, mask_dim)

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks * masks_right * masks_up * masks_down

    return masks * crop_mask.float()

def postprocess(det_output, w, h, batch_idx=0, interpolation_mode="bilinear", visualize_lincomb=False, crop_masks=True, score_threshold=0):
    """[summary]

    Args:
        det_output ([type]): [description]
        w ([type]): [description]
        h ([type]): [description]
        batch_idx (int, optional): [description]. Defaults to 0.
        interpolation_mode (str, optional): [description]. Defaults to "bilinear".
        visualize_lincomb (bool, optional): [description]. Defaults to False.
        crop_masks (bool, optional): [description]. Defaults to True.
        score_threshold (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
        classes [num_det]: The class idx for each detection
        scores [num_det]: the confidence score for each detection
        boxes [num_det, 4]: The bounding box for each detection in absolute point form
        masks [num_det, h, w]: full image masks for each detection
    """

    dets = det_output[batch_idx]
    net = dets["net"]
    dets = dets["detection"]

    if dets is None:
        return [torch.Tensor()] * 4
    
    if score_threshold > 0:
        keep = dets["score"] > score_threshold

        for k in dets:
            if k != "proto":
                dets[k] = dets[k][keep]

        if dets["score"].size(0) == 0:
            return [torch.Tensor()] * 4

    classes = dets["class"]
    boxes = dets["box"]  #(1, 4)
    scores = dets['score'] 
    masks = dets["mask"]  # (1, 32)
    print("mask shape:", masks.shape)

    if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
        proto_data = dets["proto"]  # (138, 138, 1)

        masks = proto_data @ masks.t()  # (138, 138, 32)
        # Problem: shape is different

        masks = cfg.mask_proto_mask_activation(masks)

        if crop_masks:
            masks = crop(masks, boxes)

        masks = masks.permute(2, 0, 1).contiguous()

        if cfg.use_maskiou:
            with timer.env("maskiou_net"):
                with torch.no_grad():
                    maskiou_p = net.maskiou_net(masks.unsqueeze(1))
                    maskiou_p = torch.gather(maskiou_p, dim=1, index=classes.unsqueeze(1)).squeeze(1)

                    if cfg.rescore_bbox:
                        scroes = scores * maskiou_p
                    else:
                        scores = [scroes, scores * maskiou_p]

        
        masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False)

        # binarize masks
        masks.gt_(0.5)
    
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = boxes.long()

    if cfg.mask_type == mask_type.direct and cfg.eval_mask_branch:
        print("full masks")
        # upscale masks
        full_masks = torch.zeros(masks.size(0), h, w)

        for jdx in range(masks.size(0)):
            x1, y1, x2, y2 = boxes[jdx, :]
            masks_w = x2 - x1
            masks_h = y2- y1

            if masks_w * masks_h <=0 or masks_w < 0:
                continue

            mask = masks[jdx, :].view(1, 1, cfg.mask_size, cfg.mask_size)
            mask = F.interpolate(mask, (masks_h, masks_w), mode=interpolation_mode, align_corners=False)
            mask = mask.gt(0.5).float()
            full_masks[jdx, y1:y2, x1:x2] = mask
        
        masks = full_masks

    return classes, scores, boxes, masks

# helper function
def prep_display(preds, frame, h, w, class_color=False, mask_alpha=0.45, fps_str=""):
    img_gpu = frame
    img_gpu = img_gpu / 255.0
    h, w, _ = img_gpu.shape

    # ToDo: check the content of preds 

    with timer.env("PostProcess"):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(preds, w, h, visualize_lincomb=args.display_lincomb, crop_masks=args.crop, score_threshold=args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env("Copy"):
        idx = t[1].argsort(0, descending=True)[:args.top_k]

        if cfg.eval_mask_branch:
            masks = t[3][idx.item()]

        classes, scores, boxes = [x[idx].detach().numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    

    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    def get_color(j, on_gpu=None):
        global color_cache

        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        
        else:
            color = COLORS[color_idx]

            color =(color[2], color[1], color[0])

            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.0
                color_cache[on_gpu][color_idx] = color
            else:
                color = torch.Tensor(color).float() / 255.0
                color_cache["cpu"][color_idx] = color
            
            return color

    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # num_dets, h, w, 1
        # print(masks.shape)  # right: [1, 168, 224]
        masks = masks[:num_dets_to_consider, :, :, None]
        # print(masks.shape)  # right: [1, 168, 224, 1]

        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        inv_alph_masks = masks * (-mask_alpha) + 1

        masks_color_summand = masks_color[0]

        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    

    if args.display_fps:
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6

    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args.display_fps:
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if num_dets_to_consider == 0:
        return img_numpy

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j).numpy().tolist()
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
            
            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]

                text_str = "%s: %.2f" % (_class, score) if args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img_numpy
    
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

        if gpu_flag:
            self.mean = torch.Tensor(MEANS).float().cuda()[None, :, None, None]
            self.std  = torch.Tensor( STD ).float().cuda()[None, :, None, None]
        else:
            self.mean = torch.Tensor(MEANS).float()[None, :, None, None]
            self.std  = torch.Tensor( STD ).float()[None, :, None, None]

        self.transform = cfg.backbone.transform

    def forward(self, img):

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




if __name__ == "__main__":
    # load model
    model = Yolact()
    model.load_weights(args.trained_model)
    model.eval()

    if args.cuda:
        model = model.cuda()
    # inference

    # image
    frame = cv2.imread(args.image_path)
    if gpu_flag:
        frame = torch.from_numpy(frame).cuda().float()
    else:
        frame = torch.from_numpy(frame).float()

    batch = FastBaseTransform()(frame.unsqueeze(0))
    # FastBaseTransform()(frame)
    preds = model(batch)

    img_numpy = prep_display(preds, frame, None, None)

    cv2.imwrite("test.png", img_numpy)