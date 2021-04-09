
"""
Specification

model archiver command

torch-model-archiver --model-name yolact --version 1.0 --model-file model.py --serialized-file yolact_base_54_800000.pth --export-path model_store --handler yolact_hanlder.py --extra-files "config.py,timer.py,backbone.py,transformer.py"

# Main model file
--model-file model.py
# utils file
--extra-files config.py timer.py backbone.py transformer.py
"""

import torch
import torch.nn.functional as F
import json
import base64
import io
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image

import boto3

import timer
from model import Yolact
from transformer import FastBaseTransform
from ts.torch_handler.base_handler import BaseHandler
from config import mask_type, cfg
from config import COLORS

class ModelHandler(BaseHandler):

    def __init__(self):
        self.model = None
        self.device = None
        self.context = None
        self.manifest = None
        self.map_location = None
        self.explain = False
        self.target = 0
        self.initialized = False

        # Yolact arguments
        self.score_threshold = 0.5
        self.crop_masks = True
        self.interpolate_mode = "bilinear"
        self.target_image = None
        self.display_lincomb = False
        self.mask_alpha = 0.45
        self.top_k = 15
        self.display_masks = True
        self.display_text = True
        self.display_bboxes = True
        self.display_scores = True


        # AWS arguments
        self.image_filename = "yolact_test.png"
        self.bucket = "murata-torchserve-db"
        self.bucket_location = "us-east-1"



        self.color_cache = defaultdict(lambda: {})
        self.class_color = False
        
    def initialize(self, context):
        """
        load model

        Args:
            context ([type]): [description]
        """

        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else self.map_location
        )

        self.manifest = context.manifest

        # load model
        model_pt_path = "yolact_base_54_800000.pth"  # ToDo: rename full path
        self.model = Yolact()
        self.model.load_weights(model_pt_path)
        self.model.to(self.device)
        self.model.eval()

        self.initialized = True

    def preprocess(self, data):
        """
        receive binary data and convert it into image

        if you use image as input, you should imitate this implementation. This is normal.

        Args:
            data ([type]): [description]
        """

        images = []

        for row in data:
            image = row.get("data") or row.get("body")

            if isinstance(image, str):
                image = base64.b64decode(image)
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                

            image = self.convert_image(image)

            images.append(image)

        if len(images) == 1:

            return images[0]  # batch input
        
        else:
            return torch.stack(images).to(self.device)


    def inference(self, model_input):
        """

        Args:
            model_input ([type]): [description]

        Returns:
            model_output: Yolact predictions dictionary
            format: 
                  1.model_output["detection"]
                       ["mask"]: mask coords
                       ["class"]: target object class number
                       ["scores"]: the probability of target object
                       ["bbox"]: the coords of bounding box
                  2.model_output["net"]
        """

        model_output = self.model(model_input)

        return model_output

    def postprocess(self, inference_output):
        # prep_display()
        image_numpy, json_output = self.prep_display(inference_output)

        # save image
        save_path = "/home/ubuntu/murata/Media2Cloud/Server/yolactserve/serve/image_dir/{}".format(self.image_filename)
        cv2.imwrite(save_path, image_numpy)
        # upload s3(if you do aws configure command, you should activate following code)
        # file_name = "/home/ubuntu/murata/Media2Cloud/Server/yolactserve/serve/image_dir/{}".format(self.image_filename)
        # url = self.upload_file(file_name)
        url = 'https://{}.s3.amazonaws.com/{}'.format(self.bucket, self.image_filename)
        json_output["url"] = url

        json_output = json.dumps(json_output)

        return json_output


    def upload_file(self, file_name):
        object_name = file_name
        s3_client = boto3.resource("s3")

        url = 'https://{}.s3.amazonaws.com/{}'.format(self.bucket, self.image_filename)

        s3_client.Bucket(self.bucket).upload_file(Filename=file_name, Key=self.image_filename)

        return url

    # helper sub function

    def pil2cv(self, image):
        new_image = np.array(image, dtype=np.uint8)
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
        return new_image

    def convert_image(self, image):
        image = self.pil2cv(image)  # convert opencv format
        self.target_image = image
        image = torch.from_numpy(image).to(self.device).float()

        batch = FastBaseTransform()(image.unsqueeze(0))

        return batch

    
    def crop(self, masks, boxes, padding:int=1):

        h, w, mask_dim = masks.size()
        x1, x2 = self.sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
        y1, y2 = self.sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

        rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, mask_dim)
        cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, mask_dim)

        masks_left = rows >= x1.view(1, 1, -1)
        masks_right = rows < x2.view(1, 1, -1)
        masks_up = cols >= y1.view(1, 1, -1)
        masks_down = cols < y2.view(1, 1, -1)

        crop_mask = masks * masks_right * masks_left * masks_up * masks_down

        return masks * crop_mask.float()

    
    def sanitize_coordinates(self, _x1, _x2, img_size: int, padding: int=0, cast: bool=True):

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

    # sub function for postprocess

    def get_color(self, j, classes, on_gpu=None):
        color_idx = (classes[j] * 5 if self.class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in self.color_cache[on_gpu]:
            return self.color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]

            color = (color[2], color[1], color[0])

            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.0
                self.color_cache[on_gpu][color_idx] = color
            else:
                color = torch.Tensor(color).float() / 255.0
                self.color_cache["cpu"][color_idx] = color

            return color
        

    def arrange_results(self, inference_output, h, w):
        predictions = inference_output[0]

        net = predictions["net"]
        dets = predictions["detection"]

        if dets is None:
            return [torch.Tensor()] * 4

        if self.score_threshold > 0:
            keep = dets["score"] > self.score_threshold

            for k in dets:
                if k != "proto":
                    dets[k] = dets[k][keep]

            if dets["score"].size(0) == 0:
                return [torch.Tensor()] * 4


        classes = dets["class"]
        boxes = dets['box']
        scores = dets['score']
        masks = dets['mask']

        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
            proto_data = dets['proto']

            masks = proto_data @ masks.t()

            masks = cfg.mask_proto_mask_activation(masks)

            if self.crop_masks:
                masks = self.crop(masks, boxes)

            masks = masks.permute(2, 0, 1).contiguous()

            if cfg.use_maskiou:
                with timer.env("maskiou_net"):
                    maskiou_p = net.maskiou_net(masks.unsqueeze(1))
                    maskiou_p = torch.gather(maskiou_p, dim=1, index=classes.unsqueeze(1)).squeeze(1)

                    if cfg.rescore_bbox:
                        scores = scores * maskiou_p
                    else:
                        scores = [scores, scores * maskiou_p]

            
            masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=self.interpolate_mode, align_corners=False)

            masks.gt_(0.5)

        boxes[:, 0], boxes[:, 2] = self.sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
        boxes[:, 1], boxes[:, 3] = self.sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
        boxes = boxes.long()

        if cfg.mask_type == mask_type.direct and cfg.eval_mask_branch:
            full_masks = torch.zeros(masks.size(0), h, w)
            for jdx in range(masks.size(0)):
                x1, y1, x2, y2 = boxes[jdx, :]
                masks_w = x2 - x1
                masks_h = y2 - y1

                if masks_w * masks_h <=0 or masks_w < 0:
                    continue

                mask = masks[jdx, :].view(1, 1, cfg.mask_size, cfg.mask_size)
                mask = F.interpolate(mask, (masks_h, masks_w), mode=self.interpolate_mode, align_corners=False)
                mask = mask.gt(0.5).float()
                full_masks[jdx, y1:y2, x1:x2] = mask

            masks = full_masks

        return classes, scores, boxes, masks

    def prep_display(self, inference_output):

        json_output = {}

        # prep_display()
        img_gpu = self.target_image
        if torch.cuda.is_available():
            img_gpu = torch.FloatTensor(img_gpu).cuda()
        img_gpu = img_gpu / 255.0
        h, w, _ = img_gpu.shape

        with timer.env("PostProcess"):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = self.arrange_results(inference_output, h, w)
            cfg.rescore_bbox = save

        with timer.env("Copy"):
            idx = t[1].argsort(0, descending=True)[:self.top_k]

            if cfg.eval_mask_branch:
                masks = t[3][idx.item()]

            # if cpu
            # classes, scores, boxes = [x[idx].detach().numpy() for x in t[:3]]
            # if gpu
            classes, scores, boxes = [x[idx].cpu().detach().numpy() for x in t[:3]]

        num_dets_to_consider = min(self.top_k, classes.shape[0])


        for j in range(num_dets_to_consider):
            if scores[j] < self.score_threshold:
                num_dets_to_consider = j
                break

        if self.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:

            masks = masks[:num_dets_to_consider, :, :, None]

            colors = torch.cat([self.get_color(j, classes, on_gpu=self.device).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
            masks_color = masks.repeat(1, 1, 1, 3) * colors * self.mask_alpha

            inv_alph_masks = masks * (-self.mask_alpha) + 1

            masks_color_summand = masks_color[0]

            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand


        # convert tensor image into numpy array
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if num_dets_to_consider == 0:

            return img_numpy, json_output


        if self.display_text or self.display_bboxes:

            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = self.get_color(j, classes).numpy().tolist()
                score = scores[j]

                if self.display_bboxes:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
                
                if self.display_text:
                    _class = cfg.dataset.class_names[classes[j]]

                    text_str = "%s: %.2f" % (_class, score) if self.display_scores else _class

                    json_output[str(_class)] = str(round(score, 2))

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    text_pt = (x1, y1-3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return img_numpy, json_output



# Main function
_service = ModelHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)
        
        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return [data]
    except Exception as e:
        raise e


