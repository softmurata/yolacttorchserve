import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
# from utils import timer
from torchvision.models.resnet import Bottleneck
from itertools import product
from collections import defaultdict
from math import sqrt
from typing import List
from backbone import construct_backbone
from config import cfg, mask_type
import timer

use_jit = torch.cuda.device_count() <= 1
ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None:fn

prior_cache = defaultdict(lambda: None)

gpu_flag = torch.cuda.is_available()

class Concat(nn.Module):

    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params

    def forward(self, x):

        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)

class InterpolateModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):

        return F.interpolate(x, *self.args, **self.kwargs)

def make_net(in_channels, conf, include_last_relu=True):
    """
    helper function to take a config setting and turn it into a network
    Used by protonet and extrahead


    Args:
        in_channles ([type]): [description]
        conf ([type]): [description]
        include_last_relu (bool, optional): [description]. Defaults to True.

    Returns:
        (network, out_channels)
    """

    def make_layer(layer_cfg):
        nonlocal in_channels

        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == "cat":
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])

        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
            else:
                if num_channels is None:
                    layer = InterpolateModule(scale_factor=-kernel_size, mode="bilinear", align_corners=False, **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_config[2])

        in_channels = num_channels if num_channels is not None else in_channels


        return [layer, nn.ReLU(inplace=True)]

    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]


    return nn.Sequential(*(net)), in_channels


@torch.jit.script
def intersect(box_a, box_b):

    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)

    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
    box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))

    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
    box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))

    return torch.clamp(max_xy - min_xy, min=0).prod(3)

def jaccard(box_a, box_b, iscrowd:bool=False):
    use_batch = True

    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)

    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)

    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union

    return out if use_batch else out.squeeze(0)

# helper function for Detect Module
@torch.jit.script
def point_form(boxes):
    """
    convert prior boxes to (xmin, ymin, xmax, ymax)

    Args:
        boxes ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)

@torch.jit.script
def decode(loc, priors, use_yolo_regressors:bool=False):
    """
    b_x = (sigmoid(pred_x) - 0.5) / conv_w + prior_x
    b_y = (sigmoid(pred_y) - 0.5) / conv_h + prior_y
    b_w = prior_w * exp(loc_w)
    b_h = prior_h * exp(loc_h)

    Args:
        loc ([type]): [description]
        priors ([type]): [description]
        use_yolo_regressors (bool, optional): [description]. Defaults to False.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    if use_yolo_regressors:
        boxes = torch.cat((loc[:, :2] + priors[:, :2], priors[:, 2:] * torch.exp(loc[:, 2:])), 1)

        boxes = point_form(boxes)
    else:
        variances = [0.1, 0.2]
        boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)

        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

    return boxes

class Detect(object):
    """
    At test time, Detect is the final layer of SSD
    1. Decode location predictions
    2. apply non-maximum suppression to location predictions based on conf, scores and threshold to a top_k number of output predictions for both confidence score and locations, as the predicted masks

    Args:
        object ([type]): [description]
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k

        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError("nms threshold must b e non-negative")

        self.conf_thresh = conf_thresh

        self.use_cross_class_nms = False
        self.use_fast_nms = False

    def __call__(self, predictions, net):
        """[summary]

        Args:
            loc_data: (tensor)
                      shape: [batch, num_priors, 4]
            conf_data: (tensor)
                      shape: [batch, num_priors, num_classes]
            mask_data: (tensor)
                      shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) prior boxes and variances from priorbox layers
                      shape: [num_priors, 4]
            proto_data: (tensor) if using mask_type.lincomb, the prototype masks
                      shape: [batch, mask_h, mask_w, mask_dim]

        Returns:
            output shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            class_idx, confidence, bbox_coords, mask
        """

        loc_data = predictions["loc"]
        conf_data = predictions["conf"]
        mask_data = predictions["mask"]
        prior_data = predictions["priors"]

        proto_data = predictions["proto"] if "proto" in predictions else None
        inst_data = predictions["inst"] if "inst" in predictions else None

        out = []

        with timer.env("Detect"):
            batch_size = loc_data.size(0)
            num_priors = prior_data.size(0)

            conf_preds = conf_data.view(batch_size, num_priors, self.num_classes).transpose(2, 1).contiguous()
            

            for batch_idx in range(batch_size):
                decoded_boxes = decode(loc_data[batch_idx], prior_data)
                result = self.detect(batch_idx, conf_preds, decoded_boxes, mask_data, inst_data)

                if result is not None and proto_data is not None:
                    result['proto'] = proto_data[batch_idx]

                out.append({"detection": result, "net": net})

        return out


    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data):
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores, _ = torch.max(cur_scores, dim=0)

        keep = (conf_scores > self.conf_thresh)
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]

        if inst_data is not None:
            inst = inst_data[batch_idx, keep, :]

        if scores.size(1) == 0:
            return None

        if self.use_cross_class_nms:
            boxes, masks, classes, scores = self.cc_fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
        else:
            boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)

        result = {"box": boxes, "mask": masks, "class": classes, "score": scores}

        return result

    def cc_fast_nms(self, boxes, masks, scores, nms_thresh=0.5, top_k=200):
        scores, classes = torch.max(dim=0)

        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        boxes_idx = boxes[idx]

        iou = jaccard(boxes_idx, boxes_idx)
        iou.triu_(diagonal=1)

        iou_max, _ = torch.max(iou, dim=0)

        idx_out = idx[iou_max <= nms_thresh]

        return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]

    def fast_nms(self, boxes, masks, scores, nms_thresh, top_k):
        scores, idx = scores.sort(1, descending=True)
        idx = idx[:, :top_k]

        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes)
        iou.triu_(diagonal=1)

        iou_max, _ = iou.max(dim=1)

        keep = (iou_max <= nms_thresh)

        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        scores = scores[keep]
        masks = masks[keep]

        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores

    

# Sub NN class
class FastMaskIoUNet(ScriptModuleWrapper):

    def __init__(self):
        super().__init__()
        input_channels = 1
        last_layer = [(cfg.num_classes - 1, 1, {})]
        self.maskiou_net, _ = make_net(input_channels, cfg.maskiou_net + last_layer, include_last_relu=True)

    def forward(self, x):
        x = self.maskiou_net(x)
        maskiou_p = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze(-1).squeeze(-1)

        return maskiou_p


class FPN(ScriptModuleWrapper):
    """
    Parameters:
         cfg.fpn
           - num_feature(int) : the number of output featuresin fpn layers
           - interpolation_mode(str) : the mode to pass to F.interpolate
           - num_downsample(int) : the number of downsampled layer to add onto the selected layers

    Args:
        ScriptModuleWrapper ([type]): [description]
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample', 'relu_pred_layers',
                     'lat_layers', 'pred_layers', 'downsample_layers', 'relu_downsample_layers']

    def __init__(self, in_channels):
        super().__init__()

        self.lat_layers = nn.ModuleList([
            nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1) for x in reversed(in_channels)
        ])

        padding = 1 if cfg.fpn.pad else 0
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding) for _ in in_channels
        ])

        if cfg.fpn.use_conv_downsample:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, stride=2, padding=1)
                for _ in range(cfg.fpn.num_downsample)
            ])

        self.interpolation_mode = cfg.fpn.interpolation_mode
        self.num_downsample = cfg.fpn.num_downsample
        self.use_conv_downsample = cfg.fpn.use_conv_downsample
        self.relu_pred_layers = cfg.fpn.relu_pred_layers
        self.relu_downsample_layers = cfg.fpn.relu_downsample_layers
    
    @script_method_wrapper
    def forward(self, convouts:List[torch.Tensor]):

        out = []
        x = torch.zeros(1, device=convouts[0].device)

        # Initialize out
        for i in range(len(convouts)):
            out.append(x)
        
        j = len(convouts)

        # lattent layer
        for lat_layer in self.lat_layers:
            j -= 1
            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)

            x = x + lat_layer(convouts[j])
            out[j] = x

        # prediction layer
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = pred_layer(out[j])

            if self.relu_pred_layers:
                F.relu(out[j], inplace=True)


        cur_idx = len(out)  # current layer index

        # downsample
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
            
        else:
            for idx in range(self.num_downsample):
                out.append(F.max_pool2d(out[-1], 1, stride=2))
        
        if self.relu_downsample_layers:
            for idx in range(len(out) - cur_idx):
                out[idx] = F.relu(out[idx + cur_idx], inplace=False)

        return out

class PredictionModule(nn.Module):
    """
    DSSD(paper: https://arxiv.org/pdf/1701.06659.pdf)
    
    
    Args:
        nn ([type]): [description]
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
        super().__init__()

        self.num_classes = cfg.num_classes
        self.mask_dim = cfg.mask_dim
        self.num_priors = sum(len(x) * len(scales) for x in aspect_ratios)
        self.parent = [parent]
        self.index = index
        self.num_heads = cfg.num_heads

        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type.lincomb:
            self.mask_dim = self.mask_dim // self.num_heads

        if cfg.mask_proto_prototypes_as_features:
            in_channels += self.mask_dim
        
        if parent is None:
            if cfg.extra_head_net is None:
                out_channels = in_channels

            else:
                self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)

            if cfg.use_prediction_module:
                self.block = Bottleneck(out_channels, out_channels // 4)
                self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
                self.bn = nn.BatchNorm2d(out_channels)

            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, **cfg.head_layer_params)
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, **cfg.head_layer_params)

            if cfg.use_mask_scoring:
                self.score_layer = nn.Conv2d(out_channels, self.num_priors, **cfg.head_layer_params)

            if cfg.use_instance_coeff:
                self.inst_layer = nn.Conv2d(out_channels, self.num_priors * cfg.num_instance_coeffs, **cfg.head_layer_params)

            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]

            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate:
                self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, stride=1)

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None
        self.last_img_size = None

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [batch_size, channels, conv_h, conv_w]

        Returns:
            [type]: tuple(bbox_coords, class_confs, mask_output, prior_boxes)
                    - bbox_coords: [batch_size, conv_h * conv_w * num_priors, 4](x, y, w, h)
                    - class_confs: [batch_size, conv_h * conv_w * num_priors, num_classes]
                    - mask_output: [batch_size, conv_h * conv_w * num_priors, mask_dim]
                    - prior_boxes: [conv_h * conv_w * num_priors, 4]
        """
        src = self if self.parent[0] is None else self.parent[0]

        conv_h = x.size(2)
        conv_w = x.size(3)

        if cfg.extra_head_net is not None:
            x = src.upfeature(x)
        
        if cfg.use_prediction_module:
            a = src.block(x)
            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)

            x = a + b

        bbox_x = src.bbox_extra(x)
        mask_x = src.mask_extra(x)
        conf_x = src.conf_extra(x)

        # get bbox and confidence
        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)

        if cfg.eval_mask_branch:
            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)

        if cfg.use_mask_scoring:
            score = src.score_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)
        
        if cfg.use_instance_coeff:
            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, cfg.num_instance_coeffs)

        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if cfg.eval_mask_branch:
            if cfg.mask_type == mask_type.direct:
                mask = torch.sigmoid(mask)
            elif cfg.mask_type == mask_type.lincomb:
                mask = cfg.mask_proto_coeff_activation(mask)

                if cfg.mask_proto_coeff_gate:
                    gate = src.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                    mask = mask * torch.sigmoid(gate)

        
        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:
            mask = F.pad(mask, (self.index * self.mask_dim, (self.num_heads - self.index - 1) * self.mask_dim), mode="constant", value=0)

        priors = self.make_priors(conv_h, conv_w, x.device)

        preds = {"loc": bbox, "conf": conf, "mask": mask, "priors": priors}

        if cfg.use_mask_scoring:
            preds["score"] = score
        
        if cfg.use_instance_coeff:
            preds["inst"] = inst

        return preds

    def make_priors(self, conv_h, conv_w, device):

        global prior_cache
        size = (conv_h, conv_w)

        with timer.env("makepriors"):
            if self.last_img_size != (cfg._tmp_img_w, cfg._tmp_img_h):
                prior_data = []

                for j, i in product(range(conv_h), range(conv_w)):
                    x = (i + 0.5) / conv_w
                    y = (j + 0.5) / conv_h

                    for ars in self.aspect_ratios:
                        for scale in self.scales:
                            for ar in ars:
                                if not cfg.backbone.preapply_sqrt:
                                    ar = sqrt(ar)

                                if cfg.backbone.use_pixel_scales:
                                    w = scale * ar / cfg.max_size
                                    h = scale / ar / cfg.max_size
                                else:
                                    w = scale * ar / conv_w
                                    h = scale / ar / conv_h

                                if cfg.backbone.use_square_anchors:
                                    h = w

                                prior_data += [x, y, w, h]

                # cuda device conversion
                self.priors = torch.Tensor(prior_data).cuda().view(-1, 4).detach()
                self.priors.requires_grad = False
                self.last_img_size = (cfg._tmp_img_h, cfg._tmp_img_w)
                self.last_conv_size = (conv_w, conv_h)
                prior_cache[size] = None
            elif self.priors.device != device:
                if prior_cache[size] is None:
                    prior_cache[size] = {}
                
                if device not in prior_cache[size]:
                    prior_cache[size][device] = self.priors.to(device)

                self.priors = prior_cache[size][device]


        return self.priors



# Main NN class

class Yolact(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = construct_backbone(cfg.backbone)

        if cfg.freeze_bn:
            self.freeze_bn()

        # compute mask dim
        if cfg.mask_type == mask_type.direct:
            cfg.mask_dim = cfg.mask_size ** 2
        elif cfg.mask_type == mask_type.lincomb:
            if cfg.mask_proto_use_grid:
                self.grid = torch.Tensor(np.load(cfg.mask_proto_grid_file))
                self.num_grids = self.grid.size(0)
            else:
                self.num_grids = 0
            
            self.proto_src = cfg.mask_proto_src
            in_channels = 3
            if self.proto_src is None:
                in_channels = 3
            elif cfg.fpn is not None:
                in_channels = cfg.fpn.num_features
            else:
                in_channels = self.backbone.channels[self.proto_src]
            in_channels += self.num_grids

            self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)

            if cfg.mask_proto_bias:
                cfg.mask_dim += 1
        
        self.selected_layers = cfg.backbone.selected_layers
        src_channels = self.backbone.channels

        if cfg.use_maskiou:
            self.maskiou_net = FastMaskIoUNet()

        if cfg.fpn is not None:
            self.fpn = FPN([src_channels[i] for i in self.selected_layers])
            self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)

        self.prediction_layers = nn.ModuleList()
        cfg.num_heads = len(self.selected_layers)


        for idx, layer_idx in enumerate(self.selected_layers):
            parent = None
            if cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]
            
            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
            aspect_ratios=cfg.backbone.pred_aspect_ratios[idx],
            scales=cfg.backbone.pred_scales[idx],
            parent=parent,
            index=idx)
            self.prediction_layers.append(pred)

        # extra parameters for extra loss
        if cfg.use_class_existence_loss:
            # smallest layer selected
            self.class_existence_fc = nn.Linear(src_channels[-1], cfg.num_classes - 1)
        if cfg.use_semantic_segmentation_loss:
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg.num_classes - 1, kernel_size=1)

        # for evaluation
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k, conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)

    
    def save_weights(self, path):
        
        torch.save(self.state_dict, path)
    
    def load_weights(self, path):
        if gpu_flag:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device("cpu"))

        # for backward compatibility, remove these
        for key in list(state_dict.keys()):
            if key.startswith("backbone.layer") and not key.startswith("backbone.layers"):
                del state_dict[key]

            if key.startswith("fpn.downsample_layers."):
                if cfg.fpn is not None and int(key.split(".")[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]
            

        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')
        
        # Quick lambda to test if one list contains the other
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
            # Broke in 1.4 (see issue #292), where RecursiveScriptModule is the new star of the show.
            # Note that this might break with future pytorch updates, so let me know if it does
            is_script_conv = False
            if 'Script' in type(module).__name__:
                # 1.4 workaround: now there's an original_name member so just use that
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                # 1.3 workaround: check if this has the same constants as a conv module
                else:
                    is_script_conv = (
                        all_in(module.__dict__['_constants_set'], conv_constants)
                        and all_in(conv_constants, module.__dict__['_constants_set']))
            
            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv

            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    if cfg.use_focal_loss and 'conf_layer' in name:
                        if not cfg.use_sigmoid_focal_loss:
                            # Initialize the last layer as in the focal loss paper.
                            # Because we use softmax and not sigmoid, I had to derive an alternate expression
                            # on a notecard. Define pi to be the probability of outputting a foreground detection.
                            # Then let z = sum(exp(x)) - exp(x_0). Finally let c be the number of foreground classes.
                            # Chugging through the math, this gives us
                            #   x_0 = log(z * (1 - pi) / pi)    where 0 is the background class
                            #   x_i = log(z / c)                for all i > 0
                            # For simplicity (and because we have a degree of freedom here), set z = 1. Then we have
                            #   x_0 =  log((1 - pi) / pi)       note: don't split up the log for numerical stability
                            #   x_i = -log(c)                   for all i > 0
                            module.bias.data[0]  = np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                            module.bias.data[1:] = -np.log(module.bias.size(0) - 1)
                        else:
                            module.bias.data[0]  = -np.log(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi))
                            module.bias.data[1:] = -np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()
    
    def train(self, mode=True):
        super().train(mode)

        if cfg.freeze_bn:
            self.freeze_bn()

    
    def freeze_bn(self, enable=False):
        for module in self.modules:
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    def forward(self, x):
        """[summary]
        input shape => (batch_size, 3, img_h, img_w)
        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        print(x.size())
        _, _, img_h, img_w = x.size()
        cfg._tmp_img_h = img_h
        cfg._tmp_img_w = img_w

        with timer.env("backbone"):
            outs = self.backbone(x)
        
        if cfg.fpn is not None:
            with timer.env("fpn"):
                outs = [outs[i] for i in cfg.backbone.selected_layers]
                outs = self.fpn(outs)


        proto_out = None

        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
            with timer.env("proto"):
                proto_x = x if self.proto_src is None else outs[self.proto_src]

                if self.num_grids > 0:
                    grids = self.grid.repeat(proto_x.size[0], 1, 1, 1)
                    proto_x = torch.cat([proto_x, grids], dim=1)

                proto_out = self.proto_net(proto_x)
                proto_out = cfg.mask_proto_prototype_activation(proto_out)

                if cfg.mask_proto_prototypes_as_features:
                    proto_downsampled = proto_out.clone()

                    if cfg.mask_proto_prototypes_as_features_no_grad:
                        proto_downsampled = proto_out.detach()
                    
                
                proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

                if cfg.mask_proto_bias:
                    bias_shape = [x for x in proto_out.size()]
                    bias_shape[-1] = 1
                    proto_out = torch.cat([proto_out, torch.ones(*bias_shape)], -1)

        with timer.env("pred_heads"):
            pred_outs = {'loc':[], 'conf': [], 'mask': [], 'priors': []}

            if cfg.use_mask_scoring:
                pred_outs["score"] = []

            if cfg.use_instance_coeff:
                pred_outs['inst'] = []

            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
                pred_x = outs[idx]

                if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_prototypes_as_features:
                    proto_downsampled = F.interpolate(proto_downsampled, size=outs[idx].size()[2:], mode="bilinear", align_corners=False)
                    pred_x = torch.cat([pred_x, proto_downsampled], dim=1)

                if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]:
                    pred_layer.parent = [self.prediction_layers[0]]

                p = pred_layer(pred_x)

                for k, v in p.items():
                    pred_outs[k].append(v)

        
        for k, v in pred_outs.items():
            pred_outs[k] = torch.cat(v, -2)

        if proto_out is not None:
            pred_outs["proto"] = proto_out
        
        if self.training:

            if cfg.use_class_existence_loss:
                pred_outs["classes"] = self.class_existence_fc(outs[-1].mean(dim=(2, 3)))
            
            if cfg.use_semantic_segmentation_loss:
                pred_outs["segm"] = self.semantic_seg_conv(outs[0])

            
            return pred_outs

        else:
            if cfg.use_mask_scoring:
                pred_outs["score"] = torch.sigmoid(pred_outs["score"])

            if cfg.use_focal_loss:
                if cfg.use_sigmoid_focal_loss:
                    pred_outs["conf"] = torch.sigmoid(pred_outs["conf"])

                    if cfg.use_mask_scoring:
                        pred_outs["conf"] *= pred_outs["score"]

                elif cfg.use_objectness_score:
                    objectness = torch.sigmoid(pred_outs["conf"][:, :, 0])
                    pred_outs["conf"][:, :, 1:] = objectness[:, :, None] * F.softmax(pred_outs["conf"][:, :, 1:], -1)
                    pred_outs["conf"][:, :, 0] = 1 - objectness

                else:
                    pred_outs["conf"] = F.softmax(pred_outs["conf"], -1)

            else:
                if cfg.use_objectness_score:
                    objectness = torch.sigmoid(pred_outs["conf"][:, :, 0])
                    pred_outs["conf"][:, :, 1:] = (objectness > 0.10)[:, :, None] * F.softmax(pred_outs["conf"][:, :, 1:], -1)
                else:
                    pred_outs["conf"] = F.softmax(pred_outs["conf"], -1)

        
        return self.detect(pred_outs, self)






