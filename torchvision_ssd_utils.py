import torch
import torch.nn.functional as F
import warnings
import torchvision

from collections import OrderedDict
from torch import nn, Tensor
from typing import Any, Dict, List, Optional, Tuple

box_coder = torchvision.models.detection._utils.BoxCoder(
            weights=(10., 10., 5., 5.))

neg_to_pos_ratio = 3
score_thresh: float = 0.01
topk_candidates: int = 400
nms_thresh: float = 0.45
detections_per_img: int = 200

def compute_loss(targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor], anchors: List[Tensor],
                 matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
    bbox_regression = head_outputs['bbox_regression']
    cls_logits = head_outputs['cls_logits']

    # Match original targets with default boxes
    num_foreground = 0
    bbox_loss = []
    cls_targets = []
    for (targets_per_image, bbox_regression_per_image, cls_logits_per_image, anchors_per_image,
         matched_idxs_per_image) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
        # produce the matching between boxes and targets
        foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
        foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
        num_foreground += foreground_matched_idxs_per_image.numel()

        # Calculate regression loss
        matched_gt_boxes_per_image = targets_per_image['boxes'][foreground_matched_idxs_per_image]
        bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
        anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
        target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
        bbox_loss.append(torch.nn.functional.smooth_l1_loss(
            bbox_regression_per_image,
            target_regression,
            reduction='sum'
        ))

        # Estimate ground truth for class targets
        gt_classes_target = torch.zeros((cls_logits_per_image.size(0), ), dtype=targets_per_image['labels'].dtype,
                                        device=targets_per_image['labels'].device)
        gt_classes_target[foreground_idxs_per_image] = \
            targets_per_image['labels'][foreground_matched_idxs_per_image]
        cls_targets.append(gt_classes_target)

    bbox_loss = torch.stack(bbox_loss)
    cls_targets = torch.stack(cls_targets)

    # Calculate classification loss
    num_classes = cls_logits.size(-1)
    cls_loss = F.cross_entropy(
        cls_logits.view(-1, num_classes),
        cls_targets.view(-1),
        reduction='none'
    ).view(cls_targets.size())

    # Hard Negative Sampling
    foreground_idxs = cls_targets > 0
    num_negative = neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
    negative_loss = cls_loss.clone()
    negative_loss[foreground_idxs] = -float('inf')  # use -inf to detect positive values that creeped in the sample
    values, idx = negative_loss.sort(1, descending=True)
    # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
    background_idxs = idx.sort(1)[1] < num_negative

    N = max(1, num_foreground)
    return {
        'bbox_regression': bbox_loss.sum() / N,
        'classification': (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
    }

def postprocess_detections(head_outputs: Dict[str, Tensor], image_anchors: List[Tensor],
                           image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]:
    bbox_regression = head_outputs['bbox_regression']
    pred_scores = F.softmax(head_outputs['cls_logits'], dim=-1)

    num_classes = pred_scores.size(-1)
    device = pred_scores.device

    detections: List[Dict[str, Tensor]] = []

    for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
        boxes = box_coder.decode_single(boxes, anchors)
        boxes = torchvision.ops.boxes.clip_boxes_to_image(boxes, image_shape)

        image_boxes = []
        image_scores = []
        image_labels = []
        for label in range(1, num_classes):
            score = scores[:, label]

            keep_idxs = score > score_thresh
            score = score[keep_idxs]
            box = boxes[keep_idxs]

            # keep only topk scoring predictions
            num_topk = min(topk_candidates, score.size(0))
            score, idxs = score.topk(num_topk)
            box = box[idxs]

            image_boxes.append(box)
            image_scores.append(score)
            image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        # non-maximum suppression
        keep = torchvision.ops.boxes.batched_nms(image_boxes, image_scores, image_labels, nms_thresh)
        keep = keep[:detections_per_img]

        detections.append({
            'boxes': image_boxes[keep],
            'scores': image_scores[keep],
            'labels': image_labels[keep],
        })
    return detections