# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.soft_box_head import build_soft_roi_box_head
from .mask_head.mask_head import build_roi_mask_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, teacher_cfg,student_cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.teacher_cfg = teacher_cfg.clone()
        self.student_cfg = student_cfg.clone()
        if student_cfg.MODEL.MASK_ON and student_cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.student_feature_extractor

    def forward(self, teacher_features, student_features, teacher_proposals, student_proposals, targets=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(teacher_features, student_features, teacher_proposals, student_proposals, targets)
        losses.update(loss_box)
        if self.student_cfg.MODEL.MASK_ON:
            mask_features = student_features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.student_cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)
        return x, detections, losses


def build_soft_roi_heads(teacher_cfg, student_cfg):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if not student_cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_soft_roi_box_head(teacher_cfg,student_cfg)))
    if student_cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(student_cfg)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(teacher_cfg, student_cfg, roi_heads)

    return roi_heads
