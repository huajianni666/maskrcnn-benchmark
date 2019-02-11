# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import pdb
from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.soft_roi_heads import build_soft_roi_heads
from ..rpn.softloss import make_soft_rpn_loss_evaluator
from maskrcnn_benchmark.modeling.box_coder import BoxCoder

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, teacher_cfg,student_cfg):
        super(GeneralizedRCNN, self).__init__()

        self.teacher_backbone = build_backbone(teacher_cfg)
        self.teacher_rpn = build_rpn(teacher_cfg)

        self.student_backbone = build_backbone(student_cfg)
        self.student_rpn = build_rpn(student_cfg)
        self.roi_heads = build_soft_roi_heads(teacher_cfg, student_cfg)
        
        soft_rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        soft_loss_evaluator = make_soft_rpn_loss_evaluator(student_cfg, soft_rpn_box_coder)
        self.soft_loss_evaluator = soft_loss_evaluator

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        with torch.no_grad():
            teacher_features = self.teacher_backbone(images.tensors)
            teacher_proposals, teacher_anchors, teacher_objectness, teacher_rpn_box_regression, null_losses = self.teacher_rpn(images, teacher_features, None, False)
        student_features = self.student_backbone(images.tensors)
        """
           backbone distillation
        """
        backbone_loss = []
        for teacher_feature,student_feature in zip(teacher_features,student_features):
            N, A, H, W = teacher_feature.shape
            single_stage_feature_loss = 0
            for i in range(N):
                mm = torch.nn.Softmax2d()
                max_teacher_feature = torch.max(teacher_feature[i,:,:,:],dim = 0)[0]
                max_teacher_feature_view = max_teacher_feature.view(max_teacher_feature.shape[0] * max_teacher_feature.shape[1])
                pro_teacher_feature = nn.functional.softmax(max_teacher_feature_view)
                max_student_feature = torch.max(student_feature[i,:,:,:],dim = 0)[0]
                max_student_feature_view = max_student_feature.view(max_student_feature.shape[0] * max_student_feature.shape[1])
                pro_student_feature = nn.functional.softmax(max_student_feature_view)
                single_img_feature_loss = torch.norm(pro_student_feature - pro_teacher_feature, p = 2)
                single_stage_feature_loss += single_img_feature_loss
            backbone_loss.append(single_stage_feature_loss / N)
            """
            N, A, H, W = teacher_feature.shape
            single_stage_feature_loss = 0
            for i in range(N):
                power_teacher_feature = torch.pow(teacher_feature[i,:,:,:],2)
                max_teacher_feature = torch.max(power_teacher_feature,dim = 0)[0]
                power_student_feature = torch.pow(student_feature[i,:,:,:],2)
                max_student_feature = torch.max(power_student_feature,dim = 0)[0]
                single_img_feature_loss = torch.norm(max_student_feature - max_teacher_feature, p = 2)
                single_stage_feature_loss += single_img_feature_loss / (H * W)
            backbone_loss.append(single_stage_feature_loss / N)
            """
            """
            N, A, H, W = teacher_feature.shape
            single_stage_feature_loss = 0
            for i in range(N):
                single_img_feature_loss = torch.norm((teacher_feature[i,:,:,:] - student_feature[i,:,:,:]), p =1)
                single_stage_feature_loss += single_img_feature_loss / (A * H * W)
            backbone_loss.append(single_stage_feature_loss / N)
            """
        backbone_fpn_loss = 0
        for single_fpn_loss in backbone_loss:
            backbone_fpn_loss = backbone_fpn_loss + single_fpn_loss
        backbone_fpn_loss = {"backbone_fpn_loss": backbone_fpn_loss/len(backbone_loss)}
       
        student_proposals, student_anchors, student_objectness, student_rpn_box_regression, hard_proposal_losses = self.student_rpn(images, student_features, targets)
        soft_loss_objectness=1
        soft_loss_rpn_box_reg=1
        #soft_loss_objectness, soft_loss_rpn_box_reg = self.soft_loss_evaluator(teacher_anchors, student_anchors, teacher_objectness, student_objectness, teacher_rpn_box_regression, student_rpn_box_regression, targets)
        alpha = 1
        beta = 1
        #print('hard_loss_objectness {},soft_loss_objectness {},hard_loss_rpn_box_reg {}, soft_loss_rpn_box_reg {}'.format(hard_proposal_losses["loss_objectness"],soft_loss_objectness,hard_proposal_losses["loss_rpn_box_reg"],soft_loss_rpn_box_reg))
        proposal_losses= {
           "hard_soft_loss_objectness": alpha * hard_proposal_losses["loss_objectness"] + (1-alpha) * soft_loss_objectness,
           "hard_soft_loss_rpn_box_reg": beta * hard_proposal_losses["loss_rpn_box_reg"] + (1-beta) * soft_loss_rpn_box_reg,
        }
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(teacher_features, student_features, teacher_proposals, student_proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = student_features
            result = student_proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(backbone_fpn_loss)
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
