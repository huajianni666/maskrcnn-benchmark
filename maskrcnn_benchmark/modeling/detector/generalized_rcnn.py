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
        
        self.attention_blocks = []
        fpn_channels = student_cfg.MODEL.BACKBONE.OUT_CHANNELS
        for idx in range(len(student_cfg.MODEL.RPN.ANCHOR_STRIDE)):
            attention_name = "fpn_attention{}".format(idx)      
            attention_module = nn.Conv2d(fpn_channels, 1, kernel_size=1, stride=1, bias = False)
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(attention_module.weight, a=1)
            self.add_module(attention_name, attention_module)
            self.attention_blocks.append(attention_name)
        """
        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
        """

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
        teacher_features = None
        teacher_proposals = None
        if self.training:
            with torch.no_grad():
                teacher_features = self.teacher_backbone(images.tensors)
                teacher_proposals, teacher_anchors, teacher_objectness, teacher_rpn_box_regression, null_losses = self.teacher_rpn(images, teacher_features, None, False)
        
        student_features = self.student_backbone(images.tensors)
        
        """
           backbone distillation
        """
        #student_norm = torch.nn.functional.normalize(student_feature.pow(2).mean(1).view(student_feature.size(0), -1))
        #teacher_norm = torch.nn.functional.normalize(teacher_feature.pow(2).mean(1).view(teacher_feature.size(0), -1))
        #single_stage_feature_loss = (student_norm - teacher_norm).pow(2).mean()
        if self.training:
            backbone_loss = []
            for teacher_feature,student_feature,attention_block in zip(teacher_features,student_features,self.attention_blocks):
                at_student_feature = getattr(self, attention_block)(student_feature)
                with torch.no_grad():
                    at_teacher_feature = getattr(self, attention_block)(teacher_feature)
                N, A, H, W = at_teacher_feature.shape
                single_stage_feature_loss = 0
                for i in range(N):
                    max_teacher_feature = at_teacher_feature[i,:,:,:]
                    teacher_feature_view = max_teacher_feature.view(max_teacher_feature.shape[2] * max_teacher_feature.shape[1])
                    pro_teacher_feature = nn.functional.normalize(teacher_feature_view,dim=0)
                    max_student_feature = at_student_feature[i,:,:,:]
                    student_feature_view = max_student_feature.view(max_student_feature.shape[2] * max_student_feature.shape[1])
                    pro_student_feature = nn.functional.normalize(student_feature_view,dim=0)
                    single_img_feature_loss = torch.norm(pro_student_feature - pro_teacher_feature, p = 2)
                    single_stage_feature_loss += single_img_feature_loss
                backbone_loss.append(single_stage_feature_loss / N)
                """bbox 0.302
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
            backbone_fpn_loss = 0
            for single_fpn_loss in backbone_loss:
                backbone_fpn_loss = backbone_fpn_loss + single_fpn_loss
            backbone_fpn_loss = {"backbone_fpn_loss": backbone_fpn_loss}#/len(backbone_loss)}
       
        student_proposals, student_anchors, student_objectness, student_rpn_box_regression, hard_proposal_losses = self.student_rpn(images, student_features, targets)
        ##rpn distilation no used
        if self.training:
            soft_loss_objectness, soft_loss_rpn_box_reg = self.soft_loss_evaluator(teacher_anchors, student_anchors, teacher_objectness, student_objectness, teacher_rpn_box_regression, student_rpn_box_regression, targets)
            alpha = 0.5
            beta = 0.5
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
