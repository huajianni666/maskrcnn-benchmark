# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat
import pdb
from maskrcnn_benchmark.layers import bounded_regression_loss
from maskrcnn_benchmark.layers import smooth_l1_loss #bounded_regression_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
import torch.nn as nn

class RPNsoftLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder

    def match_targets_to_anchors(self, anchor, target):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields([])
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            # discard anchors that go out of the boundaries of the image
            labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, teacher_anchors, student_anchors, teacher_objectness, student_objectness, teacher_box_regression, student_box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        student_anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in student_anchors]
        labels, regression_targets = self.prepare_targets(student_anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        student_objectness_flattened = []
        student_box_regression_flattened = []
        teacher_objectness_flattened = []
        teacher_box_regression_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for teacher_objectness_per_level, teacher_box_regression_per_level, student_objectness_per_level, student_box_regression_per_level in zip(
            teacher_objectness, teacher_box_regression, student_objectness, student_box_regression
        ):
            N, A, H, W = student_objectness_per_level.shape
            student_objectness_per_level = student_objectness_per_level.permute(0, 2, 3, 1).reshape(
                N, -1
            )
            student_box_regression_per_level = student_box_regression_per_level.view(N, -1, 4, H, W)
            student_box_regression_per_level = student_box_regression_per_level.permute(0, 3, 4, 1, 2)
            student_box_regression_per_level = student_box_regression_per_level.reshape(N, -1, 4)
            student_objectness_flattened.append(student_objectness_per_level)
            student_box_regression_flattened.append(student_box_regression_per_level)

            N, A, H, W = teacher_objectness_per_level.shape
            teacher_objectness_per_level = teacher_objectness_per_level.permute(0, 2, 3, 1).reshape(
                N, -1
            )
            teacher_box_regression_per_level = teacher_box_regression_per_level.view(N, -1, 4, H, W)
            teacher_box_regression_per_level = teacher_box_regression_per_level.permute(0, 3, 4, 1, 2)
            teacher_box_regression_per_level = teacher_box_regression_per_level.reshape(N, -1, 4)
            teacher_objectness_flattened.append(teacher_objectness_per_level)
            teacher_box_regression_flattened.append(teacher_box_regression_per_level)
        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        student_objectness = cat(student_objectness_flattened, dim=1).reshape(-1)
        student_box_regression = cat(student_box_regression_flattened, dim=1).reshape(-1, 4)

        teacher_objectness = cat(teacher_objectness_flattened, dim=1).reshape(-1)
        teacher_box_regression = cat(teacher_box_regression_flattened, dim=1).reshape(-1, 4)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        #soft_box_loss = bounded_regression_loss(
        #    teacher_box_regression[sampled_pos_inds],
        #    student_box_regression[sampled_pos_inds],
        #    regression_targets[sampled_pos_inds],
        #    beta=0.02,
        #    size_average=False,
        #)/(sampled_inds.numel())
        soft_box_loss = smooth_l1_loss(
            student_box_regression[sampled_pos_inds],
            teacher_box_regression[sampled_pos_inds],
            beta=1,
            size_average=False,
        ) / (sampled_pos_inds.numel())        
        
        teacher_objectness_label = (teacher_objectness[sampled_inds]).sigmoid()
#       student_objectness_label = (student_objectness[sampled_inds]).sigmoid()
#       soft_objectness_loss = T**2 * nn.KLDivLoss()(torch.log(student_objectness_label), teacher_objectness_label)
 
        soft_objectness_loss = F.binary_cross_entropy_with_logits(
            student_objectness[sampled_inds], teacher_objectness_label
        )

        return soft_objectness_loss, soft_box_loss


def make_soft_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNsoftLossComputation(matcher, fg_bg_sampler, box_coder)
    return loss_evaluator
