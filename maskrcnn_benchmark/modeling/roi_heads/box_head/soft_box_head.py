# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import pdb
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .soft_loss import make_soft_roi_box_loss_evaluator


class SoftROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, teacher_cfg, student_cfg):
        super(SoftROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(student_cfg)
        self.teacher_feature_extractor = make_roi_box_feature_extractor(teacher_cfg)
        self.predictor = make_roi_box_predictor(student_cfg)
        self.teacher_predictor = make_roi_box_predictor(teacher_cfg)
        self.post_processor = make_roi_box_post_processor(student_cfg)
        self.loss_evaluator = make_soft_roi_box_loss_evaluator(student_cfg)

    def forward(self, teacher_features,student_features, teacher_proposals, student_proposals,targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                student_proposals = self.loss_evaluator.subsample(student_proposals, targets)
                tx = self.teacher_feature_extractor(teacher_features, student_proposals)
                teacher_class_logits, teacher_box_regression = self.teacher_predictor(tx)
        sx = self.feature_extractor(student_features, student_proposals)
        # final classifier that converts the features into predictions
        student_class_logits, student_box_regression = self.predictor(sx)

        if not self.training:
            result = self.post_processor((student_class_logits, student_box_regression), student_proposals)
            return sx, result, {}


        loss_classifier, loss_box_reg = self.loss_evaluator(
            [teacher_class_logits], [teacher_box_regression],
            [student_class_logits], [student_box_regression]
        )
        return (
            sx,
            student_proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_soft_roi_box_head(teacher_cfg, student_cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return SoftROIBoxHead(teacher_cfg, student_cfg)
