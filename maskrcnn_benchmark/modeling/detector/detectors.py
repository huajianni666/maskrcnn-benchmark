# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(tencher_cfg,student_cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[student_cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(tencher_cfg,student_cfg)
