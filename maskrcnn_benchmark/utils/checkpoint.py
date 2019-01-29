# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch
import pdb
from maskrcnn_benchmark.utils.model_serialization import load_initial_state_dict
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
#            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
#            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def load_initial(self, tf=None, sf=None):
        self.logger.info("Loading initial weights from {},{}".format(tf,sf))
        teacher_checkpoint,student_checkpoint = self._load_initial_file(tf,sf)
        self._load_initial_model(teacher_checkpoint, student_checkpoint)
        return student_checkpoint


    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))

    def _load_initial_model(self, teacher_checkpoint, student_checkpoint):
        load_initial_state_dict(self.model, teacher_checkpoint.pop("model"), student_checkpoint.pop("model"))
    
class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        t_cfg,
        s_cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.t_cfg = t_cfg.clone()
        self.s_cfg = s_cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded

    def _load_initial_file(self, tf, sf):
        # load native detectron.pytorch checkpoint
        teacher_loaded = super(DetectronCheckpointer, self)._load_file(tf)
        student_loaded = super(DetectronCheckpointer, self)._load_file(sf) 
        if "model" not in teacher_loaded:
            teacher_loaded = dict(model=teacher_loaded)
        if "model" not in student_loaded:
            student_loaded = dict(model=student_loaded)
        return teacher_loaded,student_loaded
