"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import datetime
import errno
import json
import logging
import os
import random
import sys
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any
from collections.abc import Iterator
from collections import defaultdict

import ase
import ase.io
import numpy as np
import numpy.typing as npt
import torch
import yaml
from ase.io.trajectory import Trajectory
from ase.vibrations import Vibrations
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import pickle

from fairchem.core import __version__
from fairchem.core.common import distutils, gp_utils
from fairchem.core.common.data_parallel import BalancedBatchSampler
from fairchem.core.common.logger import WandBSingletonLogger
from fairchem.core.common.registry import registry
from fairchem.core.common.slurm import (
    add_timestamp_id_to_submission_pickle,
)
from fairchem.core.common.typing import assert_is_instance as aii
from fairchem.core.common.typing import none_throws
from fairchem.core.common.utils import (
    get_commit_hash,
    get_weight_table,
    load_state_dict,
    match_state_dict,
    save_checkpoint,
    update_config,
)
from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.base_dataset import create_dataset
from fairchem.core.modules.evaluator import Evaluator
from fairchem.core.modules.exponential_moving_average import ExponentialMovingAverage
from fairchem.core.modules.loss import DDPLoss
from fairchem.core.modules.normalization.element_references import (
    create_element_references,
    load_references_from_config,
)
from fairchem.core.modules.normalization.normalizer import (
    create_normalizer,
    load_normalizers_from_config,
)
from fairchem.core.modules.scaling.compat import load_scales_compat
from fairchem.core.modules.scaling.util import ensure_fitted
from fairchem.core.modules.scheduler import LRScheduler
from fairchem.core.preprocessing import AtomsToGraphs

if TYPE_CHECKING:
    from collections.abc import Sequence


@registry.register_trainer("base")
class BaseTrainer(ABC):
    def __init__(
        self,
        task: dict[str, str | Any],
        model: dict[str, Any],
        outputs: dict[str, str | int],
        dataset: dict[str, str | float],
        optimizer: dict[str, str | float],
        loss_functions: dict[str, str | float],
        evaluation_metrics: dict[str, str],
        identifier: str,
        # TODO: dealing with local rank is dangerous
        # T201111838 remove this and use CUDA_VISIBILE_DEVICES instead so trainers don't need to know about which devie to use
        local_rank: int,
        timestamp_id: str | None = None,
        run_dir: str | None = None,
        is_debug: bool = False,
        print_every: int = 100,
        seed: int | None = None,
        logger: str = "wandb",
        amp: bool = False,
        cpu: bool = False,
        name: str = "ocp",
        slurm=None,
        gp_gpus: int | None = None,
        inference_only: bool = False,
    ) -> None:
        if slurm is None:
            slurm = {}
        self.name = name
        self.is_debug = is_debug
        self.cpu = cpu
        self.epoch = 0
        self.step = 0
        self.ema = None

        if torch.cuda.is_available() and not self.cpu:
            logging.info(f"local rank base: {local_rank}")
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
            self.cpu = True  # handle case when `--cpu` isn't specified
            # but there are no gpu devices available

        if run_dir is None:
            run_dir = os.getcwd()

        self.timestamp_id: str
        if timestamp_id is None:
            timestamp_id = self._get_timestamp(self.device, identifier)

        self.timestamp_id = none_throws(timestamp_id)

        commit_hash = get_commit_hash()

        logger_name = logger if isinstance(logger, str) else logger["name"]
        self.config = {
            "task": task,
            "trainer": name,
            "model": model,
            "outputs": outputs,
            "optim": optimizer,
            "loss_functions": loss_functions,
            "evaluation_metrics": evaluation_metrics,
            "logger": logger,
            "amp": amp,
            "gpus": distutils.get_world_size() if not self.cpu else 0,
            "cmd": {
                "identifier": identifier,
                "print_every": print_every,
                "seed": seed,
                "timestamp_id": self.timestamp_id,
                "commit": commit_hash,
                "version": __version__,
                "checkpoint_dir": os.path.join(
                    run_dir, "checkpoints", self.timestamp_id
                ),
                "results_dir": os.path.join(run_dir, "results", self.timestamp_id),
                "logs_dir": os.path.join(
                    run_dir, "logs", logger_name, self.timestamp_id
                ),
            },
            "slurm": slurm,
            "gp_gpus": gp_gpus,
        }
        # AMP Scaler
        self.scaler = torch.GradScaler("cuda") if amp and not self.cpu else None

        # Fill in SLURM information in config, if applicable
        if "SLURM_JOB_ID" in os.environ and "folder" in self.config["slurm"]:
            if "SLURM_ARRAY_JOB_ID" in os.environ:
                self.config["slurm"]["job_id"] = "{}_{}".format(
                    os.environ["SLURM_ARRAY_JOB_ID"],
                    os.environ["SLURM_ARRAY_TASK_ID"],
                )
            else:
                self.config["slurm"]["job_id"] = os.environ["SLURM_JOB_ID"]
            self.config["slurm"]["folder"] = self.config["slurm"]["folder"].replace(
                "%j", self.config["slurm"]["job_id"]
            )
            if distutils.is_master():
                add_timestamp_id_to_submission_pickle(
                    self.config["slurm"]["folder"],
                    self.config["slurm"]["job_id"],
                    self.timestamp_id,
                )

        # Define datasets
        if isinstance(dataset, list):
            if len(dataset) > 0:
                self.config["dataset"] = dataset[0]
            if len(dataset) > 1:
                self.config["val_dataset"] = dataset[1]
            if len(dataset) > 2:
                self.config["test_dataset"] = dataset[2]
        elif isinstance(dataset, dict):
            # or {} in cases where "dataset": None is explicitly defined
            self.config["dataset"] = dataset.get("train", {}) or {}
            self.config["val_dataset"] = dataset.get("val", {}) or {}
            self.config["test_dataset"] = dataset.get("test", {}) or {}
            self.config["relax_dataset"] = dataset.get("relax", {}) or {}
            self.config["freq_validation"] = (
                dataset.get("freq_validation", {}) or {}
            )
        else:
            self.config["dataset"] = dataset or {}

        # add empty dicts for missing datasets
        for dataset_name in ("val_dataset", "test_dataset", "relax_dataset"):
            if dataset_name not in self.config:
                self.config[dataset_name] = {}

        if not is_debug and distutils.is_master():
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)

        ### backwards compatability with OCP v<2.0
        self.config = update_config(self.config)

        if distutils.is_master():
            logging.info(yaml.dump(self.config, default_flow_style=False))

        # define attributes for readability
        self.elementrefs = {}
        self.normalizers = {}
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.best_val_metric = None
        self.primary_metric = None
        self.ema = None

        self.load(inference_only)

    @abstractmethod
    def train(self, disable_eval_tqdm: bool = False) -> None:
        """Run model training iterations."""

    @staticmethod
    def _get_timestamp(device: torch.device, suffix: str | None) -> str:
        now = datetime.datetime.now().timestamp()
        timestamp_tensor = torch.tensor(now).to(device)
        # create directories from master rank only
        distutils.broadcast(timestamp_tensor, 0)
        timestamp_str = datetime.datetime.fromtimestamp(
            timestamp_tensor.float().item()
        ).strftime("%Y-%m-%d-%H-%M-%S")
        if suffix:
            timestamp_str += "-" + suffix
        return timestamp_str

    def load(self, inference_only: bool) -> None:
        self.load_seed_from_config()
        self.load_logger()
        self.load_task()
        self.load_model()

        if inference_only is False:
            self.load_datasets()
            self.load_references_and_normalizers()
            self.load_loss()
            self.load_optimizer()
            self.load_extras()

        if self.config["optim"].get("load_datasets_and_model_then_exit", False):
            sys.exit(0)

    @staticmethod
    def set_seed(seed) -> None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_seed_from_config(self) -> None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        if self.config["cmd"]["seed"] is None:
            return
        self.set_seed(self.config["cmd"]["seed"])

    def load_logger(self) -> None:
        self.logger = None
        if not self.is_debug and distutils.is_master():
            assert self.config["logger"] is not None, "Specify logger in config"

            logger = self.config["logger"]
            logger_name = logger if isinstance(logger, str) else logger["name"]
            assert logger_name, "Specify logger name"

            if logger_name == "wandb_singleton":
                WandBSingletonLogger.init_wandb(
                    config=self.config,
                    run_id=self.config["cmd"]["timestamp_id"],
                    run_name=self.config["cmd"]["identifier"],
                    log_dir=self.config["cmd"]["logs_dir"],
                    project=self.config["logger"]["project"],
                    entity=self.config["logger"]["entity"],
                    group=self.config["logger"].get("group", ""),
                )
                self.logger = WandBSingletonLogger.get_instance()
            else:
                self.logger = registry.get_logger_class(logger_name)(self.config)

    def get_sampler(
        self, dataset, batch_size: int, shuffle: bool
    ) -> BalancedBatchSampler:
        balancing_mode = self.config["optim"].get("load_balancing", None)
        on_error = self.config["optim"].get("load_balancing_on_error", None)
        if balancing_mode is not None:
            if on_error is None:
                on_error = "raise"
        else:
            balancing_mode = "atoms"

        if on_error is None:
            on_error = "warn_and_no_balance"

        if gp_utils.initialized():
            num_replicas = gp_utils.get_dp_world_size()
            rank = gp_utils.get_dp_rank()
        else:
            num_replicas = distutils.get_world_size()
            rank = distutils.get_rank()
        return BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            on_error=on_error,
            seed=self.config["cmd"]["seed"],
        )

    def get_dataloader(self, dataset, sampler, workers=None) -> DataLoader:
        num_workers = (
            self.config["optim"]["num_workers"] if workers is None else workers
        )
        return DataLoader(
            dataset,
            collate_fn=self.collater,
            num_workers=num_workers,
            pin_memory=True,
            batch_sampler=sampler,
        )

    def load_datasets(self) -> None:
        self.collater = partial(
            data_list_collater, otf_graph=self.config["model"].get("otf_graph", False)
        )
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # This is hacky and scheduled to be removed next BE week
        # move ['X_split_settings'] to ['splits'][X]
        def convert_settings_to_split_settings(config, split_name):
            config = copy.deepcopy(config)  # make sure we dont modify the original
            if f"{split_name}_split_settings" in config:
                config["splits"] = {
                    split_name: config.pop(f"{split_name}_split_settings")
                }
            return config

        # load train, val, test datasets
        if "src" in self.config["dataset"]:
            logging.info(
                f"Loading dataset: {self.config['dataset'].get('format', 'lmdb')}"
            )

            self.train_dataset = create_dataset(
                convert_settings_to_split_settings(self.config["dataset"], "train"),
                "train",
            )
            self.train_sampler = self.get_sampler(
                self.train_dataset,
                self.config["optim"].get("batch_size", 1),
                shuffle=True,
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset,
                self.train_sampler,
            )

        if (
            "first_n" in self.config["dataset"]
            or "sample_n" in self.config["dataset"]
            or "max_atom" in self.config["dataset"]
        ):
            logging.warning(
                "Dataset attributes (first_n/sample_n/max_atom) passed to all datasets! Please don't do this, its dangerous!\n"
                + "Add them under each dataset 'train_split_settings'/'val_split_settings'/'test_split_settings'"
            )

        if "src" in self.config["val_dataset"]:
            if self.config["val_dataset"].get("use_train_settings", True):
                val_config = self.config["dataset"].copy()
                val_config.update(self.config["val_dataset"])
            else:
                val_config = self.config["val_dataset"]

            self.val_dataset = create_dataset(
                convert_settings_to_split_settings(val_config, "val"), "val"
            )
            self.val_sampler = self.get_sampler(
                self.val_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"].get("batch_size", 1)
                ),
                shuffle=False,
            )
            self.val_loader = self.get_dataloader(
                self.val_dataset,
                self.val_sampler,
            )

        if "src" in self.config["test_dataset"]:
            if self.config["test_dataset"].get("use_train_settings", True):
                test_config = self.config["dataset"].copy()
                test_config.update(self.config["test_dataset"])
            else:
                test_config = self.config["test_dataset"]

            self.test_dataset = create_dataset(
                convert_settings_to_split_settings(test_config, "test"), "test"
            )
            self.test_sampler = self.get_sampler(
                self.test_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"].get("batch_size", 1)
                ),
                shuffle=False,
            )
            self.test_loader = self.get_dataloader(
                self.test_dataset,
                self.test_sampler,
            )

        if self.config["relax_dataset"]:
            if self.config["relax_dataset"].get("use_train_settings", True):
                relax_config = self.config["dataset"].copy()
                relax_config.update(self.config["relax_dataset"])
            else:
                relax_config = self.config["relax_dataset"]

            self.relax_dataset = registry.get_dataset_class(
                relax_config.get("format", "lmdb")
            )(relax_config)
            self.relax_sampler = self.get_sampler(
                self.relax_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.relax_loader = self.get_dataloader(
                self.relax_dataset,
                self.relax_sampler,
            )

    def load_references_and_normalizers(self):
        """Load or create element references and normalizers from config"""
        # Is it troublesome that we assume any normalizer info is in train? What if there is no
        # training dataset? What happens if we just specify a test

        elementref_config = (
            self.config["dataset"].get("transforms", {}).get("element_references")
        )
        norms_config = self.config["dataset"].get("transforms", {}).get("normalizer")
        elementrefs, normalizers = {}, {}
        if distutils.is_master():
            if elementref_config is not None:
                # put them in a list to allow broadcasting python objects
                elementrefs = load_references_from_config(
                    elementref_config,
                    dataset=self.train_dataset,
                    seed=self.config["cmd"]["seed"],
                    checkpoint_dir=(
                        self.config["cmd"]["checkpoint_dir"]
                        if not self.is_debug
                        else None
                    ),
                )

            if norms_config is not None:
                normalizers = load_normalizers_from_config(
                    norms_config,
                    dataset=self.train_dataset,
                    seed=self.config["cmd"]["seed"],
                    checkpoint_dir=(
                        self.config["cmd"]["checkpoint_dir"]
                        if not self.is_debug
                        else None
                    ),
                    element_references=elementrefs,
                )

                # log out the values that will be used.
                for output, normalizer in normalizers.items():
                    logging.info(
                        f"Normalization values for output {output}: mean={normalizer.mean.item()}, rmsd={normalizer.rmsd.item()}."
                    )

        # put them in a list to broadcast them
        elementrefs, normalizers = [elementrefs], [normalizers]
        distutils.broadcast_object_list(
            object_list=elementrefs, src=0, device=self.device
        )
        distutils.broadcast_object_list(
            object_list=normalizers, src=0, device=self.device
        )
        # make sure element refs and normalizers are on this device
        self.elementrefs.update(
            {
                output: elementref.to(self.device)
                for output, elementref in elementrefs[0].items()
            }
        )
        self.normalizers.update(
            {
                output: normalizer.to(self.device)
                for output, normalizer in normalizers[0].items()
            }
        )

    def load_task(self):
        self.output_targets = {}
        for target_name in self.config["outputs"]:
            self.output_targets[target_name] = self.config["outputs"][target_name]
            if "decomposition" in self.config["outputs"][target_name]:
                for subtarget in self.config["outputs"][target_name]["decomposition"]:
                    self.output_targets[subtarget] = (
                        self.config["outputs"][target_name]["decomposition"]
                    )[subtarget]
                    self.output_targets[subtarget]["parent"] = target_name
                    # inherent properties if not available
                    if "level" not in self.output_targets[subtarget]:
                        self.output_targets[subtarget]["level"] = self.config[
                            "outputs"
                        ][target_name].get("level", "system")
                    if "train_on_free_atoms" not in self.output_targets[subtarget]:
                        self.output_targets[subtarget]["train_on_free_atoms"] = (
                            self.config[
                                "outputs"
                            ][target_name].get("train_on_free_atoms", True)
                        )
                    if "eval_on_free_atoms" not in self.output_targets[subtarget]:
                        self.output_targets[subtarget]["eval_on_free_atoms"] = (
                            self.config[
                                "outputs"
                            ][target_name].get("eval_on_free_atoms", True)
                        )

        # TODO: Assert that all targets, loss fn, metrics defined are consistent
        self.evaluation_metrics = self.config.get("evaluation_metrics", {})
        self.evaluator = Evaluator(
            task=self.name,
            eval_metrics=self.evaluation_metrics.get(
                "metrics", Evaluator.task_metrics.get(self.name, {})
            ),
        )

    def load_model(self) -> None:
        # Build model
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']['name']}")

        model_config_copy = copy.deepcopy(self.config["model"])
        model_name = model_config_copy.pop("name")
        self.model = registry.get_model_class(model_name)(
            **model_config_copy,
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())

        if distutils.is_master():
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{num_params} parameters."
            )

        if self.logger is not None:
            # only "watch" model if user specify watch: True because logging gradients
            # spews too much data into W&B and makes the UI slow to respond
            if "watch" in self.config["logger"]:
                self.logger.watch(
                    self.model, log_freq=int(self.config["logger"]["watch"])
                )
            self.logger.log_summary({"num_params": num_params})

        if distutils.initialized():
            self.model = DistributedDataParallel(
                self.model,
            )

    @property
    def _unwrapped_model(self):
        module = self.model
        while isinstance(module, DistributedDataParallel):
            module = module.module
        return module

    def load_checkpoint(
        self,
        checkpoint_path: str,
        checkpoint: dict | None = None,
        inference_only: bool = False,
    ) -> None:
        map_location = torch.device("cpu") if self.cpu else self.device
        if checkpoint is None:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    errno.ENOENT, "Checkpoint file not found", checkpoint_path
                )
            logging.info(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # attributes that are necessary for training and validation
        if inference_only is False:
            self.epoch = checkpoint.get("epoch", 0)
            self.step = checkpoint.get("step", 0)
            self.best_val_metric = checkpoint.get("best_val_metric", None)
            self.primary_metric = checkpoint.get("primary_metric", None)

            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
                self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            logging.info(
                "Loading checkpoint in inference-only mode, not loading keys associated with trainer state!"
            )

        if "ema" in checkpoint and checkpoint["ema"] is not None and self.ema:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None

        new_dict = match_state_dict(self.model.state_dict(), checkpoint["state_dict"])
        strict = self.config.get("task", {}).get("strict_load", True)
        load_state_dict(self.model, new_dict, strict=strict)

        scale_dict = checkpoint.get("scale_dict", None)
        if scale_dict:
            logging.info(
                "Overwriting scaling factors with those loaded from checkpoint. "
                "If you're generating predictions with a pretrained checkpoint, this is the correct behavior. "
                "To disable this, delete `scale_dict` from the checkpoint. "
            )
            load_scales_compat(self._unwrapped_model, scale_dict)

        for key, state_dict in checkpoint["normalizers"].items():
            ### Convert old normalizer keys to new target keys
            if key == "target":
                target_key = "energy"
            elif key == "grad_target":
                target_key = "forces"
            else:
                target_key = key

            if target_key not in self.normalizers:
                self.normalizers[target_key] = create_normalizer(state_dict=state_dict)
            else:
                mkeys = self.normalizers[target_key].load_state_dict(state_dict)
                assert len(mkeys.missing_keys) == 0
                assert len(mkeys.unexpected_keys) == 0

            self.normalizers[target_key].to(map_location)

        for key, state_dict in checkpoint.get("elementrefs", {}).items():
            if key not in self.elementrefs:
                self.elementrefs[key] = create_element_references(state_dict=state_dict)
            else:
                mkeys = self.elementrefs[key].load_state_dict(state_dict)
                assert len(mkeys.missing_keys) == 0
                assert len(mkeys.unexpected_keys) == 0

            self.elementrefs[key].to(map_location)

        if self.scaler and checkpoint["amp"]:
            self.scaler.load_state_dict(checkpoint["amp"])

    def load_loss(self) -> None:
        self.loss_functions = []
        for _idx, loss in enumerate(self.config["loss_functions"]):
            for target in loss:
                assert (
                    "fn" in loss[target]
                ), f"'fn' is not defined in the {target} loss config {loss[target]}."
                loss_name = loss[target].get("fn")
                assert (
                    "coefficient" in loss[target]
                ), f"'coefficient' is not defined in the {target} loss config {loss[target]}."
                coefficient = loss[target].get("coefficient")
                loss_reduction = loss[target].get("reduction")

                loss_fn = DDPLoss(loss_name, reduction=loss_reduction)

                self.loss_functions.append(
                    (target, {"fn": loss_fn, "coefficient": coefficient})
                )

    def load_optimizer(self) -> None:
        optimizer = getattr(torch.optim, self.config["optim"].get("optimizer", "AdamW"))
        optimizer_params = self.config["optim"].get("optimizer_params", {})

        weight_decay = optimizer_params.get("weight_decay", 0)
        if "weight_decay" in self.config["optim"]:
            weight_decay = self.config["optim"]["weight_decay"]
            logging.warning(
                "Using `weight_decay` from `optim` instead of `optim.optimizer_params`."
                "Please update your config to use `optim.optimizer_params.weight_decay`."
                "`optim.weight_decay` will soon be deprecated."
            )

        if weight_decay > 0:
            self.model_params_no_wd = {}
            if hasattr(self._unwrapped_model, "no_weight_decay"):
                self.model_params_no_wd = self._unwrapped_model.no_weight_decay()

            params_decay, params_no_decay, name_no_decay = [], [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                if any(
                    name.endswith(skip_name) for skip_name in self.model_params_no_wd
                ):
                    params_no_decay.append(param)
                    name_no_decay.append(name)
                else:
                    params_decay.append(param)

            if distutils.is_master():
                logging.info("Parameters without weight decay:")
                logging.info(name_no_decay)

            self.optimizer = optimizer(
                params=[
                    {"params": params_no_decay, "weight_decay": 0},
                    {"params": params_decay, "weight_decay": weight_decay},
                ],
                lr=self.config["optim"]["lr_initial"],
                **optimizer_params,
            )
        else:
            self.optimizer = optimizer(
                params=self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
                **optimizer_params,
            )

    def load_extras(self) -> None:
        self.scheduler = LRScheduler(self.optimizer, self.config["optim"])
        self.clip_grad_norm = aii(
            self.config["optim"].get("clip_grad_norm", None), (int, float)
        )
        self.ema_decay = aii(self.config["optim"].get("ema_decay"), float)
        if self.ema_decay:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                self.ema_decay,
            )
        else:
            self.ema = None

    def save(
        self,
        metrics=None,
        checkpoint_file: str = "checkpoint.pt",
        training_state: bool = True,
    ) -> str | None:
        if not self.is_debug and distutils.is_master():
            state = {
                "state_dict": self.model.state_dict(),
                "normalizers": {
                    key: value.state_dict() for key, value in self.normalizers.items()
                },
                "elementrefs": {
                    key: value.state_dict() for key, value in self.elementrefs.items()
                },
                "config": self.config,
                "val_metrics": metrics,
                "amp": self.scaler.state_dict() if self.scaler else None,
            }
            if training_state:
                state.update(
                    {
                        "epoch": self.epoch,
                        "step": self.step,
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": (
                            self.scheduler.scheduler.state_dict()
                            if self.scheduler.scheduler_type != "Null"
                            else None
                        ),
                        "config": self.config,
                        "ema": self.ema.state_dict() if self.ema else None,
                        "best_val_metric": self.best_val_metric,
                        "primary_metric": self.evaluation_metrics.get(
                            "primary_metric",
                            self.evaluator.task_primary_metric[self.name],
                        ),
                    },
                )
                ckpt_path = save_checkpoint(
                    state,
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
            else:
                if self.ema is not None:
                    self.ema.store()
                    self.ema.copy_to()
                ckpt_path = save_checkpoint(
                    state,
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
                if self.ema:
                    self.ema.restore()
            return ckpt_path
        return None

    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm: bool = True,
    ) -> None:
        if (
            "mae" in primary_metric
            and val_metrics[primary_metric]["metric"] < self.best_val_metric
        ) or (
            "mae" not in primary_metric
            and val_metrics[primary_metric]["metric"] > self.best_val_metric
        ):
            self.best_val_metric = val_metrics[primary_metric]["metric"]
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )
            if self.test_loader is not None:
                self.predict(
                    self.test_loader,
                    results_file="predictions",
                    disable_tqdm=disable_eval_tqdm,
                )

    def update_secondary_metric(
        self,
        secondary_metrics,
        val_metrics,
        disable_eval_tqdm: bool = True,
    ) -> None:
        for metric in secondary_metrics:
            if (
                ("mae" in metric or "rmse" in metric)
                and val_metrics[metric]["metric"] < self.best_secondary_metrics[metric]
            ) or (
                ("mae" not in metric and "rmse" not in metric)
                and val_metrics[metric]["metric"] > self.best_secondary_metrics[metric]
            ):
                self.best_secondary_metrics[metric] = val_metrics[metric]["metric"]
                self.save(
                metrics=val_metrics,
                checkpoint_file=f"best_{metric}_checkpoint.pt",
                training_state=False,
                )

    def _aggregate_metrics(self, metrics):
        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        return aggregated_metrics

    @torch.no_grad()
    def validate(self, split: str = "val", disable_tqdm: bool = False):
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master():
            logging.info(f"Evaluating on {split}.")

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        metrics = {}
        evaluator = Evaluator(
            task=self.name,
            eval_metrics=self.evaluation_metrics.get(
                "metrics", Evaluator.task_metrics.get(self.name, {})
            ),
        )

        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader

        # ----------------------------------------------
        # Add frequency validation
        freq_metrics = {}
        if self.config.get("freq_validation", False):
            freq_metrics = self.run_batch_freq_analysis(
                precomputed_data_dir=Path(
                    self.config["freq_validation"]["precomputed_data_dir"]),
                traj_path=Path(
                    self.config["freq_validation"]["traj_path"]),
                chunk_size=self.config["freq_validation"].get(
                    "chunk_size", 64),
                num_workers=self.config["freq_validation"].get(
                    "num_workers", min(8, os.cpu_count())),
                output_dir=Path(self.config["freq_validation"].get(
                    "output_dir", "vib")),
                disable_tqdm=disable_tqdm,
            )
        # ----------------------------------------------

        for _i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc=f"device {rank}",
            disable=disable_tqdm,
        ):
            # Forward.
            with torch.autocast("cuda", enabled=self.scaler is not None):
                batch.to(self.device)
                out = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)

        metrics.update(freq_metrics)
        metrics = self._aggregate_metrics(metrics)

        # Correct the freq_rmse metric after aggregation
        # The aggregation computes total/numel, which for RMSE's components (sum_sq_err, num_el) results in MSE.
        # We need to take the square root to get back to RMSE for logging.
        if "freq_rmse" in metrics and metrics["freq_rmse"]["numel"] > 0:
            metrics["freq_rmse"]["metric"] = np.sqrt(
                metrics["freq_rmse"]["metric"]
            )
        elif "freq_rmse" in metrics: # Handle case where numel might be 0
            metrics["freq_rmse"]["metric"] = np.nan

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": self.epoch})
        if distutils.is_master():
            log_str = [f"{k}: {v:.4f}" for k, v in log_dict.items()]
            logging.info(", ".join(log_str))

        # Make plots.
        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        if self.ema:
            self.ema.restore()

        return metrics

    @torch.no_grad()
    def run_batch_freq_analysis(
            self,
            precomputed_data_dir: Path,
            traj_path: Path,           # Kept for reading original mols/refs
            chunk_size: int,           # This will be used as batch_size
            num_workers: int,
            disable_tqdm: bool = False,
            output_dir: Path = Path("vib"),
        ) -> dict[str, Any]:
        """Run frequency analysis using chunked processing of precomputed data."""
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Create dataset using the precomputed data directory
        dataset = ChunkedVibrationDataset(precomputed_data_dir)

        # Create loader, using chunk_size as the batch_size for efficiency
        loader = DataLoader(
            dataset,
            batch_size=chunk_size, # Use chunk_size as DataLoader batch_size
            num_workers=num_workers,
            collate_fn=collate_vibration_batch
        )

        rank = distutils.get_rank()
        all_molecule_combined_forces = defaultdict(dict)

        # 1. Run all necessary calculations for frequency analysis in batches
        # and store the calculation output in json files
        # Process chunks
        for chunk in tqdm(
            loader,
            total=len(loader),
            position=rank,
            desc=f"Freq calc device {rank}",
            disable=disable_tqdm
        ):
            batch_data = chunk["batch_data"]
            displacements = chunk["displacements"]
            n_atoms_list = chunk["n_atoms_list"]

            # Get predictions
            with torch.autocast("cuda", enabled=self.scaler is not None):
                batch_data.to(self.device)
                out = self._forward(batch_data)

            forces_pred = out.get("forces")

            # Assuming batch_data is the same 'batch' object _denorm_preds expects
            # and "forces" is the correct target key.
            forces_denormalized = self._denorm_preds(
                "forces", forces_pred, batch_data
            )
            forces_batch = forces_denormalized.cpu().numpy()

            forces_batch = forces_batch * 27.211396132 # Convert to eV/A

            current_force_idx = 0
            for i in range(len(displacements)):
                n_a = n_atoms_list[i]
                displacement_obj = displacements[i]

                # Ensure forces_batch is not None before attempting to slice
                if forces_batch is not None:
                    end_force_idx = current_force_idx + n_a
                    disp_forces = forces_batch[current_force_idx:end_force_idx]
                    current_force_idx = end_force_idx

                    assert disp_forces.shape == (n_a, 3), (
                        f"Invalid forces shape {disp_forces.shape} for "
                        f"molecule {displacement_obj.vib.atoms.info.get('index', 'UnknownMol')} "
                        f"displacement {displacement_obj.name}, expected ({n_a}, 3)"
                    )

                    atom_index_str = displacement_obj.vib.atoms.info["index"]
                    displacement_name_str = displacement_obj.name

                    all_molecule_combined_forces[atom_index_str][displacement_name_str] = {
                        "forces": self.encode_ndarray(disp_forces)
                    }
                else:
                    # Handle the case where forces_batch was None for this chunk
                    # This might mean skipping this displacement or logging a specific error
                    logging.warning(
                        f"Skipping force storage for displacement {displacement_obj.name} "
                        f"of molecule {displacement_obj.vib.atoms.info.get('index', 'UnknownMol')} "
                        "due to missing batch forces."
                    )

        torch.cuda.empty_cache()

        # Write combined JSON files for each molecule
        for atom_index_str, combined_data in all_molecule_combined_forces.items():
            # ASE Vibrations(name="path/prefix") will look for "path/prefix/combined.json"
            molecule_specific_dir = output_dir / atom_index_str
            os.makedirs(molecule_specific_dir, exist_ok=True)
            json_path = molecule_specific_dir / "combined.json"
            with open(json_path, 'w') as f:
                json.dump(combined_data, f)

        # 2. Read over the original trajectory again using traj_path
        mols = ase.io.read(traj_path, index=":")

        FREQ_THRESHOLD = 100  # cm^-1

        # Initialize lists to collect all frequency pairs across molecules
        all_ref_freqs = []
        all_calc_freqs = []
        frequency_data_to_save = [] # List to store dicts for saving

        # array containing rmse, mae, percent_in_thres for each molecule
        # result_per_mol = np.zeros((len(mols), 3)) # No longer needed
        for i, mol in enumerate(mols):
            # ASE Vibrations(name="path/prefix") will look for "path/prefix/combined.json"
            vib_name_prefix = str(output_dir / mol.info["index"])
            vib = Vibrations(mol, name=vib_name_prefix)

            ref_freqs_mol = np.sort(np.array(mol.info["vib_freqs"]))
            calc_freqs_all = self.convert_freq_format(vib.get_frequencies())
            # only use the last len(ref_freqs) frequencies
            calc_freqs_mol = np.sort(calc_freqs_all[-len(ref_freqs_mol) :])

            # Ensure lengths match before appending (should usually be the case here)
            if len(ref_freqs_mol) == len(calc_freqs_mol):
                all_ref_freqs.extend(ref_freqs_mol)
                all_calc_freqs.extend(calc_freqs_mol)
                # Append data for saving
                for ref_freq, calc_freq in zip(ref_freqs_mol, calc_freqs_mol):
                    frequency_data_to_save.append({
                        "molecule_id": mol.info["index"],
                        "reference_frequency_cm-1": ref_freq,
                        "calculated_frequency_cm-1": calc_freq,
                    })
            else:
                logging.warning(
                    f"Mismatch in frequency count for molecule {mol.info.get('index', i)}. "
                    f"Ref: {len(ref_freqs_mol)}, Calc: {len(calc_freqs_mol)}. Skipping."
                )

        # Convert collected lists to numpy arrays
        all_ref_freqs_np = np.array(all_ref_freqs)
        all_calc_freqs_np = np.array(all_calc_freqs)

        # Calculate overall metrics if data was collected
        if all_ref_freqs_np.size > 0:
            errors = all_calc_freqs_np - all_ref_freqs_np
            abs_errors = np.abs(errors)
            squared_errors = errors**2

            total_freqs = len(all_ref_freqs_np)

            mae_total = np.sum(abs_errors)
            rmse_total_sum_sq = np.sum(squared_errors)
            threshold_total_count = np.sum(abs_errors < FREQ_THRESHOLD)

            mae_metric = mae_total / total_freqs
            rmse_metric = np.sqrt(rmse_total_sum_sq / total_freqs)
            percent_metric = (threshold_total_count / total_freqs) * 100

            final_metrics = {
                "freq_rmse": {
                    "total": rmse_total_sum_sq,
                    "numel": total_freqs,
                    "metric": rmse_metric,
                },
                "freq_mae": {
                    "total": mae_total,
                    "numel": total_freqs,
                    "metric": mae_metric,
                },
                "freq_percent_in_thres": {
                    "total": threshold_total_count,
                    "numel": total_freqs,
                    "metric": percent_metric,
                },
            }
        else:
            logging.warning("No frequency data collected for metric calculation.")
            # Return empty or zeroed metrics to avoid errors downstream
            final_metrics = {
                "freq_rmse": {"total": 0.0, "numel": 0, "metric": np.nan},
                "freq_mae": {"total": 0.0, "numel": 0, "metric": np.nan},
                "freq_percent_in_thres": {"total": 0.0, "numel": 0, "metric": np.nan},
            }

        # Save the collected frequency data if any exists
        if frequency_data_to_save:
            self._save_frequency_validation_results(frequency_data_to_save)

        logging.info(
            f"Frequency validation metrics: "
            f"Freq RMSE: {final_metrics['freq_rmse']['metric']:.4f}, "
            f"Freq MAE: {final_metrics['freq_mae']['metric']:.4f}, "
            f"Freq % in threshold: "
            f"{final_metrics['freq_percent_in_thres']['metric']:.4f}"
        )

        return final_metrics

    def _save_frequency_validation_results(self, data_list: list[dict]):
        """Saves the collected frequency data to a CSV file.

        Parameters
        ----------
        data_list : list[dict]
            List of dictionaries containing frequency data with keys:
            'molecule_id', 'reference_frequency_cm-1', 'calculated_frequency_cm-1'
        """
        if not distutils.is_master():
            return  # Only master process saves the file

        try:
            # Group frequencies by molecule
            grouped_data = {}
            for item in data_list:
                mol_id = item['molecule_id']
                if mol_id not in grouped_data:
                    grouped_data[mol_id] = {
                        'ref_freqs': [],
                        'calc_freqs': []
                    }
                grouped_data[mol_id]['ref_freqs'].append(item['reference_frequency_cm-1'])
                grouped_data[mol_id]['calc_freqs'].append(
                    item['calculated_frequency_cm-1'])

            # Create result rows in the desired format
            result_rows = []
            for i, (mol_id, data) in enumerate(grouped_data.items()):
                # Format frequency lists as strings
                ref_freqs_str = str(data['ref_freqs']).replace('[', '[').replace(']', ']')
                calc_freqs_str = str(data['calc_freqs']).replace('[', '[').\
                    replace(']', ']')

                result_rows.append({
                    'frame_index': i,
                    'dft_freqs': ref_freqs_str,
                    'ocp_freqs': calc_freqs_str,
                    'calc_time': 0.0  # Placeholder value
                })

            # Create DataFrame and save to CSV
            df = pd.DataFrame(result_rows)

            # Define output path using the trainer's results directory and epoch
            output_filename = f"frequency_validation_epoch_{self.epoch}.csv"
            output_path = Path(self.config["cmd"]["results_dir"]) / output_filename

            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            df.to_csv(output_path, index=False)
            logging.info(f"Saved frequency validation results to: {output_path}")

        except Exception as e:
            logging.error(f"Failed to save frequency validation results: {e}")

    def convert_freq_format(self, freqs):
        """Convert complex frequencies to real and sort ascending.

        Complex frequencies are converted to the negative absolute value of
        their imaginary part. Real frequencies are kept as is.
        """
        return np.sort([(-abs(f.imag) if np.iscomplex(f) else f.real)
                        for f in freqs])

    def encode_ndarray(self, arr):
        return {
            "__ndarray__": [
                list(arr.shape),  # Array shape
                str(arr.dtype),   # Data type
                arr.flatten().tolist()  # Flattened data
            ]
        }


    def _backward(self, loss) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        # Scale down the gradients of shared parameters
        if hasattr(self.model, "shared_parameters"):
            for p, factor in self.model.shared_parameters:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.detach().div_(factor)
                else:
                    if not hasattr(self, "warned_shared_param_no_grad"):
                        self.warned_shared_param_no_grad = True
                        logging.warning(
                            "Some shared parameters do not have a gradient. "
                            "Please check if all shared parameters are used "
                            "and point to PyTorch parameters."
                        )
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            # log unscaled weights and grads
            if isinstance(self.config["logger"], dict):
                log_weight_frequency = self.config["logger"].get("log_weight_table", -1)
            else:
                log_weight_frequency = -1  # not wandb
            if (
                self.logger is not None
                and distutils.is_master()
                and log_weight_frequency > 0
                and self.step % log_weight_frequency == 1  # log on 1 instead of 0
            ):
                columns, data = get_weight_table(self.model)
                self.logger.log_table(name="weight_table", cols=columns, data=data)

        if self.clip_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.clip_grad_norm,
            )
            if self.logger is not None:
                self.logger.log({"grad_norm": grad_norm}, step=self.step, split="train")
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.ema:
            self.ema.update()

    def save_results(
        self,
        predictions: dict[str, npt.NDArray],
        results_file: str | None,
        keys: Sequence[str] | None = None,
    ) -> None:
        if results_file is None:
            return
        if keys is None:
            keys = predictions.keys()

        results = distutils.gather_objects(predictions)
        distutils.synchronize()
        if distutils.is_master():
            gather_results = {
                key: list(chain(*(result[key] for result in results))) for key in keys
            }

            # Because of how distributed sampler works, some system ids
            # might be repeated to make no. of samples even across GPUs.
            _, idx = np.unique(gather_results["ids"], return_index=True)
            for k in keys:
                if "chunk_idx" in k:
                    gather_results[k] = np.cumsum([gather_results[k][i] for i in idx])[
                        :-1
                    ]
                else:
                    if f"{k}_chunk_idx" in keys or k == "forces":
                        gather_results[k] = np.concatenate(
                            [gather_results[k][i] for i in idx]
                        )
                    else:
                        gather_results[k] = np.array(
                            [gather_results[k][i] for i in idx]
                        )

            full_path = os.path.join(
                self.config["cmd"]["results_dir"], f"{self.name}_{results_file}.npz"
            )
            logging.info(f"Writing results to {full_path}")
            np.savez_compressed(full_path, **gather_results)

class ChunkedVibrationDataset(IterableDataset):
    def __init__(self, precomputed_data_dir: str | Path):
        """
        Loads precomputed vibration displacement and graph data.

        Reads .pkl files generated by precompute_vibration_data.py.
        Each .pkl file corresponds to one molecule from the original trajectory
        and contains a list of dictionaries:
        [{'graph_data': ..., 'displacement_obj': ..., 'n_atoms': ...}, ...]

        Parameters
        ----------
        precomputed_data_dir : str | Path
            Directory containing the precomputed .pkl files.
        """
        self.data_dir = Path(precomputed_data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Precomputed data directory not found: {self.data_dir}")

        # Removed self.delta and self.a2g as they are no longer needed.
        # chunk_size parameter removed from signature.

        # Load all precomputed items from .pkl files
        self.precomputed_items = []
        pkl_files = sorted(list(self.data_dir.glob("*.pkl")))
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in {self.data_dir}")

        logging.info(f"Loading precomputed vibration data from {len(pkl_files)} files in {self.data_dir}...")
        for pkl_file in tqdm(pkl_files, desc="Loading precomputed files"):
            try:
                with open(pkl_file, "rb") as f_in:
                    items_for_molecule = pickle.load(f_in)
                    if isinstance(items_for_molecule, list):
                        self.precomputed_items.extend(items_for_molecule)
                    else:
                        logging.warning(f"Unexpected data format in {pkl_file}. Expected list, got {type(items_for_molecule)}. Skipping.")
            except Exception as e:
                logging.error(f"Failed to load or process {pkl_file}: {e}")

        if not self.precomputed_items:
            raise ValueError(f"No valid precomputed items loaded from {self.data_dir}")
        logging.info(f"Finished loading {len(self.precomputed_items)} total precomputed displacement items.")

    def __len__(self):
        """Return the total number of precomputed displacement items."""
        return len(self.precomputed_items)

    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            iter_start = 0
            iter_end = len(self.precomputed_items)
        else:
            # Split precomputed items among workers
            num_items = len(self.precomputed_items)
            per_worker = int(np.ceil(num_items / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, num_items)

        # Yield individual precomputed displacement items
        for i in range(iter_start, iter_end):
            yield self.precomputed_items[i]

def collate_vibration_batch(batch_of_items: list[dict]):
    """Custom collate function for vibration batches from precomputed items."""
    # Each item in batch_of_items is a dict from ChunkedVibrationDataset.__iter__
    all_graph_data = [item["graph_data"] for item in batch_of_items]
    all_displacements = [item["displacement_obj"] for item in batch_of_items]
    # n_atoms_list derived from the graph_data, which is reliable
    n_atoms_list = [data.num_nodes for data in all_graph_data]

    # otf_graph should ideally be False if graphs are fully precomputed and require no further processing.
    # However, data_list_collater might still do other important batching operations.
    # Assuming otf_graph=True is safe, but could be optimized if graphs are truly final.
    batched_graphs = data_list_collater(all_graph_data, otf_graph=True)

    return {
        "batch_data": batched_graphs,
        "displacements": all_displacements, # List of ASE Displacement objects
        "n_atoms_list": n_atoms_list,
        # mol_idx and disp_idx are omitted; essential info is in Displacement objects.
    }