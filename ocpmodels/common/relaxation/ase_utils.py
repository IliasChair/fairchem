"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.



Utilities to interface OCP models/trainers with the Atomic Simulation
Environment (ASE)
"""
from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.constraints import FixAtoms

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    load_config,
    setup_imports,
    setup_logging,
    update_config,
)
from ocpmodels.datasets import data_list_collater
from ocpmodels.preprocessing import AtomsToGraphs

if TYPE_CHECKING:
    from torch_geometric.data import Batch


def batch_to_atoms(batch: Batch):
    n_systems = batch.natoms.shape[0]
    natoms = batch.natoms.tolist()
    numbers = torch.split(batch.atomic_numbers, natoms)
    fixed = torch.split(batch.fixed, natoms)
    forces = torch.split(batch.force, natoms)
    positions = torch.split(batch.pos, natoms)
    tags = torch.split(batch.tags, natoms)
    cells = batch.cell
    energies = batch.y.tolist()

    atoms_objects = []
    for idx in range(n_systems):
        atoms = Atoms(
            numbers=numbers[idx].tolist(),
            positions=positions[idx].cpu().detach().numpy(),
            tags=tags[idx].tolist(),
            cell=cells[idx].cpu().detach().numpy(),
            constraint=FixAtoms(mask=fixed[idx].tolist()),
            pbc=[True, True, True],
        )
        calc = sp(
            atoms=atoms,
            energy=energies[idx],
            forces=forces[idx].cpu().detach().numpy(),
        )
        atoms.set_calculator(calc)
        atoms_objects.append(atoms)

    return atoms_objects


class OCPCalculator(Calculator):
    def __init__(
        self,
        config_yml: str | None = None,
        checkpoint_path: str | None = None,
        trainer: str | None = None,
        cpu: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        OCP-ASE Calculator

        Args:
            config_yml (str):
                Path to yaml config or could be a dictionary.
            checkpoint_path (str):
                Path to trained checkpoint.
            trainer (str):
                OCP trainer to be used. "forces" for S2EF, "energy" for IS2RE.
            cpu (bool):
                Whether to load and run the model on CPU. Set `False` for GPU.
        """
        setup_imports()
        setup_logging()
        super().__init__()

        # Either the config path or the checkpoint path needs to be provided
        assert config_yml or checkpoint_path is not None

        checkpoint = None
        if config_yml is not None:
            if isinstance(config_yml, str):
                config, duplicates_warning, duplicates_error = load_config(
                    config_yml
                )
                if len(duplicates_warning) > 0:
                    logging.warning(
                        f"Overwritten config parameters from included configs "
                        f"(non-included parameters take precedence): {duplicates_warning}"
                    )
                if len(duplicates_error) > 0:
                    raise ValueError(
                        f"Conflicting (duplicate) parameters in simultaneously "
                        f"included configs: {duplicates_error}"
                    )
            else:
                config = config_yml

            # Only keeps the train data that might have normalizer values
            if isinstance(config["dataset"], list):
                config["dataset"] = config["dataset"][0]
            elif isinstance(config["dataset"], dict):
                config["dataset"] = config["dataset"].get("train", None)
        else:
            # Loads the config from the checkpoint directly (always on CPU).
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device("cpu")
            )
            config = checkpoint["config"]

        if trainer is not None:
            config["trainer"] = trainer
        else:
            config["trainer"] = config.get("trainer", "ocp")

        if "model_attributes" in config:
            config["model_attributes"]["name"] = config.pop("model")
            config["model"] = config["model_attributes"]

        # for checkpoints with relaxation datasets defined, remove to avoid
        # unnecesarily trying to load that dataset
        if "relax_dataset" in config["task"]:
            del config["task"]["relax_dataset"]

        # Calculate the edge indices on the fly
        config["model"]["otf_graph"] = True

        # Save config so obj can be transported over network (pkl)
        config = update_config(config)
        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = checkpoint_path
        del config["dataset"]["src"]

        self.trainer = registry.get_trainer_class(config["trainer"])(
            task=config["task"],
            model=config["model"],
            dataset=[config["dataset"]],
            outputs=config["outputs"],
            loss_fns=config["loss_fns"],
            eval_metrics=config["eval_metrics"],
            optimizer=config["optim"],
            identifier="",
            slurm=config.get("slurm", {}),
            local_rank=config.get("local_rank", 0),
            is_debug=config.get("is_debug", True),
            cpu=cpu,
            amp=config.get("amp", False),
        )

        if checkpoint_path is not None:
            self.load_checkpoint(
                checkpoint_path=checkpoint_path, checkpoint=checkpoint
            )

        seed = seed if seed is not None else self.trainer.config["cmd"]["seed"]
        if seed is None:
            logging.warning(
                "No seed has been set in modelcheckpoint or OCPCalculator! Results may not be reproducible on re-run"
            )
        else:
            self.trainer.set_seed(seed)

        self.a2g = AtomsToGraphs(
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_pbc=True,
        )
        self.implemented_properties = list(self.config["outputs"].keys())

    def load_checkpoint(
        self, checkpoint_path: str, checkpoint: dict | None = None
    ) -> None:
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
            checkpoint: dict
                A pretrained checkpoint dict
        """
        checkpoint = checkpoint or {}
        try:
            self.trainer.load_checkpoint(checkpoint_path, checkpoint)
        except NotImplementedError:
            logging.warning("Unable to load checkpoint!")

    def calculate(
        self, atoms: Atoms | Batch, properties, system_changes
    ) -> None:
        """Calculate implemented properties for a single Atoms object or a Batch of them."""
        super().calculate(self, atoms, properties, system_changes)
        if isinstance(atoms, Atoms):
            data_object = self.a2g.convert(atoms)
            batch = data_list_collater([data_object], otf_graph=True)
        else:
            batch = atoms

        predictions = self.trainer.predict(
            batch, per_image=False, disable_tqdm=True
        )

        for key in predictions:
            _pred = predictions[key]
            _pred = _pred.item() if _pred.numel() == 1 else _pred.cpu().numpy()
            self.results[key] = _pred
