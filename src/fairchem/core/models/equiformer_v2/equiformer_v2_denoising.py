from __future__ import annotations

import contextlib
import logging
import math
from functools import partial

import torch
import torch.nn as nn

from fairchem.core.common import gp_utils
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import (
    GraphModelMixin,
    HeadInterface,
)
from fairchem.core.models.scn.smearing import GaussianSmearing

with contextlib.suppress(ImportError):
    from e3nn import o3


import typing

from fairchem.core.models.equiformer_v2.equiformer_v2 import (
    EquiformerV2Backbone,
    eqv2_init_weights,
)
from .module_list import ModuleListInfo
from .radial_function import RadialFunction
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
)
from .transformer_block import (
    FeedForwardNetwork,
    SO2EquivariantGraphAttention,
    TransBlockV2,
)

if typing.TYPE_CHECKING:
    from torch_geometric.data.batch import Batch

    from fairchem.core.models.base import GraphData

# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773  # IS2RE: 100k, max_radius = 5, max_neighbors = 100


@registry.register_model("equiformer_v2_dens")
class EquiformerV2S_OC20_DenoisingPos(EquiformerV2Backbone):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        use_pbc_single (bool):         Process batch PBC graphs one at a time
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid

        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks

        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
        enforce_max_neighbors_strictly (bool):      When edges are subselected based on the `max_neighbors` arg, arbitrarily select amongst equidistant / degenerate edges to have exactly the correct number.
        avg_num_nodes (float):      Normalization factor for sum aggregation over nodes
        avg_degree (float):         Normalization factor for sum aggregation over edges
        use_force_encoding (bool):                  For ablation study, whether to encode forces during denoising positions. Default: True.
        use_noise_schedule_sigma_encoding (bool):   For ablation study, whether to encode the sigma (sampled std of Gaussian noises) during
                                                    denoising positions when `fixed_noise_std` = False in config files. Default: False.
        use_denoising_energy (bool):                For ablation study, whether to predict the energy of the original structure given
                                                    a corrupted structure. If `False`, we zero out the energy prediction. Default: True.

        use_energy_lin_ref (bool):  Whether to add the per-atom energy references during prediction.
                                    During training and validation, this should be kept `False` since we use the `lin_ref` parameter in the OC22 dataloader to subtract the per-atom linear references from the energy targets.
                                    During prediction (where we don't have energy targets), this can be set to `True` to add the per-atom linear references to the predicted energies.
        load_energy_lin_ref (bool): Whether to add nn.Parameters for the per-element energy references.
                                    This additional flag is there to ensure compatibility when strict-loading checkpoints, since the `use_energy_lin_ref` flag can be either True or False even if the model is trained with linear references.
                                    You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine.
    """

    def __init__(
        self,
        use_pbc: bool = True,
        use_pbc_single: bool = True,
        regress_forces: bool = True,
        otf_graph: bool = True,
        max_neighbors: int = 500,
        max_radius: float = 5.0,
        max_num_elements: int = 90,
        num_layers: int = 12,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 128,
        num_heads: int = 8,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 512,
        norm_type: str = "rms_norm_sh",
        lmax_list: list[int] | None = None,
        mmax_list: list[int] | None = None,
        grid_resolution: int | None = None,
        num_sphere_samples: int = 128,
        edge_channels: int = 128,
        use_atom_edge_embedding: bool = True,
        share_atom_edge_embedding: bool = False,
        use_m_share_rad: bool = False,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        attn_activation: str = "scaled_silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "scaled_silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        proj_drop: float = 0.0,
        weight_init: str = "normal",
        enforce_max_neighbors_strictly: bool = True,
        avg_num_nodes: float | None = _AVG_NUM_NODES,
        avg_degree: float | None = _AVG_DEGREE,
        use_denoising_stress=True,
        use_energy_lin_ref: bool | None = False,
        load_energy_lin_ref: bool | None = False,
        activation_checkpoint: bool | None = False,
        # following params are part of DeNS:
        use_force_encoding=True,
        use_noise_schedule_sigma_encoding=False,
        use_denoising_energy=True,
    ):
        super().__init__(
        use_pbc=use_pbc,
        use_pbc_single=use_pbc_single,
        regress_forces=regress_forces,
        otf_graph=otf_graph,
        max_neighbors=max_neighbors,
        max_radius=max_radius,
        max_num_elements=max_num_elements,
        num_layers=num_layers,
        sphere_channels=sphere_channels,
        attn_hidden_channels=attn_hidden_channels,
        num_heads=num_heads,
        attn_alpha_channels=attn_alpha_channels,
        attn_value_channels=attn_value_channels,
        ffn_hidden_channels=ffn_hidden_channels,
        norm_type=norm_type,
        lmax_list=lmax_list,
        mmax_list=mmax_list,
        grid_resolution=grid_resolution,
        num_sphere_samples=num_sphere_samples,
        edge_channels=edge_channels,
        use_atom_edge_embedding=use_atom_edge_embedding,
        share_atom_edge_embedding=share_atom_edge_embedding,
        use_m_share_rad=use_m_share_rad,
        distance_function=distance_function,
        num_distance_basis=num_distance_basis,
        attn_activation=attn_activation,
        use_s2_act_attn=use_s2_act_attn,
        use_attn_renorm=use_attn_renorm,
        ffn_activation=ffn_activation,
        use_gate_act=use_gate_act,
        use_grid_mlp=use_grid_mlp,
        use_sep_s2_act=use_sep_s2_act,
        alpha_drop=alpha_drop,
        drop_path_rate=drop_path_rate,
        proj_drop=proj_drop,
        weight_init=weight_init,
        enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
        avg_num_nodes=avg_num_nodes,
        avg_degree=avg_degree,
        use_energy_lin_ref=use_energy_lin_ref,
        load_energy_lin_ref=load_energy_lin_ref,
        activation_checkpoint=activation_checkpoint,
        )

        # for denoising position
        self.use_force_encoding = use_force_encoding
        self.use_noise_schedule_sigma_encoding = use_noise_schedule_sigma_encoding
        self.use_denoising_energy = use_denoising_energy
        self.use_denoising_stress = use_denoising_stress

        # for denoising position, encode node-wise forces as node features
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=max(self.lmax_list), p=1)
        self.force_embedding = SO3_LinearV2(
            in_features=1, out_features=self.sphere_channels, lmax=max(self.lmax_list)
        )

        if self.use_noise_schedule_sigma_encoding:
            self.noise_schedule_sigma_embedding = torch.nn.Linear(
                in_features=1, out_features=self.sphere_channels
            )

        self.apply(partial(eqv2_init_weights, weight_init=self.weight_init))

    @conditional_grad(torch.enable_grad())
    def forward(self, data: Batch) -> dict[str, torch.Tensor]:
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device
        atomic_numbers = data.atomic_numbers.long()
        assert (
            atomic_numbers.max().item() < self.max_num_elements
        ), "Atomic number exceeds that given in model config"
        graph = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        )

        data_batch = data.batch
        if gp_utils.initialized():
            (
                atomic_numbers,
                data_batch,
                node_offset,
                edge_index,
                edge_distance,
                edge_distance_vec,
            ) = self._init_gp_partitions(
                graph.atomic_numbers_full,
                graph.batch_full,
                graph.edge_index,
                graph.edge_distance,
                graph.edge_distance_vec,
            )
            graph.node_offset = node_offset
            graph.edge_index = edge_index
            graph.edge_distance = edge_distance
            graph.edge_distance_vec = edge_distance_vec

        ###############################################################
        # Entering Graph Parallel Region
        # after this point, if using gp, then node, edge tensors are split
        # across the graph parallel ranks, some full tensors such as
        # atomic_numbers_full are required because we need to index into the
        # full graph when computing edge embeddings or reducing nodes from neighbors
        #
        # all tensors that do not have the suffix "_full" refer to the partial tensors.
        # if not using gp, the full values are equal to the partial values
        # ie: atomic_numbers_full == atomic_numbers
        ###############################################################

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = SO3_Embedding(
            len(atomic_numbers),
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        #-------------------------DeNS-Section-Begin---------------------------
        # Node-wise force encoding during denoising positions
        num_atoms = len(graph.atomic_numbers_full)
        force_embedding = SO3_Embedding(
            num_atoms, self.lmax_list, 1, self.device, self.dtype
        )
        if hasattr(data, "denoising_pos_forward") and data.denoising_pos_forward:
            assert hasattr(data, "forces")
            force_data = data.forces
            force_sh = o3.spherical_harmonics(
                l=self.irreps_sh,
                x=force_data,
                normalize=True,
                normalization="component",
            )
            force_sh = force_sh.view(num_atoms, (max(self.lmax_list) + 1) ** 2, 1)
            force_norm = force_data.norm(dim=-1, keepdim=True)
            if hasattr(data, "noise_mask"):
                noise_mask_tensor = data.noise_mask.view(-1, 1, 1)
                force_sh = force_sh * noise_mask_tensor
        else:
            force_sh = torch.zeros(
                (num_atoms, (max(self.lmax_list) + 1) ** 2, 1),
                dtype=data.pos.dtype,
                device=data.pos.device,
            )
            force_norm = torch.zeros(
                (num_atoms, 1), dtype=data.pos.dtype, device=data.pos.device
            )

        if not self.use_force_encoding:
            # for ablation study, we enforce the force encoding to be zero.
            force_sh = torch.zeros(
                (num_atoms, (max(self.lmax_list) + 1) ** 2, 1),
                dtype=data.pos.dtype,
                device=data.pos.device,
            )
            force_norm = torch.zeros(
                (num_atoms, 1), dtype=data.pos.dtype, device=data.pos.device
            )

        force_norm = force_norm.view(-1, 1, 1)
        force_norm = force_norm / math.sqrt(
            3.0
        )  # since we use `component` normalization
        force_embedding.embedding = force_sh * force_norm

        force_embedding = self.force_embedding(force_embedding)
        x.embedding = x.embedding + force_embedding.embedding

        # noise schedule sigma encoding
        if self.use_noise_schedule_sigma_encoding:
            if hasattr(data, "denoising_pos_forward") and data.denoising_pos_forward:
                assert hasattr(data, "sigmas")
                sigmas = data.sigmas
            else:
                sigmas = torch.zeros(
                    (num_atoms, 1), dtype=data.pos.dtype, device=data.pos.device
                )
            noise_schedule_sigma_enbedding = self.noise_schedule_sigma_embedding(sigmas)
            x.embedding[:, 0, :] = x.embedding[:, 0, :] + noise_schedule_sigma_enbedding
        #-------------------------DeNS-Section-End-----------------------------

        # Edge encoding (distance and atom edge)
        graph.edge_distance = self.distance_expansion(graph.edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = graph.atomic_numbers_full[
                graph.edge_index[0]
            ]  # Source atom atomic number
            target_element = graph.atomic_numbers_full[
                graph.edge_index[1]
            ]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            graph.edge_distance = torch.cat(
                (graph.edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            graph.atomic_numbers_full,
            graph.edge_distance,
            graph.edge_index,
            len(atomic_numbers),
            graph.node_offset,
        )
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            if self.activation_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    self.blocks[i],
                    x,  # SO3_Embedding
                    graph.atomic_numbers_full,
                    graph.edge_distance,
                    graph.edge_index,
                    data_batch,  # for GraphDropPath
                    graph.node_offset,
                    use_reentrant=not self.training,
                )
            else:
                x = self.blocks[i](
                    x,  # SO3_Embedding
                    graph.atomic_numbers_full,
                    graph.edge_distance,
                    graph.edge_index,
                    batch=data_batch,  # for GraphDropPath
                    node_offset=graph.node_offset,
                )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        return {"node_embedding": x, "graph": graph}

@registry.register_model("equiformer_v2_energy_head_denoising")
class EquiformerV2EnergyHeadDenoising(nn.Module, HeadInterface):
    def __init__(self, backbone, reduce: str = "sum"):
        super().__init__()
        self.reduce = reduce
        self.avg_num_nodes = backbone.avg_num_nodes
        self.energy_block = FeedForwardNetwork(
            backbone.sphere_channels,
            backbone.ffn_hidden_channels,
            1,
            backbone.lmax_list,
            backbone.mmax_list,
            backbone.SO3_grid,
            backbone.ffn_activation,
            backbone.use_gate_act,
            backbone.use_grid_mlp,
            backbone.use_sep_s2_act,
        )
        # for denoisng energy
        self.use_denoising_energy = backbone.use_denoising_energy
        self.apply(partial(eqv2_init_weights, weight_init=backbone.weight_init))

    def forward(self, data: Batch, emb: dict[str, torch.Tensor | GraphData]):
        node_energy = self.energy_block(emb["node_embedding"])
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        if gp_utils.initialized(): #TODO: Check why gp_utils was removed
            node_energy = gp_utils.gather_from_model_parallel_region(node_energy, dim=0)
        energy = torch.zeros(
            len(data.natoms),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy.index_add_(0, data.batch, node_energy.view(-1))
        if self.reduce == "sum":
            energy = energy / self.avg_num_nodes
        elif self.reduce == "mean":
            energy = energy / data.natoms
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )

        #------------for denoising positions------------------
        # zero out denoising energy for ablation study
        if (
            hasattr(data, "denoising_pos_forward")
            and data.denoising_pos_forward
            and not self.use_denoising_energy
        ):
            energy = energy * 0.0
        #------------for denoising positions------------------
        return {"energy": energy}

@registry.register_model("equiformer_v2_force_head_denoising")
class EquiformerV2ForceHeadDenoising(nn.Module, HeadInterface):
    def __init__(self, backbone):
        super().__init__()

        self.activation_checkpoint = backbone.activation_checkpoint
        self.force_block = SO2EquivariantGraphAttention(
            backbone.sphere_channels,
            backbone.attn_hidden_channels,
            backbone.num_heads,
            backbone.attn_alpha_channels,
            backbone.attn_value_channels,
            1,
            backbone.lmax_list,
            backbone.mmax_list,
            backbone.SO3_rotation,
            backbone.mappingReduced,
            backbone.SO3_grid,
            backbone.max_num_elements,
            backbone.edge_channels_list,
            backbone.block_use_atom_edge_embedding,
            backbone.use_m_share_rad,
            backbone.attn_activation,
            backbone.use_s2_act_attn,
            backbone.use_attn_renorm,
            backbone.use_gate_act,
            backbone.use_sep_s2_act,
            alpha_drop=0.0,
        )

        self.denoising_pos_block = SO2EquivariantGraphAttention(
            backbone.sphere_channels,
            backbone.attn_hidden_channels,
            backbone.num_heads,
            backbone.attn_alpha_channels,
            backbone.attn_value_channels,
            1,
            backbone.lmax_list,
            backbone.mmax_list,
            backbone.SO3_rotation,
            backbone.mappingReduced,
            backbone.SO3_grid,
            backbone.max_num_elements,
            backbone.edge_channels_list,
            backbone.block_use_atom_edge_embedding,
            backbone.use_m_share_rad,
            backbone.attn_activation,
            backbone.use_s2_act_attn,
            backbone.use_attn_renorm,
            backbone.use_gate_act,
            backbone.use_sep_s2_act,
            alpha_drop=0.0,
        )

        self.apply(partial(eqv2_init_weights, weight_init=backbone.weight_init))

    def forward(self, data: Batch, emb: dict[str, torch.Tensor]):
        if self.activation_checkpoint:
            forces = torch.utils.checkpoint.checkpoint(
                self.force_block,
                emb["node_embedding"],
                emb["graph"].atomic_numbers_full,
                emb["graph"].edge_distance,
                emb["graph"].edge_index,
                emb["graph"].node_offset,
                use_reentrant=not self.training,
            )
        else:
            forces = self.force_block(
                emb["node_embedding"],
                emb["graph"].atomic_numbers_full,
                emb["graph"].edge_distance,
                emb["graph"].edge_index,
                node_offset=emb["graph"].node_offset,
            )
        forces = forces.embedding.narrow(1, 1, 3)
        #TODO: Check if contiguous is required prev version did not have it
        forces = forces.view(-1, 3).contiguous()
        if gp_utils.initialized(): # TODO: Check why gp_utils was removed
            forces = gp_utils.gather_from_model_parallel_region(forces, dim=0)

        #------------for denoising positions------------------
        denoising_pos_vec = self.denoising_pos_block(
            emb["node_embedding"],
            emb["graph"].atomic_numbers_full,
            emb["graph"].edge_distance,
            emb["graph"].edge_index,
        )
        denoising_pos_vec = denoising_pos_vec.embedding.narrow(1, 1, 3)
        denoising_pos_vec = denoising_pos_vec.view(-1, 3)

        if hasattr(data, "denoising_pos_forward") and data.denoising_pos_forward:
            if hasattr(data, "noise_mask"):
                noise_mask_tensor = data.noise_mask.view(-1, 1)
                forces = denoising_pos_vec * noise_mask_tensor + forces * (
                    ~noise_mask_tensor
                )
            else:
                forces = denoising_pos_vec + 0 * forces
        else:
            forces = 0 * denoising_pos_vec + forces
        #-------------for denoising positions------------------

        return {"forces": forces}
