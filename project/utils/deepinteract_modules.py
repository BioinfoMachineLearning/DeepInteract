import os
from argparse import ArgumentParser
from math import sqrt

import atom3.case as ca
import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm
import wandb
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GraphConv
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from project.utils.deepinteract_constants import FEATURE_INDICES, RESIDUE_COUNT_LIMIT, NODE_COUNT_LIMIT
from project.utils.deepinteract_utils import construct_interact_tensor, glorot_orthogonal, get_geo_feats_from_edges, \
    construct_subsequenced_interact_tensors, insert_interact_tensor_logits, \
    remove_padding, remove_subsequenced_input_padding, calculate_top_k_prec, extract_object
from project.utils.graph_utils import src_dot_dst, scaling, imp_exp_attn, out_edge_features, exp
from project.utils.vision_modules import DeepLabV3Plus


# ------------------
# PyTorch Modules
# ------------------

# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from 'A Generalization of Transformer Networks to Graphs' (https://github.com/graphdeeplearning/graphtransformer):
# -------------------------------------------------------------------------------------------------------------------------------------
class MultiHeadGeometricAttentionLayer(nn.Module):
    """Compute attention scores with a DGLGraph's node and edge (geometric) features."""

    def __init__(self, num_input_feats: int, num_output_feats: int,
                 num_heads: int, using_bias: bool, update_edge_feats: bool):
        super().__init__()

        # Declare shared variables
        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias
        self.update_edge_feats = update_edge_feats

        # Define node features' query, key, and value tensors, and define edge features' projection tensors
        self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.edge_feats_projection = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        if self.using_bias:
            glorot_orthogonal(self.Q.weight, scale=scale)
            self.Q.bias.data.fill_(0)

            glorot_orthogonal(self.K.weight, scale=scale)
            self.K.bias.data.fill_(0)

            glorot_orthogonal(self.V.weight, scale=scale)
            self.V.bias.data.fill_(0)

            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
            self.edge_feats_projection.bias.data.fill_(0)
        else:
            glorot_orthogonal(self.Q.weight, scale=scale)
            glorot_orthogonal(self.K.weight, scale=scale)
            glorot_orthogonal(self.V.weight, scale=scale)
            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)

    def propagate_attention(self, graph: dgl.DGLGraph):
        # Compute attention scores
        graph.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)

        # Scale and clip attention scores
        graph.apply_edges(scaling('score', np.sqrt(self.num_output_feats), 5.0))

        # Use available edge features to modify the attention scores
        graph.apply_edges(imp_exp_attn('score', 'proj_e'))

        # Copy edge features as e_out to be passed to edge_feats_MLP
        if self.update_edge_feats:
            graph.apply_edges(out_edge_features('score'))

        # Apply softmax to attention scores, followed by clipping
        graph.apply_edges(exp('score', 5.0))

        # Send weighted values to target nodes
        e_ids = graph.edges()
        graph.send_and_recv(e_ids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        graph.send_and_recv(e_ids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, graph: dgl.DGLGraph, node_feats: torch.Tensor, edge_feats: torch.Tensor):
        with graph.local_scope():
            e_out = None
            node_feats_q = self.Q(node_feats)
            node_feats_k = self.K(node_feats)
            node_feats_v = self.V(node_feats)
            edge_feats_projection = self.edge_feats_projection(edge_feats)

            # Reshape tensors into [num_nodes, num_heads, feat_dim] to get projections for multi-head attention
            graph.ndata['Q_h'] = node_feats_q.view(-1, self.num_heads, self.num_output_feats)
            graph.ndata['K_h'] = node_feats_k.view(-1, self.num_heads, self.num_output_feats)
            graph.ndata['V_h'] = node_feats_v.view(-1, self.num_heads, self.num_output_feats)
            graph.edata['proj_e'] = edge_feats_projection.view(-1, self.num_heads, self.num_output_feats)

            # Disperse attention information
            self.propagate_attention(graph)

            # Compute final node and edge representations after multi-head attention
            h_out = graph.ndata['wV'] / (graph.ndata['z'] + torch.full_like(graph.ndata['z'], 1e-6))  # Add eps to all
            if self.update_edge_feats:
                e_out = graph.edata['e_out']

        # Return attention-updated node and edge representations
        return h_out, e_out


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------

class InitEdgeModule(nn.Module):
    """An edge initializer module."""

    def __init__(
            self,
            node_count_limit: int,
            num_edge_feats: int,
            num_dist_feats: int,
            num_dir_feats: int,
            num_orient_feats: int,
            num_amide_feats: int,
            num_hidden_channels: int,
            activ_fn=nn.SiLU,
            feature_indices=FEATURE_INDICES
    ):
        super().__init__()

        # Record parameters given
        self.activ_fn = activ_fn
        self.feature_indices = feature_indices

        # --------------------
        # Initializer Module
        # --------------------
        # Establish node embedding strategy
        self.node_embedding = nn.Embedding(node_count_limit, num_hidden_channels)

        # Define linear layers for the edge initializer module
        self.edge_messages_linear_0 = nn.Linear(num_edge_feats, num_hidden_channels, bias=False)
        self.dist_linear_0 = nn.Linear(num_dist_feats, num_hidden_channels, bias=False)
        self.dir_linear_0 = nn.Linear(num_dir_feats, num_hidden_channels, bias=False)
        self.orient_linear_0 = nn.Linear(num_orient_feats, num_hidden_channels, bias=False)
        self.amide_linear_0 = nn.Linear(num_amide_feats, num_hidden_channels, bias=False)

        self.combined_linear_0 = nn.Linear(7 * num_hidden_channels, num_hidden_channels, bias=False)

        self.edge_messages_linear_1 = nn.Linear(num_edge_feats, num_hidden_channels, bias=False)
        self.dist_linear_1 = nn.Linear(num_dist_feats, num_hidden_channels, bias=False)
        self.dir_linear_1 = nn.Linear(num_dir_feats, num_hidden_channels, bias=False)
        self.orient_linear_1 = nn.Linear(num_orient_feats, num_hidden_channels, bias=False)
        self.amide_linear_1 = nn.Linear(num_amide_feats, num_hidden_channels, bias=False)

        combined_out_channels = num_edge_feats + num_dist_feats + num_dir_feats + num_orient_feats + num_amide_feats
        self.combined_linear_1 = nn.Linear(num_hidden_channels, combined_out_channels, bias=False)
        self.combined_linear_2 = nn.Linear(combined_out_channels, num_hidden_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        self.node_embedding.weight.data.uniform_(-sqrt(3), sqrt(3))

        glorot_orthogonal(self.edge_messages_linear_0.weight, scale=scale)
        glorot_orthogonal(self.dist_linear_0.weight, scale=scale)
        glorot_orthogonal(self.dir_linear_0.weight, scale=scale)
        glorot_orthogonal(self.orient_linear_0.weight, scale=scale)
        glorot_orthogonal(self.amide_linear_0.weight, scale=scale)

        glorot_orthogonal(self.combined_linear_0.weight, scale=scale)

        glorot_orthogonal(self.edge_messages_linear_1.weight, scale=scale)
        glorot_orthogonal(self.dist_linear_1.weight, scale=scale)
        glorot_orthogonal(self.dir_linear_1.weight, scale=scale)
        glorot_orthogonal(self.orient_linear_1.weight, scale=scale)
        glorot_orthogonal(self.amide_linear_1.weight, scale=scale)

        glorot_orthogonal(self.combined_linear_1.weight, scale=scale)
        glorot_orthogonal(self.combined_linear_2.weight, scale=scale)

    def init_edge_module_message_func(self, edges: dgl.udf.EdgeBatch):
        """Edge Initialization Module: Compute the messages for an EdgeBatch of edges.
        This function is set up as a User Defined Function in DGL.

        Parameters
        ----------
        edges : EdgeBatch
            A batch of edges for which to compute messages
        """

        # Embed node features given
        node_indices = edges.src['i_all'][0]  # Rely on temp node indices being cached prior to the call to message_func
        node_feats = self.node_embedding(node_indices)
        src_node_feats, dst_node_feats = node_feats[edges.src['i']], node_feats[edges.dst['i']]

        # Update edge features given
        edge_pos_enc_index = self.feature_indices['edge_pos_enc']
        edge_pos_enc = edges.data['f'][:, edge_pos_enc_index]
        edge_weight_index = self.feature_indices['edge_weights']  # Only use edge weights for now
        edge_weights = edges.data['f'][:, edge_weight_index]
        edge_messages_init = torch.cat((edge_pos_enc.reshape(-1, 1), edge_weights.reshape(-1, 1)), dim=1)
        edge_messages_0 = self.edge_messages_linear_0(edge_messages_init)
        dist_feats, dir_feats, orient_feats, amide_feats = get_geo_feats_from_edges(
            edges.data['f'],
            self.feature_indices
        )
        dist_feats_0 = self.activ_fn(self.dist_linear_0(dist_feats))
        dir_feats_0 = self.activ_fn(self.dir_linear_0(dir_feats))
        orient_feats_0 = self.activ_fn(self.orient_linear_0(orient_feats))
        amide_feats_0 = self.activ_fn(self.amide_linear_0(amide_feats.reshape(-1, 1)))
        combined_edge_logits = self.activ_fn(self.combined_linear_0(torch.cat([
            src_node_feats,
            dst_node_feats,
            edge_messages_0,
            dist_feats_0,
            dir_feats_0,
            orient_feats_0,
            amide_feats_0
        ], dim=1)))

        # Gate edge representations
        edge_messages_1_input = torch.cat((edge_pos_enc.reshape(-1, 1), edge_weights.reshape(-1, 1)), dim=1)
        edge_messages_1 = self.edge_messages_linear_1(edge_messages_1_input) * combined_edge_logits
        dist_feats_1 = self.activ_fn(self.dist_linear_1(dist_feats)) * combined_edge_logits
        dir_feats_1 = self.activ_fn(self.dir_linear_1(dir_feats)) * combined_edge_logits
        orient_feats_1 = self.activ_fn(self.orient_linear_1(orient_feats)) * combined_edge_logits
        amide_feats_1 = self.activ_fn(self.amide_linear_1(amide_feats.reshape(-1, 1))) * combined_edge_logits

        # Combine gated edge representations with linear transformations
        combined_edge_feats = edge_messages_1 + dist_feats_1 + dir_feats_1 + orient_feats_1 + amide_feats_1
        edge_feats = self.combined_linear_1(combined_edge_feats)
        edge_feats = self.combined_linear_2(edge_feats)

        return {
            'f': edge_feats
        }

    def forward(self, graph: dgl.DGLGraph):
        """Perform a forward pass of the edge initializer module to get initial edge representations."""
        with graph.local_scope():
            nodes = graph.nodes()
            num_nodes = len(nodes)
            graph.ndata['i'] = nodes  # Record indices of nodes for message passing
            graph.ndata['i_all'] = torch.cat((nodes, nodes.repeat(num_nodes - 1))).reshape(num_nodes, num_nodes)
            graph.apply_edges(self.init_edge_module_message_func)
            edge_feats = graph.edata['f']
        return edge_feats


class ConformationModule(nn.Module):
    """A geometry-evolving (i.e., conforming) module."""

    def __init__(
            self,
            num_dist_feats: int,
            num_dir_feats: int,
            num_orient_feats: int,
            num_amide_feats: int,
            dist_embed_size: int,
            dir_embed_size: int,
            orient_embed_size: int,
            amide_embed_size: int,
            shared_embed_size: int,
            num_hidden_channels: int,
            num_pre_res_blocks: int,
            num_post_res_blocks: int,
            activ_fn=nn.SiLU(),
            norm_to_apply='batch',
            feature_indices=FEATURE_INDICES
    ):
        super().__init__()

        # Record parameters given
        self.activ_fn = activ_fn
        self.feature_indices = feature_indices

        # --------------------
        # Conformation Module
        # --------------------
        # Define geometric modules for a conformation module
        self.dist_linear_0 = nn.Linear(num_dist_feats, dist_embed_size, bias=False)
        self.dist_linear_1 = nn.Linear(dist_embed_size, num_hidden_channels, bias=False)

        self.dir_linear_0 = nn.Linear(num_dir_feats, dir_embed_size, bias=False)
        self.dir_linear_1 = nn.Linear(dir_embed_size, shared_embed_size, bias=False)

        self.orient_linear_0 = nn.Linear(num_orient_feats, orient_embed_size, bias=False)
        self.orient_linear_1 = nn.Linear(orient_embed_size, shared_embed_size, bias=False)

        self.amide_linear_0 = nn.Linear(num_amide_feats, amide_embed_size, bias=False)
        self.amide_linear_1 = nn.Linear(amide_embed_size, shared_embed_size, bias=False)

        # Define utility modules for a conformation module's representation projections
        self.nbr_linear = nn.Linear(num_hidden_channels, num_hidden_channels)
        self.orig_msg_linear = nn.Linear(num_hidden_channels, num_hidden_channels)

        self.downward_proj = nn.Linear(num_hidden_channels, shared_embed_size, bias=False)
        self.upward_proj = nn.Linear(shared_embed_size, num_hidden_channels, bias=False)

        self.res_connect_linear = nn.Linear(num_hidden_channels, num_hidden_channels)

        # Define residual blocks
        self.pre_res_blocks = nn.ModuleList([
            ResBlock(num_hidden_channels, activ_fn, norm_to_apply) for _ in range(num_pre_res_blocks)
        ])
        self.post_res_blocks = nn.ModuleList([
            ResBlock(num_hidden_channels, activ_fn, norm_to_apply) for _ in range(num_post_res_blocks)
        ])

        # Define final linear layers
        self.final_dist_linear = nn.Linear(num_dist_feats, num_hidden_channels, bias=False)
        self.final_dir_linear = nn.Linear(num_dir_feats, num_hidden_channels, bias=False)
        self.final_orient_linear = nn.Linear(num_orient_feats, num_hidden_channels, bias=False)
        self.final_amide_linear = nn.Linear(num_amide_feats, num_hidden_channels, bias=False)
        self.final_linear = nn.Linear(num_hidden_channels, num_hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        # Initialize geometric modules for a conformation module
        scale = 2.0
        glorot_orthogonal(self.dist_linear_0.weight, scale=scale)
        glorot_orthogonal(self.dist_linear_1.weight, scale=scale)

        glorot_orthogonal(self.dir_linear_0.weight, scale=scale)
        glorot_orthogonal(self.dir_linear_1.weight, scale=scale)

        glorot_orthogonal(self.orient_linear_0.weight, scale=scale)
        glorot_orthogonal(self.orient_linear_1.weight, scale=scale)

        glorot_orthogonal(self.amide_linear_0.weight, scale=scale)
        glorot_orthogonal(self.amide_linear_1.weight, scale=scale)

        # Initialize utility modules for a conformation module's representation projections
        glorot_orthogonal(self.nbr_linear.weight, scale=scale)
        self.nbr_linear.bias.data.fill_(0)

        glorot_orthogonal(self.orig_msg_linear.weight, scale=scale)
        self.orig_msg_linear.bias.data.fill_(0)

        glorot_orthogonal(self.downward_proj.weight, scale=scale)
        glorot_orthogonal(self.upward_proj.weight, scale=scale)

        glorot_orthogonal(self.res_connect_linear.weight, scale=scale)
        self.res_connect_linear.bias.data.fill_(0)

        # Initialize final linear layers
        glorot_orthogonal(self.final_dist_linear.weight, scale=scale)
        glorot_orthogonal(self.final_dir_linear.weight, scale=scale)
        glorot_orthogonal(self.final_orient_linear.weight, scale=scale)
        glorot_orthogonal(self.final_amide_linear.weight, scale=scale)
        glorot_orthogonal(self.final_linear.weight, scale=scale)
        self.final_linear.bias.data.fill_(0)

    def conformation_module_message_func(self, edges: dgl.udf.EdgeBatch):
        """Conformation Module: Compute the updated geometric features for an EdgeBatch of edges.
        This function is set up as a User Defined Function in DGL.

        Parameters
        ----------
        edges : EdgeBatch
            A batch of edges for which to compute messages
        """

        # Secure neighboring edge features of interest and cache original edge features for residual reconnections
        src_nbr_e_ids = edges.data['src_nbr_e_ids'].permute(1, 0)
        dst_nbr_e_ids = edges.data['dst_nbr_e_ids'].permute(1, 0)
        src_nbr_edge_feats = edges.data['f'][src_nbr_e_ids]
        dst_nbr_edge_feats = edges.data['f'][dst_nbr_e_ids]
        nbr_edge_feats = torch.cat((src_nbr_edge_feats, dst_nbr_edge_feats))

        nbr_edge_feats = self.activ_fn(self.nbr_linear(nbr_edge_feats))
        orig_edge_feats = edges.data['orig_f']
        res_edge_feats = edges.data['f']  # Cache for future residual connections

        # - Gather initial geometric edge features
        dist_feats, dir_feats, orient_feats, amide_feats = get_geo_feats_from_edges(
            orig_edge_feats,
            self.feature_indices
        )

        # - Create and apply gating with distance embedding
        embedded_dist_feats = self.dist_linear_1(self.dist_linear_0(dist_feats))
        nbr_edge_feats = nbr_edge_feats * embedded_dist_feats

        # - Reduce dimensionality of neighboring edges' representations
        nbr_edge_feats = self.activ_fn(self.downward_proj(nbr_edge_feats))

        # - Curate and apply gating with direction embedding
        nbr_edge_feats = nbr_edge_feats * self.dir_linear_1(self.dir_linear_0(dir_feats))

        # - Make and apply gating with orientation embedding
        nbr_edge_feats = nbr_edge_feats * self.orient_linear_1(self.orient_linear_0(orient_feats))

        # - Make and apply gating with amide plane-amide plane angle embedding
        nbr_edge_feats = nbr_edge_feats * self.amide_linear_1(self.amide_linear_0(amide_feats.reshape(-1, 1)))

        # - Aggregate each edge's neighboring edge representations and increase resulting edge representations' size
        nbr_edge_feats = torch.sum(nbr_edge_feats, dim=0)  # Each edge is (partially) the sum of its neighboring edges
        nbr_edge_feats = self.activ_fn(self.upward_proj(nbr_edge_feats))

        # - Reintroduce the original edge features and combine them with the learned neighboring geometric edge features
        edge_feats = self.orig_msg_linear(res_edge_feats)
        edge_feats = edge_feats + nbr_edge_feats

        # Use pre-residual connection residual modules as specified
        for layer in self.pre_res_blocks:
            edge_feats = layer(edge_feats)

        # Make a residual reconnection with the original edge features given
        edge_feats = res_edge_feats + self.activ_fn(self.res_connect_linear(edge_feats))

        # Use post-residual connection residual modules as specified
        for layer in self.post_res_blocks:
            edge_feats = layer(edge_feats)

        # Gate with each type of original geometric edge feature
        gated_edge_feats_0 = self.final_dist_linear(dist_feats) * edge_feats
        gated_edge_feats_1 = self.final_dir_linear(dir_feats) * edge_feats
        gated_edge_feats_2 = self.final_orient_linear(orient_feats) * edge_feats
        gated_edge_feats_3 = self.final_amide_linear(amide_feats.reshape(-1, 1)) * edge_feats

        # Combine gated edge feature representations to get final edge representations after residual reconnecting
        combined_gated_edge_feats = gated_edge_feats_0 + gated_edge_feats_1 + gated_edge_feats_2 + gated_edge_feats_3
        edge_feats = res_edge_feats + self.activ_fn(self.final_linear(combined_gated_edge_feats))

        return {
            'f': edge_feats
        }

    def forward(self, graph: dgl.DGLGraph, orig_edge_feats: torch.Tensor):
        """Perform a forward pass of a conformation module to get intermediate edge representations."""
        with graph.local_scope():
            graph.edata['orig_f'] = orig_edge_feats
            graph.apply_edges(self.conformation_module_message_func)
            edge_features = graph.edata['f']
        return edge_features


class ResBlock(nn.Module):
    """A residual block for a conformation module."""

    def __init__(self, hidden_channels, activ_fn=nn.SiLU(), norm_to_apply='batch'):
        super().__init__()

        # Record parameters given
        self.activ_fn = activ_fn

        # Define projection layers for a conformation module residual block
        norm_layer = nn.LayerNorm(hidden_channels) if norm_to_apply == 'layer' else nn.BatchNorm1d(hidden_channels)
        self.res_block = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels),
            norm_layer,
            activ_fn,
            nn.Linear(hidden_channels, hidden_channels),
            norm_layer,
            activ_fn,
            nn.Linear(hidden_channels, hidden_channels),
            norm_layer,
            activ_fn
        ])

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        for layer in self.res_block:
            not_norm_layer = not (hasattr(layer, 'normalized_shape') or hasattr(layer, 'running_mean'))
            if hasattr(layer, 'weight') and not_norm_layer:  # Skip init for activation functions and normalizing layers
                glorot_orthogonal(layer.weight, scale=scale)
                layer.bias.data.fill_(0)

    def forward(self, x):
        """Perform a forward pass using residual layers with intermediate activation functions applied."""
        x_res = x
        for layer in self.res_block:
            x_res = layer(x_res)
        return x + x_res


class GeometricTransformerModule(nn.Module):
    """A Geometric Transformer module (equivalent to one layer of graph convolutions)."""

    def __init__(
            self,
            shared_embed_size: int,
            num_dist_feats: int,
            num_dir_feats: int,
            num_orient_feats: int,
            num_amide_feats: int,
            dist_embed_size: int,
            dir_embed_size: int,
            orient_embed_size: int,
            amide_embed_size: int,
            num_hidden_channels: int,
            num_pre_res_blocks: int,
            num_post_res_blocks: int,
            activ_fn=nn.SiLU(),
            residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            knn=20,
            num_layers=4,
            feature_indices=FEATURE_INDICES,
            disable_geometric_mode=False
    ):
        super().__init__()

        """Geometry-Focused Graph Transformer Layer

        Parameters
        ----------
        shared_embed_size : int
            Size of shared embedding in a conformation module.
        dist_embed_size : int
            Size of distance embedding in a conformation module.
        dir_embed_size : int
            Size of direction embedding in a conformation module.
        orient_embed_size : int
            Size of orientation embedding in a conformation module.
        amide_embed_size : int
            Size of embedding in a conformation module for amide plane-amide plane normal vector angles.
        num_hidden_channels : int
            Hidden channel size for both nodes and edges.
        num_pre_res_blocks : int
            Number of residual blocks to apply prior to a residual reconnection.
        num_post_res_blocks : int
            Number of residual blocks to apply following a residual reconnection.
        activ_fn : Module
            Activation function to apply in MLPs.
        residual : bool
            Whether to use a residual update strategy for node features.
        num_attention_heads : int
            How many attention heads to apply to the input node features in parallel.
        norm_to_apply : str
            Which normalization scheme to apply to node and edge representations (i.e. 'batch' or 'layer').
        dropout_rate : float
            How much dropout (i.e. forget rate) to apply before activation functions.
        knn : int
            How many nearest neighbors were used when constructing node neighborhoods.
        num_layers : int
            How many layers of geometric attention to apply.
        feature_indices : dict
            A dictionary listing the start and end indices for each node and edge feature.
        disable_geometric_mode : bool
            Whether to convert the Geometric Transformer into the original Graph Transformer.
        """

        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.knn = knn
        self.num_layers = num_layers
        self.feature_indices = feature_indices
        self.disable_geometric_mode = disable_geometric_mode

        # --------------------
        # Conformation Module
        # --------------------
        # Define all modules related to a conformation module
        if not self.disable_geometric_mode:
            self.conformation_module = ConformationModule(num_dist_feats=num_dist_feats,
                                                          num_dir_feats=num_dir_feats,
                                                          num_orient_feats=num_orient_feats,
                                                          num_amide_feats=num_amide_feats,
                                                          dist_embed_size=dist_embed_size,
                                                          dir_embed_size=dir_embed_size,
                                                          orient_embed_size=orient_embed_size,
                                                          amide_embed_size=amide_embed_size,
                                                          shared_embed_size=shared_embed_size,
                                                          num_hidden_channels=num_hidden_channels,
                                                          num_pre_res_blocks=num_pre_res_blocks,
                                                          num_post_res_blocks=num_post_res_blocks,
                                                          activ_fn=activ_fn,
                                                          norm_to_apply=norm_to_apply,
                                                          feature_indices=feature_indices)

        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a Geometric Transformer module
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadGeometricAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
            update_edge_feats=True
        )

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        # MLP for node features
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        # MLP for edge features
        self.edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)

        glorot_orthogonal(self.O_edge_feats.weight, scale=scale)
        self.O_edge_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):  # Skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)

        for layer in self.edge_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

    def run_gt_layer(self, graph: dgl.DGLGraph, node_feats: torch.Tensor,
                     edge_feats: torch.Tensor, orig_edge_feats: torch.Tensor):
        """Perform a forward pass of geometric attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats  # Cache node representations for first residual connection
        edge_feats_in1 = edge_feats  # Cache edge representations for first residual connection

        # Update the input graph's geometric edge features iteratively
        if not self.disable_geometric_mode:
            edge_feats = self.conformation_module(graph, orig_edge_feats)

        # Apply first round of normalization before applying geometric attention, for performance enhancement
        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        # Get multi-head attention output using provided node and edge representations
        node_attn_out, edge_attn_out = self.mha_module(graph, node_feats, edge_feats)

        node_feats = node_attn_out.view(-1, self.num_output_feats)
        edge_feats = edge_attn_out.view(-1, self.num_output_feats)

        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        edge_feats = F.dropout(edge_feats, self.dropout_rate, training=self.training)

        node_feats = self.O_node_feats(node_feats)
        edge_feats = self.O_edge_feats(edge_feats)

        # Make first residual connection
        if self.residual:
            node_feats = node_feats_in1 + node_feats  # Make first node residual connection
            edge_feats = edge_feats_in1 + edge_feats  # Make first edge residual connection

        node_feats_in2 = node_feats  # Cache node representations for second residual connection
        edge_feats_in2 = edge_feats  # Cache edge representations for second residual connection

        # Apply second round of normalization after first residual connection has been made
        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
            edge_feats = self.layer_norm2_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm2_node_feats(node_feats)
            edge_feats = self.batch_norm2_edge_feats(edge_feats)

        # Apply MLPs for node and edge features
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)

        # Make second residual connection
        if self.residual:
            node_feats = node_feats_in2 + node_feats  # Make second node residual connection
            edge_feats = edge_feats_in2 + edge_feats  # Make second edge residual connection

        # Return edge representations along with node representations (for tasks other than interface prediction)
        return node_feats, edge_feats

    def forward(self, graph: dgl.DGLGraph, orig_edge_feats: torch.Tensor):
        """Perform a forward pass of a Geometric Transformer to get intermediate node and edge representations."""
        node_feats, edge_feats = self.run_gt_layer(graph, graph.ndata['f'], graph.edata['f'], orig_edge_feats)
        return node_feats, edge_feats


class FinalGeometricTransformerModule(nn.Module):
    """A (final layer) Geometric Transformer module that combines node and edge representations using self-attention."""

    def __init__(
            self,
            shared_embed_size: int,
            num_dist_feats: int,
            num_dir_feats: int,
            num_orient_feats: int,
            num_amide_feats: int,
            dist_embed_size: int,
            dir_embed_size: int,
            orient_embed_size: int,
            amide_embed_size: int,
            num_hidden_channels: int,
            num_pre_res_blocks: int,
            num_post_res_blocks: int,
            activ_fn=nn.SiLU(),
            residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            knn=20,
            num_layers=4,
            feature_indices=FEATURE_INDICES,
            disable_geometric_mode=False
    ):
        super().__init__()

        """Geometry-Focused Graph Transformer (Final) Layer

        Parameters
        ----------
        shared_embed_size : int
            Size of shared embedding in a conformation module.
        dist_embed_size : int
            Size of distance embedding in a conformation module.
        dir_embed_size : int
            Size of direction embedding in a conformation module.
        orient_embed_size : int
            Size of orientation embedding in a conformation module.
        amide_embed_size : int
            Size of embedding in a conformation module for amide plane-amide plane normal vector angles.
        num_hidden_channels : int
            Hidden channel size for both nodes and edges.
        num_pre_res_blocks : int
            Number of residual blocks to apply prior to a residual reconnection.
        num_post_res_blocks : int
            Number of residual blocks to apply following a residual reconnection.
        activ_fn : Module
            Activation function to apply in MLPs.
        residual : bool
            Whether to use a residual update strategy for node features.
        num_attention_heads : int
            How many attention heads to apply to the input node features in parallel.
        norm_to_apply : str
            Which normalization scheme to apply to node and edge representations (i.e. 'batch' or 'layer').
        dropout_rate : float
            How much dropout (i.e. forget rate) to apply before activation functions.
        knn : int
            How many nearest neighbors were used when constructing node neighborhoods.
        num_layers : int
            How many layers of geometric attention to apply.
        feature_indices : dict
            A dictionary listing the start and end indices for each node and edge feature.
        disable_geometric_mode : bool
            Whether to convert the Geometric Transformer into the original Graph Transformer.
        """

        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.knn = knn
        self.num_layers = num_layers
        self.feature_indices = feature_indices
        self.disable_geometric_mode = disable_geometric_mode

        # --------------------
        # Conformation Module
        # --------------------
        # Define all modules related to a conformation module
        if self.disable_geometric_mode:
            total_edge_feats_size = 4 + num_dist_feats + num_dir_feats + num_orient_feats + num_amide_feats
            self.conformation_module = nn.Linear(total_edge_feats_size, num_hidden_channels, bias=False)
        else:
            self.conformation_module = ConformationModule(num_dist_feats=num_dist_feats,
                                                          num_dir_feats=num_dir_feats,
                                                          num_orient_feats=num_orient_feats,
                                                          num_amide_feats=num_amide_feats,
                                                          dist_embed_size=dist_embed_size,
                                                          dir_embed_size=dir_embed_size,
                                                          orient_embed_size=orient_embed_size,
                                                          amide_embed_size=amide_embed_size,
                                                          shared_embed_size=shared_embed_size,
                                                          num_hidden_channels=num_hidden_channels,
                                                          num_pre_res_blocks=num_pre_res_blocks,
                                                          num_post_res_blocks=num_post_res_blocks,
                                                          activ_fn=activ_fn,
                                                          norm_to_apply=norm_to_apply,
                                                          feature_indices=feature_indices)

        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a Geometric Transformer module
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadGeometricAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
            update_edge_feats=False
        )

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        # MLP for node features
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):  # Skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)

        if self.disable_geometric_mode:
            glorot_orthogonal(self.conformation_module.weight, scale=scale)

    def run_gt_layer(self, graph: dgl.DGLGraph, node_feats: torch.Tensor, orig_edge_feats: torch.Tensor):
        """Perform a forward pass of geometric attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats  # Cache node representations for first residual connection

        # Update the input graph's geometric edge features iteratively
        if self.disable_geometric_mode:
            edge_pos_enc_index = self.feature_indices['edge_pos_enc']
            edge_pos_enc = graph.edata['f'][:, edge_pos_enc_index]
            edge_weight_index = self.feature_indices['edge_weights']
            edge_weights = graph.edata['f'][:, edge_weight_index]
            edge_messages_init = torch.cat((edge_pos_enc.reshape(-1, 1), edge_weights.reshape(-1, 1)), dim=1)
            edge_feats_init = torch.cat((edge_messages_init, orig_edge_feats), dim=1)
            edge_feats = self.conformation_module(edge_feats_init)
        else:
            edge_feats = self.conformation_module(graph, orig_edge_feats)

        # Apply first round of normalization before applying geometric attention, for performance enhancement
        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        # Get multi-head attention output using provided node and edge representations
        node_attn_out, _ = self.mha_module(graph, node_feats, edge_feats)

        node_feats = node_attn_out.view(-1, self.num_output_feats)

        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)

        node_feats = self.O_node_feats(node_feats)

        # Make first residual connection
        if self.residual:
            node_feats = node_feats_in1 + node_feats  # Make first node residual connection

        node_feats_in2 = node_feats  # Cache node representations for second residual connection

        # Apply second round of normalization after first residual connection has been made
        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm2_node_feats(node_feats)

        # Apply MLP for node features
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)

        # Make second residual connection
        if self.residual:
            node_feats = node_feats_in2 + node_feats  # Make second node residual connection

        # Return node representations
        return node_feats

    def forward(self, graph: dgl.DGLGraph, orig_edge_feats: torch.Tensor):
        """Perform a forward pass of a Geometric Transformer to get final node representations."""
        node_feats = self.run_gt_layer(graph, graph.ndata['f'], orig_edge_feats)
        return node_feats


class SEBlock(torch.nn.Module):
    """A squeeze-and-excitation block for PyTorch."""

    def __init__(self, ch, ratio=16):
        super(SEBlock, self).__init__()
        self.ratio = ratio
        self.linear1 = torch.nn.Linear(ch, ch // ratio)
        self.linear2 = torch.nn.Linear(ch // ratio, ch)
        self.act = nn.ReLU()

    def forward(self, in_block):
        x = torch.reshape(in_block, (in_block.shape[0], in_block.shape[1], -1))
        x = torch.mean(x, dim=-1)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = torch.sigmoid(x)
        return torch.einsum('bcij,bc->bcij', in_block, x)


class ResNet(nn.Module):
    """A custom ResNet module for PyTorch."""

    # Parameter initialization
    def __init__(self,
                 num_channels,
                 num_chunks,
                 module_name,
                 activ_fn=F.elu,
                 inorm=False,
                 initial_projection=False,
                 extra_blocks=False,
                 dilation_cycle=None,
                 verbose=False):

        self.num_channel = num_channels
        self.num_chunks = num_chunks
        self.module_name = module_name
        self.activ_fn = activ_fn
        self.inorm = inorm
        self.initial_projection = initial_projection
        self.extra_blocks = extra_blocks
        self.dilation_cycle = [1, 2, 4, 8] if dilation_cycle is None else dilation_cycle
        self.verbose = verbose

        super(ResNet, self).__init__()

        if self.initial_projection:
            self.add_module(f'resnet_{self.module_name}_init_proj', nn.Conv2d(in_channels=num_channels,
                                                                              out_channels=num_channels,
                                                                              kernel_size=(1, 1)))

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                if self.inorm:
                    self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_1',
                                    nn.InstanceNorm2d(num_channels, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_2',
                                    nn.InstanceNorm2d(num_channels // 2, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_3',
                                    nn.InstanceNorm2d(num_channels // 2, eps=1e-06, affine=True))

                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_1',
                                nn.Conv2d(num_channels, num_channels // 2, kernel_size=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_2',
                                nn.Conv2d(num_channels // 2,
                                          num_channels // 2,
                                          kernel_size=(3, 3),
                                          dilation=dilation_rate,
                                          padding=dilation_rate))
                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_3',
                                nn.Conv2d(num_channels // 2, num_channels, kernel_size=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_se_block',
                                SEBlock(num_channels, ratio=16))

        if self.extra_blocks:
            for i in range(2):
                if self.inorm:
                    self.add_module(f'resnet_{self.module_name}_extra{i}_inorm_1',
                                    nn.InstanceNorm2d(num_channels, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_extra{i}_inorm_2',
                                    nn.InstanceNorm2d(num_channels // 2, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_extra{i}_inorm_3',
                                    nn.InstanceNorm2d(num_channels // 2, eps=1e-06, affine=True))

                self.add_module(f'resnet_{self.module_name}_extra{i}_conv2d_1',
                                nn.Conv2d(num_channels, num_channels // 2, kernel_size=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_extra{i}_conv2d_2',
                                nn.Conv2d(num_channels // 2,
                                          num_channels // 2,
                                          kernel_size=(3, 3),
                                          dilation=(1, 1),
                                          padding=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_extra{i}_conv2d_3',
                                nn.Conv2d(num_channels // 2, num_channels, kernel_size=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_extra{i}_se_block',
                                SEBlock(num_channels, ratio=16))

    def forward(self, x):
        """Compute ResNet output."""
        activ_fn = self.activ_fn

        if self.initial_projection:
            x = self._modules[f'resnet_{self.module_name}_init_proj'](x)

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                _residual = x

                # Internal block
                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_1'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_1'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_2'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_2'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_3'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_3'](x)

                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_se_block'](x)

                x = x + _residual

        if self.extra_blocks:
            for i in range(2):
                _residual = x

                # Internal block
                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_extra{i}_inorm_1'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_extra{i}_conv2d_1'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_extra{i}_inorm_2'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_extra{i}_conv2d_2'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_extra{i}_inorm_3'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_extra{i}_conv2d_3'](x)

                x = self._modules[f'resnet_{self.module_name}_extra{i}_se_block'](x)

                x = x + _residual

        return x


class MultiHeadRegionalAttention(nn.Module):
    """A multi-head attention block for PyTorch that operates regionally."""

    @staticmethod
    def get_stretch_weight(s):
        w = np.zeros((s * s, 1, 1, s, s))
        for i in range(s):
            for j in range(s):
                w[s * i + j, 0, 0, i, j] = 1
        return np.asarray(w).astype(np.float32)

    def __init__(self, in_dim=3, region_size=3, d_k=16, d_v=32, n_head=4, att_drop=0.1, output_score=False):
        super(MultiHeadRegionalAttention, self).__init__()
        self.temper = int(np.sqrt(d_k))
        self.dk_per_head = d_k // n_head
        self.dv_per_head = d_v // n_head
        self.dropout_layer = nn.Dropout(att_drop)
        self.output_score = output_score
        self.q_layer = nn.Conv2d(in_dim, d_k, kernel_size=(1, 1), bias=False)
        self.k_layer = nn.Conv2d(in_dim, d_k, kernel_size=(1, 1), bias=False)
        self.v_layer = nn.Conv2d(in_dim, d_v, kernel_size=(1, 1), bias=False)
        self.softmax_layer = nn.Softmax(1)
        self.stretch_layer = nn.Conv3d(in_channels=1,
                                       out_channels=region_size * region_size,
                                       kernel_size=(1, region_size, region_size),
                                       bias=False,
                                       padding=(0, 1, 1))
        self.stretch_layer.weight = nn.Parameter(
            torch.tensor(self.get_stretch_weight(region_size)), requires_grad=False
        )

    def forward(self, x):
        """Compute attention output and attention score."""
        Q = self.stretch_layer(self.q_layer(x).unsqueeze(1))
        K = self.stretch_layer(self.k_layer(x).unsqueeze(1))
        V = self.stretch_layer(self.v_layer(x).unsqueeze(1))
        qk = torch.mul(Q, K).permute(0, 2, 1, 3, 4)
        qk1 = qk.view((-1, self.dk_per_head, qk.shape[2], qk.shape[3], qk.shape[4]))
        attention_score = self.softmax_layer(torch.div(torch.sum(qk1, 1), self.temper))
        attention_score2 = self.dropout_layer(attention_score)
        attention_score2 = torch.repeat_interleave(attention_score2.unsqueeze(0).permute(0, 2, 1, 3, 4),
                                                   repeats=self.dv_per_head, dim=2)
        attention_out = torch.sum(torch.mul(attention_score2, V), dim=1)
        return attention_out, attention_score if self.output_score else attention_out


class ResNet2DInputWithOptAttention(nn.Module):
    """A ResNet and (optionally) regionally-attentive convolution module for a pair of 2D feature tensors."""

    def __init__(self,
                 num_chunks=4,
                 init_channels=128,
                 num_channels=128,
                 num_classes=2,
                 use_attention=False,
                 n_head=4,
                 module_name=None,
                 activ_fn=F.elu,
                 dropout=0.1,
                 verbose=False):
        super(ResNet2DInputWithOptAttention, self).__init__()
        self.num_chunks = num_chunks
        self.init_channels = init_channels
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.n_head = n_head
        self.module_name = module_name
        self.activ_fun = activ_fn
        self.dropout = dropout
        self.verbose = verbose

        self.add_module('conv2d_1', nn.Conv2d(in_channels=self.init_channels,
                                              out_channels=self.num_channels,
                                              kernel_size=(1, 1),
                                              padding=(0, 0)))
        self.add_module('inorm_1', nn.InstanceNorm2d(self.num_channels, eps=1e-06, affine=True))

        self.add_module('base_resnet', ResNet(num_channels,
                                              self.num_chunks,
                                              module_name='base_resnet',
                                              activ_fn=self.activ_fun,
                                              inorm=True,
                                              initial_projection=True,
                                              extra_blocks=False))

        self.add_module('phase2_resnet', ResNet(num_channels,
                                                num_chunks=1,
                                                module_name='bin_resnet',
                                                activ_fn=self.activ_fun,
                                                inorm=False,
                                                initial_projection=True,
                                                extra_blocks=True))
        self.add_module('phase2_conv', nn.Conv2d(in_channels=self.num_channels,
                                                 out_channels=self.num_classes,
                                                 kernel_size=(1, 1),
                                                 padding=(0, 0)))
        if self.use_attention:
            self.add_module('MHA2D_1', MultiHeadRegionalAttention(self.num_channels,
                                                                  d_v=self.num_channels,
                                                                  n_head=self.n_head,
                                                                  att_drop=self.dropout,
                                                                  output_score=True))
            self.add_module('MHA2D_2', MultiHeadRegionalAttention(self.num_channels,
                                                                  d_v=self.num_channels,
                                                                  n_head=self.n_head,
                                                                  att_drop=self.dropout,
                                                                  output_score=True))

        # Reset learnable parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        # Reinitialize final output layer
        final_layer_bias = self._modules['phase2_conv'].bias.clone()
        final_layer_bias[1] = -7.0  # -7 chosen as the second term's bias s.t. positives are predicted with prob=0.001
        self._modules['phase2_conv'].bias = nn.Parameter(final_layer_bias, requires_grad=True)

    def forward(self, f2d_tile: torch.Tensor):
        """Compute final convolution output."""
        activ_fun = self.activ_fun
        out_conv2d_1 = self._modules['conv2d_1'](f2d_tile)
        out_inorm_1 = activ_fun(self._modules['inorm_1'](out_conv2d_1))

        # First ResNet
        out_base_resnet = activ_fun(self._modules['base_resnet'](out_inorm_1))
        if self.use_attention:
            out_base_resnet, attention_scores_1 = self._modules['MHA2D_1'](out_base_resnet)
            out_base_resnet = activ_fun(out_base_resnet)

        # Second ResNet
        out_bin_predictor = activ_fun(self._modules['phase2_resnet'](out_base_resnet))
        if self.use_attention:
            out_bin_predictor, attention_scores_2 = self._modules['MHA2D_2'](out_bin_predictor)
            out_bin_predictor = activ_fun(out_bin_predictor)

        # Output convolution
        out_layer = self._modules['phase2_conv'](out_bin_predictor)
        return out_layer


# ------------------
# DGL Modules
# ------------------

class DGLGeometricTransformer(nn.Module):
    """A geometry-focused graph transformer, as a DGL module.

    DGLGeometricTransformer stands for a geometry-focused Graph Transformer layer. It is the
    equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph conv layer in a GCN.
    """

    def __init__(
            self,
            node_count_limit=NODE_COUNT_LIMIT,
            shared_embed_size=64,
            dist_embed_size=8,
            dir_embed_size=8,
            orient_embed_size=8,
            amide_embed_size=8,
            num_hidden_channels=128,
            num_pre_res_blocks=2,
            num_post_res_blocks=2,
            activ_fn=nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            knn=20,
            num_layers=4,
            feature_indices=FEATURE_INDICES,
            disable_geometric_mode=False,
            **kwargs
    ):
        """Geometry-Focused Graph Transformer Layer

        Parameters
        ----------
        node_count_limit : int
            Maximum number of nodes allowed in a given graph.
        shared_embed_size : int
            Size of shared embedding in a conformation module.
        dist_embed_size : int
            Size of distance embedding in a conformation module.
        dir_embed_size : int
            Size of direction embedding in a conformation module.
        orient_embed_size : int
            Size of orientation embedding in a conformation module.
        amide_embed_size : int
            Size of embedding in a conformation module for amide plane-amide plane normal vector angles.
        num_hidden_channels : int
            Hidden channel size for both nodes and edges.
        num_pre_res_blocks : int
            Number of residual blocks to apply prior to a residual reconnection.
        num_post_res_blocks : int
            Number of residual blocks to apply following a residual reconnection.
        activ_fn : Module
            Activation function to apply in MLPs.
        transformer_residual : bool
            Whether to use a transformer-residual update strategy for node features.
        num_attention_heads : int
            How many attention heads to apply to the input node features in parallel.
        norm_to_apply : str
            Which normalization scheme to apply to node and edge representations (i.e. 'batch' or 'layer').
        dropout_rate : float
            How much dropout (i.e. forget rate) to apply before activation functions.
        knn : int
            How many nearest neighbors were used when constructing node neighborhoods.
        num_layers : int
            How many layers of geometric attention to apply.
        feature_indices : dict
            A dictionary listing the start and end indices for each node and edge feature.
        disable_geometric_mode : bool
            Whether to convert the Geometric Transformer into the original Graph Transformer.
        """
        super().__init__()
        assert feature_indices is not None, 'Dictionary of indices for node and edge features to be used must be given'

        # Initialize model parameters
        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.knn = knn
        self.num_layers = num_layers
        self.feature_indices = feature_indices
        self.disable_geometric_mode = disable_geometric_mode

        # Parse dimensionality of each type of node and edge feature
        num_dist_feats = feature_indices['edge_dist_feats_end'] - feature_indices['edge_dist_feats_start']
        num_dir_feats = feature_indices['edge_dir_feats_end'] - feature_indices['edge_dir_feats_start']
        num_orient_feats = feature_indices['edge_orient_feats_end'] - feature_indices['edge_orient_feats_start']
        num_amide_feats = 1

        # --------------------
        # Initializer Modules
        # --------------------
        # Define all modules related to edge and node initialization
        if self.disable_geometric_mode:
            total_edge_feats_size = 4 + num_dist_feats + num_dir_feats + num_orient_feats + num_amide_feats
            self.init_edge_module = nn.Linear(total_edge_feats_size, num_hidden_channels, bias=False)
        else:
            self.init_edge_module = InitEdgeModule(node_count_limit=node_count_limit,
                                                   num_edge_feats=2,  # Only use pos. encod. and edge weights initially
                                                   num_dist_feats=num_dist_feats,
                                                   num_dir_feats=num_dir_feats,
                                                   num_orient_feats=num_orient_feats,
                                                   num_amide_feats=num_amide_feats,
                                                   num_hidden_channels=num_hidden_channels,
                                                   activ_fn=activ_fn,
                                                   feature_indices=feature_indices)

        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a variable number of Geometric Transformer modules
        num_intermediate_layers = max(0, num_layers - 1)
        gt_block_modules = [
            GeometricTransformerModule(shared_embed_size=shared_embed_size,
                                       num_dist_feats=num_dist_feats,
                                       num_dir_feats=num_dir_feats,
                                       num_orient_feats=num_orient_feats,
                                       num_amide_feats=num_amide_feats,
                                       dist_embed_size=dist_embed_size,
                                       dir_embed_size=dir_embed_size,
                                       orient_embed_size=orient_embed_size,
                                       amide_embed_size=amide_embed_size,
                                       num_hidden_channels=num_hidden_channels,
                                       num_pre_res_blocks=num_pre_res_blocks,
                                       num_post_res_blocks=num_post_res_blocks,
                                       activ_fn=activ_fn,
                                       residual=transformer_residual,
                                       num_attention_heads=num_attention_heads,
                                       norm_to_apply=norm_to_apply,
                                       dropout_rate=dropout_rate,
                                       knn=knn,
                                       num_layers=num_layers,
                                       feature_indices=feature_indices,
                                       disable_geometric_mode=self.disable_geometric_mode)
            for _ in range(num_intermediate_layers)
        ]
        if num_layers > 0:
            gt_block_modules.extend([
                FinalGeometricTransformerModule(shared_embed_size=shared_embed_size,
                                                num_dist_feats=num_dist_feats,
                                                num_dir_feats=num_dir_feats,
                                                num_orient_feats=num_orient_feats,
                                                num_amide_feats=num_amide_feats,
                                                dist_embed_size=dist_embed_size,
                                                dir_embed_size=dir_embed_size,
                                                orient_embed_size=orient_embed_size,
                                                amide_embed_size=amide_embed_size,
                                                num_hidden_channels=num_hidden_channels,
                                                num_pre_res_blocks=num_pre_res_blocks,
                                                num_post_res_blocks=num_post_res_blocks,
                                                activ_fn=activ_fn,
                                                residual=transformer_residual,
                                                num_attention_heads=num_attention_heads,
                                                norm_to_apply=norm_to_apply,
                                                dropout_rate=dropout_rate,
                                                knn=knn,
                                                num_layers=num_layers,
                                                feature_indices=feature_indices,
                                                disable_geometric_mode=self.disable_geometric_mode)
            ])
        self.gt_block = nn.ModuleList(gt_block_modules)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        if self.disable_geometric_mode:
            glorot_orthogonal(self.init_edge_module.weight, scale=scale)

    def forward(
            self,
            graph: dgl.DGLGraph
    ):
        """Forward pass of the network

        Parameters
        ----------
        graph : DGLGraph
            DGL input graph
        """
        # Cache the original batch number of nodes and edges
        batch_num_nodes, batch_num_edges = graph.batch_num_nodes(), graph.batch_num_edges()

        # Cache original edge features
        orig_edge_feats = graph.edata['f']

        # Prepare initial edge representations
        if self.disable_geometric_mode:
            edge_pos_enc_index = self.feature_indices['edge_pos_enc']
            edge_pos_enc = graph.edata['f'][:, edge_pos_enc_index]
            edge_weight_index = self.feature_indices['edge_weights']
            edge_weights = graph.edata['f'][:, edge_weight_index]
            edge_messages_init = torch.cat((edge_pos_enc.reshape(-1, 1), edge_weights.reshape(-1, 1)), dim=1)
            edge_feats_init = torch.cat((edge_messages_init, graph.edata['f']), dim=1)
            graph.edata['f'] = self.init_edge_module(edge_feats_init)
        else:
            graph.edata['f'] = self.init_edge_module(graph)

        # Apply a given number of intermediate geometric attention layers to the node and edge features given
        for gt_layer in self.gt_block[:-1]:
            graph.ndata['f'], graph.edata['f'] = gt_layer(graph, orig_edge_feats)

        # Apply final layer to update node representations by merging current node and edge representations
        graph.ndata['f'] = self.gt_block[-1](graph, orig_edge_feats)

        # Retain the original batch number of nodes and edges
        graph.set_batch_num_nodes(batch_num_nodes), graph.set_batch_num_edges(batch_num_edges)

        # Return updated node and edge features inside provided DGLGraph
        return graph

    def __repr__(self):
        return f'DGLGeometricTransformer(structure=' \
               f'h_in{self.num_node_input_feats}-h_hid{self.node_output_embedding_hidden_feats}-h_out{self.num_node_input_feats}' \
               f'-e_in{self.num_edge_input_feats}-e_hid{self.edge_input_mlp_output_size}-e_out{self.num_edge_input_feats})'


# ------------------
# Lightning Modules
# ------------------

class LitGINI(pl.LightningModule):
    """A geometry-focused inter-graph node interaction (GINI) module."""

    def __init__(self, num_node_input_feats: int, num_edge_input_feats: int, gnn_activ_fn=nn.SiLU(),
                 num_classes=2, max_num_graph_nodes=NODE_COUNT_LIMIT, max_num_residues=RESIDUE_COUNT_LIMIT,
                 testing_with_casp_capri=False, pos_prob_threshold=0.5, gnn_layer_type='geotran', num_gnn_layers=2,
                 num_gnn_hidden_channels=128, num_gnn_attention_heads=4, knn=20, interact_module_type='dil_resnet',
                 num_interact_layers=14, num_interact_hidden_channels=128, use_interact_attention=False,
                 num_interact_attention_heads=4, disable_geometric_mode=False, num_epochs=50, pn_ratio=0.1,
                 dropout_rate=0.2, metric_to_track='val_ce', weight_decay=1e-2, batch_size=1, lr=1e-3, pad=False,
                 viz_every_n_epochs=1, use_wandb_logger=True, weight_classes=False):
        """Initialize all the parameters for a LitGINI module."""
        super().__init__()

        # Build the network
        self.num_node_input_feats = num_node_input_feats
        self.num_edge_input_feats = num_edge_input_feats
        self.gnn_activ_fn = gnn_activ_fn
        self.num_classes = num_classes
        self.max_num_graph_nodes = max_num_graph_nodes
        self.max_num_residues = max_num_residues
        self.testing_with_casp_capri = testing_with_casp_capri
        self.pos_prob_threshold = pos_prob_threshold

        # GNN module's keyword arguments provided via the command line
        self.gnn_layer_type = gnn_layer_type
        self.num_gnn_layers = num_gnn_layers
        self.num_gnn_hidden_channels = num_gnn_hidden_channels
        self.num_gnn_attention_heads = num_gnn_attention_heads
        self.nbrhd_size = knn

        # Interaction module's keyword arguments provided via the command line
        self.interact_module_type = interact_module_type
        self.num_interact_layers = num_interact_layers
        self.num_interact_hidden_channels = num_interact_hidden_channels
        self.use_interact_attention = use_interact_attention
        self.num_interact_attention_heads = num_interact_attention_heads
        self.disable_geometric_mode = disable_geometric_mode

        # Derive shortcut booleans for convenient future reference
        self.using_dil_resnet = self.interact_module_type.lower() == 'dil_resnet'
        self.using_deeplab = self.interact_module_type.lower() == 'deeplab'

        # Model hyperparameter keyword arguments provided via the command line
        self.num_epochs = num_epochs
        self.pn_ratio = pn_ratio
        self.dropout_rate = dropout_rate
        self.metric_to_track = metric_to_track
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr = lr
        self.pad = pad
        self.viz_every_n_epochs = viz_every_n_epochs  # Visualize model predictions every 'n' epochs
        self.use_wandb_logger = use_wandb_logger  # Whether to use WandB as the primary means of logging
        self.weight_classes = weight_classes  # Whether to use class weighting in our training Cross Entropy

        # Set up GNN node and edge embedding layers (if requested)
        self.using_gcn = self.gnn_layer_type.lower() == 'gcn'
        self.using_gat = self.gnn_layer_type.lower() == 'gat'
        self.using_node_embedding = self.num_node_input_feats != self.num_gnn_hidden_channels
        self.node_in_embedding = nn.Linear(self.num_node_input_feats, self.num_gnn_hidden_channels, bias=False) \
            if self.using_node_embedding \
            else nn.Identity()

        # Assemble the layers of the network
        self.build_gnn_module(), self.build_interaction_module()

        # Declare loss functions and metrics for training, validation, and testing
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = tm.Accuracy(num_classes=self.num_classes, average=None)
        self.train_prec = tm.Precision(num_classes=self.num_classes, average=None)
        self.train_recall = tm.Recall(num_classes=self.num_classes, average=None)

        self.val_acc = tm.Accuracy(num_classes=self.num_classes, average=None)
        self.val_prec = tm.Precision(num_classes=self.num_classes, average=None)
        self.val_recall = tm.Recall(num_classes=self.num_classes, average=None)
        self.val_auroc = tm.AUROC(num_classes=self.num_classes, average=None)
        self.val_auprc = tm.AveragePrecision(num_classes=self.num_classes)
        self.val_f1 = tm.F1(num_classes=self.num_classes, average=None)

        self.test_acc = tm.Accuracy(num_classes=self.num_classes, average=None)
        self.test_prec = tm.Precision(num_classes=self.num_classes, average=None)
        self.test_recall = tm.Recall(num_classes=self.num_classes, average=None)
        self.test_auroc = tm.AUROC(num_classes=self.num_classes, average=None)
        self.test_auprc = tm.AveragePrecision(num_classes=self.num_classes)
        self.test_f1 = tm.F1(num_classes=self.num_classes, average=None)

        # Reset learnable parameters and log hyperparameters
        self.reset_parameters()
        self.save_hyperparameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.using_node_embedding:
            # Reinitialize node input embedding
            glorot_orthogonal(self.node_in_embedding.weight, scale=2.0)

    def build_gnn_module(self):
        """Define all layers for the chosen GNN module."""
        # Marshal all GNN layers, allowing the user to choose which kind of graph learning scheme they would like to use
        num_node_input_feats = self.num_gnn_hidden_channels \
            if self.using_node_embedding \
            else self.num_node_input_feats
        if self.using_gcn:
            gnn_layers = [GraphConv(in_feats=num_node_input_feats,
                                    out_feats=num_node_input_feats,
                                    weight=True,
                                    activation=None,
                                    allow_zero_in_degree=False) for _ in range(self.num_gnn_layers)]
        elif self.using_gat:
            gnn_layers = [GATConv(in_feats=num_node_input_feats,
                                  out_feats=num_node_input_feats,
                                  num_heads=self.num_gnn_attention_heads,
                                  feat_drop=0.1,
                                  attn_drop=0.1,
                                  negative_slope=0.2,
                                  residual=False,
                                  activation=None,
                                  allow_zero_in_degree=False) for _ in range(self.num_gnn_layers)]
        else:  # Default to using a Geometric Transformer for learning node representations
            if self.num_gnn_layers > 0:
                gnn_layers = [DGLGeometricTransformer(node_count_limit=self.max_num_graph_nodes,
                                                      shared_embed_size=64,
                                                      dist_embed_size=8,
                                                      dir_embed_size=8,
                                                      orient_embed_size=8,
                                                      amide_embed_size=8,
                                                      num_hidden_channels=self.num_gnn_hidden_channels,
                                                      num_pre_res_blocks=2,
                                                      num_post_res_blocks=2,
                                                      activ_fn=self.gnn_activ_fn,
                                                      transformer_residual=True,
                                                      num_attention_heads=self.num_gnn_attention_heads,
                                                      norm_to_apply='batch',
                                                      dropout_rate=self.dropout_rate,
                                                      knn=self.nbrhd_size,
                                                      num_layers=self.num_gnn_layers,
                                                      feature_indices=FEATURE_INDICES,
                                                      disable_geometric_mode=self.disable_geometric_mode)]
            else:
                gnn_layers = []
        self.gnn_module = nn.ModuleList(gnn_layers)

    def get_interact_module(self):
        """Retrieve an interaction module of a specific type (e.g. Dilated ResNet or DeepLabV3Plus)."""
        if self.using_deeplab:
            interact_module = DeepLabV3Plus(
                encoder_name="resnet34",
                encoder_depth=self.num_interact_layers,
                encoder_output_stride=16,
                decoder_channels=self.num_interact_hidden_channels,
                decoder_atrous_rates=(12, 24, 36),
                in_channels=self.num_gnn_hidden_channels * 2,
                classes=self.num_classes,
                upsampling=4
            )
        else:  # Otherwise, default to using our dilated ResNet with squeeze-and-excitation (SE)
            interact_module = ResNet2DInputWithOptAttention(num_chunks=self.num_interact_layers,
                                                            init_channels=self.num_gnn_hidden_channels * 2,
                                                            num_channels=self.num_interact_hidden_channels,
                                                            num_classes=self.num_classes,
                                                            use_attention=self.use_interact_attention,
                                                            n_head=self.num_interact_attention_heads,
                                                            activ_fn=F.elu,
                                                            dropout=self.dropout_rate,
                                                            verbose=False)
        return interact_module

    def build_interaction_module(self):
        """Define all layers for the chosen interaction module."""
        # Dilated ResNets and DeepLabV3Plus package all their forward pass logic
        self.interact_module = self.get_interact_module()

    # ---------------------
    # Training
    # ---------------------
    def gnn_forward(self, graph: dgl.DGLGraph):
        """Make a forward pass through a single GNN module."""
        # Embed input features a priori
        if self.using_node_embedding:
            graph.ndata['f'] = self.node_in_embedding(graph.ndata['f']).squeeze()
        if self.using_gcn:
            # Forward propagate with each GNN layer
            for layer in self.gnn_module:
                # Cache the original batch number of nodes and edges
                batch_num_nodes, batch_num_edges = graph.batch_num_nodes(), graph.batch_num_edges()
                graph.ndata['f'] = layer(graph, graph.ndata['f'], edge_weight=graph.edata['f'][:, 1]).squeeze()
                # Retain the original batch number of nodes and edges
                graph.set_batch_num_nodes(batch_num_nodes), graph.set_batch_num_edges(batch_num_edges)
        elif self.using_gat:
            # Forward propagate with each GNN layer
            for layer in self.gnn_module:
                # Cache the original batch number of nodes and edges
                batch_num_nodes, batch_num_edges = graph.batch_num_nodes(), graph.batch_num_edges()
                if self.num_gnn_attention_heads > 1:
                    graph.ndata['f'] = torch.sum(layer(graph, graph.ndata['f']), dim=1)  # Sum the attention heads
                else:
                    graph.ndata['f'] = layer(graph, graph.ndata['f']).squeeze()
                # Retain the original batch number of nodes and edges
                graph.set_batch_num_nodes(batch_num_nodes), graph.set_batch_num_edges(batch_num_edges)
        else:  # The GeometricTransformer updates simply by returning a graph containing the updated node/edge feats
            for layer in self.gnn_module:
                graph = layer(graph)  # Geometric Transformers can handle their own depth
        # Unbatch and collect each individual graph's predicted node features
        graphs = dgl.unbatch(graph)
        node_feats = [graph.ndata['f'] for graph in graphs]
        return node_feats

    def interact_forward(self, interact_tensor: torch.Tensor):
        """Make a forward pass through the interaction module."""
        # Dilated ResNets and DeepLabV3Plus package all their forward pass logic
        logits = self.interact_module(interact_tensor)
        return logits

    def shared_step(self, graph1: dgl.DGLGraph, graph2: dgl.DGLGraph):
        """Make a forward pass through the entire siamese network."""
        # Learn structural features for each structure's nodes
        graph1_node_feats = self.gnn_forward(graph1)
        graph2_node_feats = self.gnn_forward(graph2)

        # Interleave node features from both graphs to achieve the desired interaction tensor
        current_phase_batch_size = len(graph1_node_feats)  # Use feature tensor collection length as phase's batch size
        graph1_is_within_size_limit = graph1.num_nodes() < self.max_num_residues
        graph2_is_within_size_limit = graph2.num_nodes() < self.max_num_residues
        both_graphs_within_limit = graph1_is_within_size_limit and graph2_is_within_size_limit
        # high_mem_model = self.using_dil_resnet  # Reinstate if wanting to restrict memory consumption with dil_resnet
        high_mem_model = False  # Ignore subsequencing for Dilated ResNets
        subsequencing_input = both_graphs_within_limit is False and current_phase_batch_size == 1 and high_mem_model
        if subsequencing_input:  # Need to subsequence input complexes to avoid running out of GPU memory
            interact_tensors = construct_subsequenced_interact_tensors(
                graph1_node_feats, graph2_node_feats, current_phase_batch_size, pad=self.pad,
                max_len=self.max_num_residues
            )
        else:
            interact_tensors = [
                construct_interact_tensor(
                    g1_node_feats, g2_node_feats, pad=self.pad, max_len=self.max_num_residues
                )
                for g1_node_feats, g2_node_feats in zip(graph1_node_feats, graph2_node_feats)
            ]

        # Handle for optional padding
        if self.pad:  # When subsequencing, we must address padding inside insert() below
            interact_tensors = torch.cat(interact_tensors)
            # Predict node-node pair interactions using an interaction module (i.e. series of interaction layers)
            logits = self.interact_forward(interact_tensors)
            # Remove any added padding from learned interaction tensors
            remove_padding_fn = remove_subsequenced_input_padding if subsequencing_input else remove_padding
            logits_list = remove_padding_fn(logits, graph1_node_feats, graph2_node_feats, self.max_num_residues)
        else:
            logits_list = [self.interact_forward(interact_tensor) for interact_tensor in interact_tensors]

        # Recombine subsequenced logits into the original interaction tensor's shape
        if subsequencing_input:
            if current_phase_batch_size == 1:
                interact_tensor = torch.zeros(1,
                                              self.num_classes,
                                              graph1.num_nodes(),
                                              graph2.num_nodes(),
                                              device=self.device)
                interact_tensor = insert_interact_tensor_logits(logits_list, interact_tensor, self.max_num_residues)
                logits_list = [interact_tensor]
            else:
                # TODO: Implement subsequence batching of graph batches
                raise NotImplementedError

        # Return network prediction and learned node and edge representations for both graphs
        g1_nf, g1_ef = graph1.ndata['f'].detach().cpu().numpy(), graph1.edata['f'].detach().cpu().numpy()
        g2_nf, g2_ef = graph2.ndata['f'].detach().cpu().numpy(), graph2.edata['f'].detach().cpu().numpy()
        return logits_list, g1_nf, g1_ef, g2_nf, g2_ef

    def downsample_examples(self, examples: torch.tensor):
        """Randomly sample enough negative pairs to achieve requested positive-negative class ratio (via shuffling)."""
        examples = examples[torch.randperm(len(examples))]  # Randomly shuffle training examples
        pos_examples = examples[examples[:, 2] == 1]  # Find out how many interacting pairs there are
        num_neg_pairs_to_sample = int(len(pos_examples) / self.pn_ratio)  # Determine negative sample size
        neg_examples = examples[examples[:, 2] == 0][:num_neg_pairs_to_sample]  # Sample negative pairs
        downsampled_examples = torch.cat((neg_examples, pos_examples))
        return downsampled_examples

    def training_step(self, batch, batch_idx):
        """Lightning calls this inside the training loop."""
        # Separate training batch from validation batch (with the latter being used for visualizations)
        train_batch, val_batch = batch['train_batch'], batch['val_batch']
        graph1, graph2, examples_list, filepaths = train_batch[0], train_batch[1], train_batch[2], train_batch[3]

        # Forward propagate with network layers
        logits_list, _, _, _, _ = self.shared_step(graph1, graph2)  # The forward method must be named something new

        # Collect flattened sampled logits and their corresponding labels
        sampled_examples = []
        sampled_logits = torch.tensor([], device=self.device)
        for logits, examples in zip(logits_list, examples_list):
            logits = logits.squeeze()  # Remove extraneous dimensions from predicted interaction matrices
            # examples = self.downsample_examples(examples) if self.pn_ratio > 0.0015 else examples
            sampled_examples.append(examples)  # Add modified examples into new list
            sampled_indices = examples[:, :2][:, 1] + logits.shape[2] * examples[:, :2][:, 0]  # 1d_idx = x + width * y
            flattened_logits = torch.flatten(logits, start_dim=1)
            flattened_sampled_logits = flattened_logits[:, sampled_indices]
            sampled_logits = torch.cat((sampled_logits, flattened_sampled_logits.transpose(1, 0)))
        examples = torch.cat(sampled_examples)

        # Down-weight negative pairs to achieve desired PN weight, and then up-weight positive pairs appropriately
        if self.weight_classes:
            neg_class_weight = 1.0  # Modify by how much negative class samples are weighted
            pos_class_weight = 5.0  # Modify by how much positive class samples are weighted
            class_weights = torch.tensor([neg_class_weight, pos_class_weight], device=self.device)
            loss_fn = nn.CrossEntropyLoss(
                weight=class_weights  # Weight each class separately for a given complex
            )
        else:
            loss_fn = nn.CrossEntropyLoss()

        # Make predictions
        preds = torch.softmax(sampled_logits, dim=1)
        preds_rounded = preds.clone()
        preds_rounded[:, 0] = (preds[:, 0] >= (1 - self.pos_prob_threshold)).float()
        preds_rounded[:, 1] = (preds[:, 1] >= self.pos_prob_threshold).float()
        labels = examples[:, 2]

        # Calculate the protein interface prediction (PICP) loss along with additional PICP metrics
        loss = loss_fn(sampled_logits, labels)  # Calculate loss of a single complex
        train_acc = self.train_acc(preds_rounded, labels)[1]  # Calculate Accuracy of a single complex
        train_prec = self.train_prec(preds_rounded, labels)[1]  # Calculate Precision of a single complex
        train_recall = self.train_recall(preds_rounded, labels)[1]  # Calculate Recall of a single complex

        # Log training step metric(s)
        self.log(f'train_ce', loss, on_step=False, on_epoch=True, sync_dist=True)

        # Manually evaluate training performance on a held-out validation dataset purely for visualizing model dynamics
        is_viz_epoch = self.current_epoch % self.viz_every_n_epochs == 0
        time_to_visualize_preds = is_viz_epoch and batch_idx == 0
        if time_to_visualize_preds:  # Visualize a single training and validation model prediction every 'n' epochs
            self.train(mode=False)  # Set the module to evaluation mode temporarily
            with torch.no_grad():
                # ------------
                # Train Sample
                # ------------
                # Reuse existing logits for the first training sample
                train_logits = logits_list[0].squeeze()
                train_len_1, train_len_2 = train_logits.shape[1:]

                # Construct the predicted M x N interaction tensor and its corresponding labels
                train_preds = torch.softmax(torch.flatten(train_logits, start_dim=1).transpose(1, 0), dim=1)
                train_preds_rounded = train_preds.clone()
                train_preds_rounded[:, 0] = (train_preds[:, 0] >= (1 - self.pos_prob_threshold)).float()
                train_preds_rounded[:, 1] = (train_preds[:, 1] >= self.pos_prob_threshold).float()
                train_preds = train_preds[:, 1].reshape(train_len_1, train_len_2).cpu().numpy()
                train_preds_rounded = train_preds_rounded[:, 1].reshape(train_len_1, train_len_2).cpu().numpy()
                train_labels = examples_list[0][:, 2].reshape(train_len_1, train_len_2).float().cpu().numpy()

                # ------------
                # Val Sample
                # ------------
                # Make a forward pass through the network for a held-out validation complex for visualization
                val_graph1, val_graph2, val_examples_list = val_batch[0], val_batch[1], val_batch[2][0]

                # Forward propagate with network layers without accumulating any gradients
                val_logits, _, _, _, _ = self.shared_step(val_graph1, val_graph2)[0].squeeze()
                val_len_1, val_len_2 = val_logits.shape[1:]

                # Construct the predicted M x N interaction tensor and its corresponding labels
                val_preds = torch.softmax(torch.flatten(val_logits, start_dim=1).transpose(1, 0), dim=1)
                val_preds_rounded = val_preds.clone()
                val_preds_rounded[:, 0] = (val_preds[:, 0] >= (1 - self.pos_prob_threshold)).float()
                val_preds_rounded[:, 1] = (val_preds[:, 1] >= self.pos_prob_threshold).float()
                val_preds = val_preds[:, 1].reshape(val_len_1, val_len_2).cpu().numpy()
                val_preds_rounded = val_preds_rounded[:, 1].reshape(val_len_1, val_len_2).cpu().numpy()
                val_labels = val_examples_list[:, 2].reshape(val_len_1, val_len_2).float().cpu().numpy()

                # ------------
                # Visualize
                # ------------
                if self.use_wandb_logger:
                    # Convert predictions and labels to grayscale images
                    viz_preds = [wandb.Image(viz) for viz in [train_preds, val_preds]]
                    viz_preds_rounded = [wandb.Image(viz) for viz in [train_preds_rounded, val_preds_rounded]]
                    viz_labels = [wandb.Image(viz) for viz in [train_labels, val_labels]]

                    # Log validation predictions with their ground-truth interaction tensors to WandB for inspection
                    self.trainer.logger.experiment.log({'sample_preds': viz_preds})
                    self.trainer.logger.experiment.log({'sample_preds_rounded': viz_preds_rounded})
                    self.trainer.logger.experiment.log({'sample_labels': viz_labels})

                else:  # Assume we are instead using the TensorBoardLogger
                    self.logger.experiment.add_image(
                        'sample_train_preds', train_preds, self.global_step, dataformats='HW'
                    )
                    self.logger.experiment.add_image(
                        'sample_val_preds', val_preds, self.global_step, dataformats='HW'
                    )
                    self.logger.experiment.add_image(
                        'sample_train_preds_rounded', train_preds_rounded, self.global_step, dataformats='HW'
                    )
                    self.logger.experiment.add_image(
                        'sample_val_preds_rounded', val_preds_rounded, self.global_step, dataformats='HW'
                    )
                    self.logger.experiment.add_image(
                        'sample_train_labels', train_labels, self.global_step, dataformats='HW'
                    )
                    self.logger.experiment.add_image(
                        'sample_val_labels', val_labels, self.global_step, dataformats='HW'
                    )

            # Set the current LightningModule back to training mode
            self.train(mode=True)

        return {
            'loss': loss,
            'train_acc': train_acc,
            'train_prec': train_prec,
            'train_recall': train_recall
        }

    def training_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        """Lightning calls this at the end of every training epoch."""
        # Tuplize scores for the current device (e.g. Rank 0)
        train_accs = torch.cat([output_dict['train_acc'].unsqueeze(0) for output_dict in outputs])
        train_precs = torch.cat([output_dict['train_prec'].unsqueeze(0) for output_dict in outputs])
        train_recalls = torch.cat([output_dict['train_recall'].unsqueeze(0) for output_dict in outputs])

        # Concatenate scores over all devices (e.g. Rank 0 | ... | Rank N) - Warning: Memory Intensive
        train_accs = torch.cat([train_acc for train_acc in self.all_gather(train_accs)], dim=0)
        train_precs = torch.cat([train_prec for train_prec in self.all_gather(train_precs)], dim=0)
        train_recalls = torch.cat([train_recall for train_recall in self.all_gather(train_recalls)], dim=0)

        # Reset training TorchMetrics for all devices
        self.train_acc.reset()
        self.train_prec.reset()
        self.train_recall.reset()

        # Log metric(s) aggregated from all ranks
        self.log('med_train_acc', torch.median(train_accs))  # Log MedAccuracy of an epoch
        self.log('med_train_prec', torch.median(train_precs))  # Log MedPrecision of an epoch
        self.log('med_train_recall', torch.median(train_recalls))  # Log MedRecall of an epoch

    def validation_step(self, batch, batch_idx):
        """Lightning calls this inside the validation loop."""
        # Make a forward pass through the network for a batch of protein complexes
        graph1, graph2, examples_list, filepaths = batch[0], batch[1], batch[2], batch[3]

        # Forward propagate with network layers
        logits_list, _, _, _, _ = self.shared_step(graph1, graph2)

        # Collect flattened, sampled logits and their corresponding labels
        sampled_logits = torch.tensor([], device=self.device)
        for i, (logits, examples) in enumerate(zip(logits_list, examples_list)):
            logits = logits.squeeze()  # Remove extraneous dimensions from predicted interaction matrices
            examples_list[i] = examples  # Replace original examples tensor in-place
            sampled_indices = examples[:, :2][:, 1] + logits.shape[2] * examples[:, :2][:, 0]  # 1d_idx = x + width * y
            flattened_logits = torch.flatten(logits, start_dim=1)
            flattened_sampled_logits = flattened_logits[:, sampled_indices]
            sampled_logits = torch.cat((sampled_logits, flattened_sampled_logits.transpose(1, 0)))
        examples = torch.cat(examples_list)

        # Make predictions
        preds = torch.softmax(sampled_logits, dim=1)
        preds_rounded = preds.clone()
        preds_rounded[:, 0] = (preds[:, 0] >= (1 - self.pos_prob_threshold)).float()
        preds_rounded[:, 1] = (preds[:, 1] >= self.pos_prob_threshold).float()
        labels = examples[:, 2]

        # Calculate top-k metrics
        calculating_l_by_n_metrics = True
        # Log only first 50 validation top-k precisions to limit algorithmic complexity due to sorting (if requested)
        # calculating_l_by_n_metrics = batch_idx in [i for i in range(50)]
        if calculating_l_by_n_metrics:
            l = graph1.num_nodes() + graph2.num_nodes()
            sorted_pred_indices = torch.argsort(preds[:, 1], descending=True)
            top_10_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=10)
            top_25_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=25)
            top_50_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=50) if l > 50 else 0.0  # Catch short seq.
            top_l_by_10_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=(l // 10))
            top_l_by_5_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=(l // 5))
            top_l_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=l)

        # Calculate the protein interface prediction (PICP) loss along with additional PIP metrics
        loss = self.loss_fn(sampled_logits, labels)  # Calculate loss of a single complex
        val_acc = self.val_acc(preds_rounded, labels)[1]  # Calculate Accuracy of a single complex
        val_prec = self.val_prec(preds_rounded, labels)[1]  # Calculate Precision of a single complex
        val_recall = self.val_recall(preds_rounded, labels)[1]  # Calculate Recall of a single complex
        val_f1 = self.val_f1(preds_rounded, labels)[1]  # Calculate F1 score of a single complex
        val_auroc = self.val_auroc(preds, labels)[1]  # Calculate AUROC of a complex
        val_auprc = self.val_auprc(preds, labels)[1]  # Calculate AveragePrecision (i.e. AUPRC) of a complex

        # Log validation step metric(s)
        self.log(f'val_ce', loss, sync_dist=True)
        if calculating_l_by_n_metrics:
            self.log('val_top_10_prec', top_10_prec, sync_dist=True)
            self.log('val_top_25_prec', top_25_prec, sync_dist=True)
            self.log('val_top_50_prec', top_50_prec, sync_dist=True)
            self.log('val_top_l_by_10_prec', top_l_by_10_prec, sync_dist=True)
            self.log('val_top_l_by_5_prec', top_l_by_5_prec, sync_dist=True)
            self.log('val_top_l_prec', top_l_prec, sync_dist=True)

        return {
            'loss': loss,
            'val_acc': val_acc,
            'val_prec': val_prec,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'val_auroc': val_auroc,
            'val_auprc': val_auprc
        }

    def validation_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        """Lightning calls this at the end of every validation epoch."""
        # Tuplize scores for the current device (e.g. Rank 0)
        val_accs = torch.cat([output_dict['val_acc'].unsqueeze(0) for output_dict in outputs])
        val_precs = torch.cat([output_dict['val_prec'].unsqueeze(0) for output_dict in outputs])
        val_recalls = torch.cat([output_dict['val_recall'].unsqueeze(0) for output_dict in outputs])
        val_f1s = torch.cat([output_dict['val_f1'].unsqueeze(0) for output_dict in outputs])
        val_aurocs = torch.cat([output_dict['val_auroc'].unsqueeze(0) for output_dict in outputs])
        val_auprcs = torch.cat([output_dict['val_auprc'].unsqueeze(0) for output_dict in outputs])

        # Concatenate scores over all devices (e.g. Rank 0 | ... | Rank N) - Warning: Memory Intensive
        val_accs = torch.cat([val_acc for val_acc in self.all_gather(val_accs)])
        val_precs = torch.cat([val_prec for val_prec in self.all_gather(val_precs)])
        val_recalls = torch.cat([val_recall for val_recall in self.all_gather(val_recalls)])
        val_f1s = torch.cat([val_f1 for val_f1 in self.all_gather(val_f1s)])
        val_aurocs = torch.cat([val_auroc for val_auroc in self.all_gather(val_aurocs)])
        val_auprcs = torch.cat([val_auprc for val_auprc in self.all_gather(val_auprcs)])

        # Reset validation TorchMetrics for all devices
        self.val_acc.reset()
        self.val_prec.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()

        # Log metric(s) aggregated from all ranks
        self.log('med_val_acc', torch.median(val_accs))  # Log MedAccuracy of an epoch
        self.log('med_val_prec', torch.median(val_precs))  # Log MedPrecision of an epoch
        self.log('med_val_recall', torch.median(val_recalls))  # Log MedRecall of an epoch
        self.log('med_val_f1', torch.median(val_f1s))  # Log MedF1 of an epoch
        self.log('med_val_auroc', torch.median(val_aurocs))  # Log MedAUROC of an epoch
        self.log('med_val_auprc', torch.median(val_auprcs))  # Log epoch MedAveragePrecision

    def test_step(self, batch, batch_idx):
        """Lightning calls this inside the testing loop."""
        # Make a forward pass through the network for a batch of protein complexes (batch_size=1)
        graph1, graph2, examples_list, filepaths = batch[0], batch[1], batch[2], batch[3]

        # Forward propagate with network layers
        logits_list, _, _, _, _ = self.shared_step(graph1, graph2)

        # Collect flattened, sampled logits and their corresponding labels
        sampled_logits = torch.tensor([], device=self.device)
        for i, (logits, examples) in enumerate(zip(logits_list, examples_list)):
            logits = logits.squeeze()  # Remove extraneous dimensions from predicted interaction matrices
            examples_list[i] = examples  # Replace original examples tensor in-place
            sampled_indices = examples[:, :2][:, 1] + logits.shape[2] * examples[:, :2][:, 0]  # 1d_idx = x + width * y
            flattened_logits = torch.flatten(logits, start_dim=1)
            flattened_sampled_logits = flattened_logits[:, sampled_indices]
            sampled_logits = torch.cat((sampled_logits, flattened_sampled_logits.transpose(1, 0)))
        examples = torch.cat(examples_list)

        # Make predictions
        preds = torch.softmax(sampled_logits, dim=1)
        preds_rounded = preds.clone()
        preds_rounded[:, 0] = (preds[:, 0] >= (1 - self.pos_prob_threshold)).float()
        preds_rounded[:, 1] = (preds[:, 1] >= self.pos_prob_threshold).float()
        labels = examples[:, 2]

        # Calculate top-k metrics
        l = min(graph1.num_nodes(), graph2.num_nodes())  # Use the smallest length of the two chains as our denominator
        sorted_pred_indices = torch.argsort(preds[:, 1], descending=True)
        top_10_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=10)
        top_25_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=25)
        top_50_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=50) if l > 50 else 0.0  # Catch short seq.
        top_l_by_10_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=(l // 10))
        top_l_by_5_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=(l // 5))
        top_l_prec = calculate_top_k_prec(sorted_pred_indices, labels, k=l)

        # Calculate the protein interface prediction (PICP) loss along with additional PIP metrics
        loss = self.loss_fn(sampled_logits, labels)  # Calculate loss of a single complex
        test_acc = self.test_acc(preds_rounded, labels)[1]  # Calculate Accuracy of a single complex
        test_prec = self.test_prec(preds_rounded, labels)[1]  # Calculate Precision of a single complex
        test_recall = self.test_recall(preds_rounded, labels)[1]  # Calculate Recall of a single complex
        test_f1 = self.test_f1(preds_rounded, labels)[1]  # Calculate F1 score of a single complex
        test_auroc = self.test_auroc(preds, labels)[1]  # Calculate AUROC of a complex
        test_auprc = self.test_auprc(preds, labels)[1]  # Calculate AveragePrecision (i.e. AUPRC) of a complex

        # Manually evaluate test performance by collecting all predicted and ground-truth interaction tensors
        test_preds = preds.detach()
        test_preds_rounded = test_preds.clone()
        test_preds_rounded[:, 0] = (test_preds[:, 0] >= (1 - self.pos_prob_threshold)).float()
        test_preds_rounded[:, 1] = (test_preds[:, 1] >= self.pos_prob_threshold).float()
        test_preds = test_preds[:, 1].reshape(graph1.num_nodes(), graph2.num_nodes()).cpu().numpy()
        test_preds_rounded = test_preds_rounded[:, 1].reshape(graph1.num_nodes(), graph2.num_nodes()).cpu().numpy()

        test_labels = examples[:, 2].detach()
        test_labels = test_labels.reshape(graph1.num_nodes(), graph2.num_nodes()).float().cpu().numpy()

        # Log test step metric(s)
        self.log(f'test_ce', loss, sync_dist=True)
        self.log('test_top_10_prec', top_10_prec, sync_dist=True)
        self.log('test_top_25_prec', top_25_prec, sync_dist=True)
        self.log('test_top_50_prec', top_50_prec, sync_dist=True)
        self.log('test_top_l_by_10_prec', top_l_by_10_prec, sync_dist=True)
        self.log('test_top_l_by_5_prec', top_l_by_5_prec, sync_dist=True)
        self.log('test_top_l_prec', top_l_prec, sync_dist=True)

        return {
            'loss': loss,
            'test_acc': test_acc,
            'test_prec': test_prec,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auroc': test_auroc,
            'test_auprc': test_auprc,
            'test_preds': test_preds,
            'test_preds_rounded': test_preds_rounded,
            'test_labels': test_labels,
            'top_10_prec': top_10_prec,
            'top_l_by_10_prec': top_l_by_10_prec,
            'top_l_by_5_prec': top_l_by_5_prec,
            'target': filepaths[0].split(os.sep)[-1][:4]
        }

    def test_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT):
        """Lightning calls this at the end of every test epoch."""
        # Tuplize scores for the current device (e.g. Rank 0)
        test_accs = torch.cat([output_dict['test_acc'].unsqueeze(0) for output_dict in outputs]).unsqueeze(1)
        test_precs = torch.cat([output_dict['test_prec'].unsqueeze(0) for output_dict in outputs]).unsqueeze(1)
        test_recalls = torch.cat([output_dict['test_recall'].unsqueeze(0) for output_dict in outputs]).unsqueeze(1)
        test_f1s = torch.cat([output_dict['test_f1'].unsqueeze(0) for output_dict in outputs]).unsqueeze(1)
        test_aurocs = torch.cat([output_dict['test_auroc'].unsqueeze(0) for output_dict in outputs]).unsqueeze(1)
        test_auprcs = torch.cat([output_dict['test_auprc'].unsqueeze(0) for output_dict in outputs]).unsqueeze(1)

        # Concatenate scores over all devices (e.g. Rank 0 | ... | Rank N) - Warning: Memory Intensive
        test_accs = torch.cat([test_acc for test_acc in self.all_gather(test_accs)])
        test_precs = torch.cat([test_prec for test_prec in self.all_gather(test_precs)])
        test_recalls = torch.cat([test_recall for test_recall in self.all_gather(test_recalls)])
        test_f1s = torch.cat([test_f1 for test_f1 in self.all_gather(test_f1s)])
        test_aurocs = torch.cat([test_auroc for test_auroc in self.all_gather(test_aurocs)])
        test_auprcs = torch.cat([test_auprc for test_auprc in self.all_gather(test_auprcs)])

        if self.use_wandb_logger:
            test_preds = [wandb.Image(output_dict['test_preds']) for output_dict in outputs]  # Convert to image
            test_preds_rounded = [wandb.Image(output_dict['test_preds_rounded']) for output_dict in outputs]  # Rounded
            test_labels = [wandb.Image(output_dict['test_labels']) for output_dict in outputs]  # Convert to image
        else:  # Assume we are instead using the TensorBoardLogger
            test_preds = [output_dict['test_preds'] for output_dict in outputs]
            test_preds_rounded = [(output_dict['test_preds_rounded']) for output_dict in outputs]
            test_labels = [output_dict['test_labels'] for output_dict in outputs]

        # Write out test top-k precision results to CSV
        prec_data = {
            'top_10_prec': [extract_object(output_dict['top_10_prec']) for output_dict in outputs],
            'top_l_by_10_prec': [extract_object(output_dict['top_l_by_10_prec']) for output_dict in outputs],
            'top_l_by_5_prec': [extract_object(output_dict['top_l_by_5_prec']) for output_dict in outputs],
            'target': [extract_object(output_dict['target']) for output_dict in outputs],
        }
        prec_df = pd.DataFrame(data=prec_data)
        prec_df_name_prefix = 'casp_capri' if self.testing_with_casp_capri else 'dips_plus_test'
        prec_df_name = prec_df_name_prefix + '_top_prec.csv'
        prec_df.to_csv(prec_df_name)

        if not self.testing_with_casp_capri:  # Testing with DIPS-Plus
            # Filter out all but the first 55 test predictions and labels to reduce storage requirements
            test_preds, test_preds_rounded, test_labels = test_preds[:55], test_preds_rounded[:55], test_labels[:55]

        # Reset test TorchMetrics for all devices
        self.test_acc.reset()
        self.test_prec.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_auroc.reset()
        self.test_auprc.reset()

        # Log metric(s) aggregated from all ranks
        self.log('med_test_acc', torch.median(test_accs))  # Log MedAccuracy of an epoch
        self.log('med_test_prec', torch.median(test_precs))  # Log MedPrecision of an epoch
        self.log('med_test_recall', torch.median(test_recalls))  # Log MedRecall of an epoch
        self.log('med_test_f1', torch.median(test_f1s))  # Log MedF1 of an epoch
        self.log('med_test_auroc', torch.median(test_aurocs))  # Log MedAUROC of an epoch
        self.log('med_test_auprc', torch.median(test_auprcs))  # Log epoch MedAveragePrecision

        # Log test predictions with their ground-truth interaction tensors to WandB for visual inspection
        if self.use_wandb_logger:
            self.trainer.logger.experiment.log({'test_preds': test_preds})
            self.trainer.logger.experiment.log({'test_preds_rounded': test_preds_rounded})
            self.trainer.logger.experiment.log({'test_labels': test_labels})
        else:  # Assume we are instead using the TensorBoardLogger
            for i, (t_preds, t_preds_rounded, t_labels) in enumerate(zip(test_preds, test_preds_rounded, test_labels)):
                self.logger.experiment.add_image('test_preds', t_preds, i, dataformats='HW')
                self.logger.experiment.add_image('test_preds_rounded', t_preds_rounded, i, dataformats='HW')
                self.logger.experiment.add_image('test_labels', t_labels, i, dataformats='HW')

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Lightning calls this inside the predict loop."""
        # Make predictions for a batch of protein complexes
        graph1, graph2 = batch[0], batch[1]
        # Forward propagate with network layers - (batch_size x self.num_channels x len(graph1) x len(graph2))
        logits_list, g1_nf, g1_ef, g2_nf, g2_ef = self.shared_step(graph1, graph2)
        return logits_list, g1_nf, g1_ef, g2_nf, g2_ef

    # ---------------------
    # Training Setup
    # ---------------------
    def configure_optimizers(self):
        """Called to configure the trainer's optimizer(s)."""
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-8, verbose=True),
                "monitor": self.metric_to_track,
            }
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # -----------------
        # Model arguments
        # -----------------
        parser.add_argument('--gnn_layer_type', type=str, default='geotran',
                            help='Which type of GNN layer to use'
                                 ' (i.e. gat for DGLGATConv or geotran for DGLGeometricTransformer)')
        parser.add_argument('--num_gnn_hidden_channels', type=int, default=128,
                            help='Dimensionality of GNN filters (for nodes and edges alike after embedding)')
        parser.add_argument('--num_gnn_attention_heads', type=int, default=4,
                            help='How many multi-head GNN attention blocks to run in parallel')
        parser.add_argument('--interact_module_type', type=str, default='dil_resnet',
                            help='Which type of dense prediction interaction module to use'
                                 ' (i.e. dil_resnet for Dilated ResNet, or deeplab for DeepLabV3Plus)')
        parser.add_argument('--num_interact_hidden_channels', type=int, default=128,
                            help='Dimensionality of interaction module filters')
        parser.add_argument('--use_interact_attention', action='store_true', dest='use_interact_attention',
                            help='Whether to employ attention in, for example, a Dilated ResNet')
        parser.add_argument('--num_interact_attention_heads', type=int, default=4,
                            help='How many multi-head interact attention blocks to use in parallel')
        parser.add_argument('--disable_geometric_mode', action='store_true', dest='disable_geometric_mode',
                            help='Whether to convert the Geometric Transformer into the original Graph Transformer')
        parser.add_argument('--viz_every_n_epochs', type=int, default=1,
                            help='By how many epochs to space out model prediction visualizations during training')
        parser.add_argument('--weight_classes', action='store_true', dest='weight_classes',
                            help='Whether to use class weighting in our training Cross Entropy')
        parser.add_argument('--left_pdb_filepath', type=str, default='test_data/4heq_l.pdb',
                            help='A filepath to the left input PDB chain')
        parser.add_argument('--right_pdb_filepath', type=str, default='test_data/4heq_r.pdb',
                            help='A filepath to the right input PDB chain')
        return parser
