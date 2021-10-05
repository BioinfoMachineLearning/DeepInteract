from typing import List, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.nn.pytorch import pairwise_squared_distance
from torch import FloatTensor

from project.utils.deepinteract_constants import DEFAULT_MISSING_HSAAC, HSAAC_DIM

try:
    from types import SliceType
except ImportError:
    SliceType = slice


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from GraphTransformer (https://github.com/graphdeeplearning/graphtransformer/):
# -------------------------------------------------------------------------------------------------------------------------------------
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        """Compute the dot product between source nodes' and destination nodes' representations."""
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

    return func


def scaling(field, scale_constant, clip_constant):
    def func(edges):
        """Scale edge representation value using a constant divisor."""
        return {field: ((edges.data[field]) / scale_constant).clamp(-clip_constant, clip_constant)}

    return func


def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """

    def func(edges):
        """Improve implicit attention scores with explicit edge features, if available."""
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

    return func


def out_edge_features(edge_feat):
    def func(edges):
        """Copy edge features to be passed to FFN_e."""
        return {'e_out': edges.data[edge_feat]}

    return func


def exp(field, clip_constant):
    def func(edges):
        """Clamp edge representations for softmax numerical stability."""
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-clip_constant, clip_constant))}

    return func


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Atom3D (https://github.com/drorlab/atom3d/blob/master/benchmarking/pytorch_geometric/ppi_dataloader.py):
# -------------------------------------------------------------------------------------------------------------------------------------
def prot_df_to_dgl_graph_feats(df: pd.DataFrame, feat_cols: List, allowable_feats: List[List], knn: int):
    r"""Convert protein in dataframe representation to a graph compatible with DGL, where each node is a residue.

    :param df: Protein structure in dataframe format.
    :type df: pandas.DataFrame
    :param feat_cols: Columns of dataframe in which to find node feature values. For example, for residues use ``feat_cols=["element", ...]`` and for residues use ``feat_cols=["resname", ...], or both!``
    :type feat_cols: list[list[Any]]
    :param allowable_feats: List of lists containing all possible values of node type, to be converted into 1-hot node features.
        Any elements in ``feat_col`` that are not found in ``allowable_feats`` will be added to an appended "unknown" bin (see :func:`atom3d.util.graph.one_of_k_encoding_unk`).
    :param knn: Maximum number of nearest neighbors (i.e. edges) to allow for a given node.
    :type knn: int

    :return: tuple containing
        - knn_graph (dgl.DGLGraph): K-nearest neighbor graph for the structure DataFrame given.

        - pairwise_dists (torch.FloatTensor): Pairwise squared distances for the K-nearest neighbor graph's coordinates.

        - node_coords (torch.FloatTensor): Cartesian coordinates of each node.

        - node_feats (torch.FloatTensor): Features for each node, one-hot encoded by values in ``allowable_feats``.
    :rtype: Tuple
    """
    # Exit early if feat_cols or allowable_feats do not align in dimensionality
    if len(feat_cols) != len(allowable_feats):
        raise Exception('feat_cols does not match the length of allowable_feats')

    # Aggregate structure-based node features
    node_feats = FloatTensor([])
    for i in range(len(feat_cols)):
        # Search through embedded 2D list for allowable values
        feat_vecs = [one_of_k_encoding_unk(feat, allowable_feats[i], feat_cols[i]) for feat in df[feat_cols[i]]]
        one_hot_feat_vecs = FloatTensor(feat_vecs)
        node_feats = torch.cat((node_feats, one_hot_feat_vecs), 1)

    # Organize residue coordinates into a FloatTensor
    node_coords = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float32)

    # Define edges - KNN argument determines whether a residue-residue edge gets created in the resulting graph
    knn_graph = dgl.knn_graph(node_coords, knn)
    pairwise_dists = torch.topk(pairwise_squared_distance(node_coords), knn, 1, largest=False).values

    return knn_graph, pairwise_dists, node_coords, node_feats


def one_of_k_encoding_unk(feat, allowable_set, feat_col):
    """Converts input to 1-hot encoding given a set of (or sets of) allowable values. Additionally maps inputs not in the allowable set to the last element."""
    if len(allowable_set) == 0:  # e.g. RSA values
        return [feat]
    elif len(allowable_set) == 1 and type(allowable_set[0]) == list and len(allowable_set[0]) == 0:  # e.g. HSAAC values
        if len(feat) == 0:
            return DEFAULT_MISSING_HSAAC if feat_col == 'hsaac' else []  # Else means skip encoding amide_norm_vec
        if feat_col == 'hsaac' and len(feat) > HSAAC_DIM:  # Handle for edge case from postprocessing
            return np.array(DEFAULT_MISSING_HSAAC)
        return feat if feat_col == 'hsaac' or feat_col == 'sequence_feats' else []  # Else means skip encoding amide_norm_vec as a node feature
    else:  # e.g. Residue element type values
        if feat not in allowable_set:
            feat = allowable_set[-1]
        return list(map(lambda s: feat == s, allowable_set))
