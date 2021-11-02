import difflib
import itertools
import logging
import os
import pickle
import random
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import atom3.case as ca
import atom3.complex as comp
import atom3.conservation as con
import atom3.database as db
import atom3.neighbors as nb
import atom3.pair as pair
import atom3.parse as pa
import dgl
import dill
import numpy as np
import pandas as pd
import parallel as par
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from Bio import pairwise2
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from biopandas.pdb import PandasPdb
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from project.utils.deepinteract_constants import FEAT_COLS, ALLOWABLE_FEATS, D3TO1
from project.utils.dips_plus_utils import postprocess_pruned_pairs, impute_postprocessed_missing_feature_values
from project.utils.graph_utils import prot_df_to_dgl_graph_feats
from project.utils.protein_feature_utils import GeometricProteinFeatures

try:
    from types import SliceType
except ImportError:
    SliceType = slice


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------
def glorot_orthogonal(tensor, scale):
    """Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def dgl_collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)


def dgl_picp_collate(complex_dicts: List[dict]):
    """Assemble a protein complex dictionary batch into two large batched DGLGraphs and a batched labels tensor."""
    batched_graph1 = dgl.batch([complex_dict['graph1'] for complex_dict in complex_dicts])
    batched_graph2 = dgl.batch([complex_dict['graph2'] for complex_dict in complex_dicts])
    examples_list = [complex_dict['examples'] for complex_dict in complex_dicts]
    complex_filepaths = [complex_dict['filepath'] for complex_dict in complex_dicts]
    return batched_graph1, batched_graph2, examples_list, complex_filepaths


def get_geo_feats_from_edges(orig_edge_feats: torch.Tensor, feature_indices: dict):
    """Retrieve and return geometric features from a given batch of edges."""
    dist_feats = orig_edge_feats[:, feature_indices['edge_dist_feats_start']:feature_indices['edge_dist_feats_end']]
    dir_feats = orig_edge_feats[:, feature_indices['edge_dir_feats_start']:feature_indices['edge_dir_feats_end']]
    o_feats = orig_edge_feats[:, feature_indices['edge_orient_feats_start']:feature_indices['edge_orient_feats_end']]
    amide_feats = orig_edge_feats[:, feature_indices['edge_amide_angles']]
    return dist_feats, dir_feats, o_feats, amide_feats


def min_max_normalize_tensor(tensor: torch.Tensor, device=None):
    """Normalize provided tensor to have values be in range [0, 1]."""
    min_value = min(tensor)
    max_value = max(tensor)
    tensor = torch.tensor([(value - min_value) / (max_value - min_value) for value in tensor], device=device)
    return tensor


def construct_filenames_frame_txt_filenames(mode: str, percent_to_use: float, filename_sampling: bool, root: str):
    """Build the file path of the requested filename DataFrame text file."""
    base_txt_filename = f'pairs-postprocessed' if mode == 'full' else f'pairs-postprocessed-{mode}'
    filenames_frame_txt_filename = base_txt_filename + f'-{int(percent_to_use * 100)}%-sampled.txt' \
        if filename_sampling else base_txt_filename + '.txt'
    filenames_frame_txt_filepath = os.path.join(root, filenames_frame_txt_filename)
    return base_txt_filename, filenames_frame_txt_filename, filenames_frame_txt_filepath


def build_filenames_frame_error_message(dataset: str, task: str, filenames_frame_txt_filepath: str):
    """Assemble the standard error message for a corrupt or missing filenames DataFrame text file."""
    return f'Unable to {task} {dataset} filenames text file' \
           f' (i.e. {filenames_frame_txt_filepath}).' \
           f' Please make sure it is downloaded and not corrupted.'


def calculate_and_store_dists_in_graph(graph: dgl.DGLGraph, init=False):
    """Derive all node-node distance features from a given batch of DGLGraphs."""
    graphs = dgl.unbatch(graph)
    for graph in graphs:
        graph.edata['c'] = graph.edata['c'] \
            if init \
            else graph.ndata['x'][graph.edges()[1]] - graph.ndata['x'][graph.edges()[0]]
        graph.edata['r'] = torch.sum(graph.edata['c'] ** 2, 1).reshape(-1, 1)
    graph = dgl.batch(graphs)
    return graph


def construct_subsequenced_interact_tensors(graph1_feats: List[torch.Tensor], graph2_feats: List[torch.Tensor],
                                            batch_size: int, pad=False, max_len=256):
    """Build subsequenced interaction tensors for node representations, optionally padding up to the node limit."""
    if batch_size == 1:  # User is using a singular batch size, so we can batch the single input complex
        # Unpack both graphs' features
        graph1_feats, graph2_feats = graph1_feats[0], graph2_feats[0]

        # Collect subsequence batches from the first graph
        g1_subseq_batches = []
        num_g1_subseq_batches = 1 + ((len(graph1_feats) - 1) // max_len)
        for i in range(num_g1_subseq_batches):
            index_iterator = max_len if len(graph1_feats) > max_len else len(graph1_feats)
            start_index, end_index = i * index_iterator, (i + 1) * index_iterator
            g1_subseq_batches.append(graph1_feats[start_index: end_index, :])

        # Collect subsequence batches from the second graph
        g2_subseq_batches = []
        num_g2_subseq_batches = 1 + ((len(graph2_feats) - 1) // max_len)
        for i in range(num_g2_subseq_batches):
            index_iterator = max_len if len(graph2_feats) > max_len else len(graph2_feats)
            start_index, end_index = i * index_iterator, (i + 1) * index_iterator
            g2_subseq_batches.append(graph2_feats[start_index: end_index, :])

        subseq_batch_combos = list(itertools.product(g1_subseq_batches, g2_subseq_batches))

        # Repackage collected subsequence batches
        interact_tensors = [
            construct_interact_tensor(graph1_feats, graph2_feats, pad=pad, max_len=max_len)
            for graph1_feats, graph2_feats in subseq_batch_combos
        ]
    else:
        # TODO: Implement subsequence batching of graph batches
        raise NotImplementedError
    return interact_tensors


def construct_interact_tensor(graph1_feats: torch.Tensor, graph2_feats: torch.Tensor, pad=False, max_len=256):
    """Build the interaction tensor for given node representations, optionally padding up to the node count limit."""
    # Get descriptors and reshaped versions of the input feature matrices
    len_1, len_2 = graph1_feats.shape[0], graph2_feats.shape[0]
    x_a, x_b = graph1_feats.permute(1, 0).unsqueeze(0), graph2_feats.permute(1, 0).unsqueeze(0)
    if pad:
        x_a_num_zeros = max_len - x_a.shape[2]
        x_b_num_zeros = max_len - x_b.shape[2]
        x_a = F.pad(x_a, (0, x_a_num_zeros, 0, 0, 0, 0))  # Pad the start of 3D tensors
        x_b = F.pad(x_b, (0, x_b_num_zeros, 0, 0, 0, 0))  # Pad the end of 3D tensors
        len_1, len_2 = max_len, max_len
    # Interleave 2D input matrices into a 3D interaction tensor
    interact_tensor = torch.cat((torch.repeat_interleave(x_a.unsqueeze(3), repeats=len_2, dim=3),
                                 torch.repeat_interleave(x_b.unsqueeze(2), repeats=len_1, dim=2)), dim=1)
    return interact_tensor


def remove_padding(logits: torch.Tensor, g1_node_feats: torch.tensor, g2_node_feats: torch.tensor, max_len: int):
    """Discard zero padding added to each 2D interaction tensor a posteriori."""
    logits_with_padding_removed = []
    for i, (g1_nf, g2_nf) in enumerate(zip(g1_node_feats, g2_node_feats)):
        sliced_logits = logits[i, :, :g1_nf.shape[0], :g2_nf.shape[0]]
        logits_with_padding_removed.append(sliced_logits.squeeze())  # Remove extraneous 1-channel upon appending
    return logits_with_padding_removed


def remove_subsequenced_input_padding(logits: torch.Tensor, g1_nf: torch.tensor, g2_nf: torch.tensor, max_len: int):
    """Discard zero padding added to each subsequenced 2D interaction tensor a posteriori."""
    # Initialize variables for tracking state
    logits_with_padding_removed = []
    idx_1, idx_2 = g1_nf[0].shape[0], g2_nf[0].shape[0]
    orig_idx_1, idx_1_iter, orig_idx_2, idx_2_iter = idx_1, 0, idx_2, 1
    idx_1_overflow, idx_2_overflow = ((idx_1 + max_len) % max_len), ((idx_2 + max_len) % max_len)
    just_finished_idx_1, just_finished_idx_2, return_now = False, False, True
    traversing_idx_1, idx_1_traversed, traversing_idx_2, idx_2_traversed = False, False, True, False

    # Begin iterations over each incoming "subsequenced" logits tensor
    for i in range(logits.shape[0]):
        if i == (logits.shape[0] - 1):  # Handle the last iteration's indices uniquely
            # Add the surplus amount to the current multiple of max_len we are at
            idx_1 = max_len if idx_1_overflow == 0 else min(idx_1, idx_1_overflow)
            idx_2 = max_len if idx_2_overflow == 0 else min(idx_2, idx_2_overflow)
        else:  # Otherwise, we are increasing indices by increments of max_len if there remains a surplus for an index
            if i == 0:
                idx_1 = min(max_len, orig_idx_1)
                idx_2 = min(max_len, orig_idx_2)
            else:  # Process intermediate iterations by traversing idx_2 first followed by idx_1, to leave a final round
                if traversing_idx_2:
                    if ((idx_2_iter * max_len) + idx_2_overflow) == orig_idx_2 or idx_2 == orig_idx_2:  # Final iter.
                        idx_2 = idx_2_overflow  # Use overflow only to adjust idx_2
                        idx_1_iter += 1  # Increment idx_1 iteration counter
                        idx_2_iter = 1  # Reset idx_2 iteration counter
                        idx_2_traversed, just_finished_idx_2 = True, True  # Maintain idx_2's final walk state
                        traversing_idx_1, traversing_idx_2 = True, False  # Stop walking idx_2 and begin walking idx_1
                        return_now = True  # Ensure that we slice the final logits for idx_2 "before" moving to idx_1
                    else:  # Traverse idx_2 in step lengths of max_len
                        idx_2 += max_len
                        idx_2_iter += 1
                if traversing_idx_1 and not return_now:
                    if ((idx_1_iter * max_len) + idx_1_overflow) == orig_idx_1 or idx_1 == orig_idx_1:  # Final iter.
                        idx_1 = idx_1_overflow  # Use overflow only to adjust idx_1
                        idx_1_iter = 1  # Reset idx_1 iteration counter
                        idx_1_traversed, idx_2_traversed = True, False  # Maintain both indices final walk states
                        traversing_idx_1, traversing_idx_2 = False, True  # Stop walking idx_1 and begin walking idx_2
                    else:  # Traverse idx_1 in step lengths of max_len
                        idx_1 += max_len
                        traversing_idx_1, traversing_idx_2 = False, True

        # Process the most recent iteration on an index
        return_now = False if idx_2_traversed else return_now  # Force a return-like action in the intermediate walks
        sliced_logits = logits[i, :, :idx_1, :idx_2].unsqueeze(0)
        logits_with_padding_removed.append(sliced_logits)  # Add extraneous batch channel upon appending for insert()

        # Shift idx_2 back to support idx_1 for the next idx_1-columnwise "batch"
        if just_finished_idx_2:
            idx_2 = min(max_len, orig_idx_2)
            just_finished_idx_2 = False

    return logits_with_padding_removed


def insert_interact_tensor_logits(logits_list: List[torch.Tensor], interact_tensor: torch.Tensor, max_len: int):
    """Fill in an empty interaction tensor of an original interaction tensor's size, logits batch by logits batch."""
    # Initialize variables for tracking state
    orig_idx_1, idx_1_iter, orig_idx_2, idx_2_iter = interact_tensor.shape[2], 0, interact_tensor.shape[3], 1
    start_index_1, end_index_1, start_index_2, end_index_2 = 0, 0, 0, 0  # The indices we aim to derive dynamically
    idx_1_overflowed, idx_2_overflowed = orig_idx_1 > max_len, orig_idx_2 > max_len
    idx_1_overflow = (orig_idx_1 % max_len) if idx_1_overflowed else 0
    idx_2_overflow = (orig_idx_2 % max_len) if idx_2_overflowed else 0
    just_finished_idx_1, just_finished_idx_2, return_now = False, False, True
    traversing_idx_1, idx_1_traversed, traversing_idx_2, idx_2_traversed = False, False, True, False

    # Iterate over all incoming "unpadded" subsequenced logits
    for i, logits in enumerate(logits_list):
        new_end_index_1, new_end_index_2 = logits.shape[2:]  # Retrieve latest end indices
        if i == (len(logits_list) - 1):  # Handle the last iteration's indices uniquely
            # Add the surplus amount to the current multiple of max_len we are at
            if traversing_idx_1:
                start_index_increment_amount = max_len
                end_index_increment_amount = max_len if idx_1_overflow == 0 else idx_1_overflow
                start_index_1 += start_index_increment_amount
                end_index_1 += end_index_increment_amount
            if traversing_idx_2:
                start_index_increment_amount = max_len
                end_index_increment_amount = max_len if idx_2_overflow == 0 else idx_2_overflow
                start_index_2 += start_index_increment_amount
                end_index_2 += end_index_increment_amount
        else:  # Otherwise, use unique base indices to increment index counters
            if i == 0:  # For the first iteration, simply accept the incoming indices
                end_index_1, end_index_2 = new_end_index_1, new_end_index_2
                # Catch an edge case where we exhaust the columns' length (i.e., index 2) at the first iteration
                is_second_to_last_iter = i == len(logits_list) - 2
                if is_second_to_last_iter and orig_idx_1 > orig_idx_2:
                    traversing_idx_2, just_finished_idx_2, traversing_idx_1 = False, True, True
            else:  # Process intermediate iterations by traversing idx_2 first followed by idx_1, to leave a final round
                if traversing_idx_2:
                    if ((idx_2_iter * max_len) + idx_2_overflow) == orig_idx_2:  # Handle final iteration
                        start_index_2 += max_len
                        end_index_2 += max_len if idx_2_overflow == 0 else idx_2_overflow
                        idx_1_iter += 1  # Increment idx_1 iteration counter
                        idx_2_iter = 1  # Reset idx_2 iteration counter
                        idx_2_traversed, just_finished_idx_2 = True, True  # Maintain idx_2's final walk state
                        traversing_idx_1, traversing_idx_2 = True, False  # Stop walking idx_2 and begin walking idx_1
                        return_now = True  # Ensure that we slice the final logits for idx_2 "before" moving to idx_1
                    else:  # Traverse idx_2 in step lengths of max_len
                        start_index_2 += max_len
                        end_index_2 += max_len
                        idx_2_iter += 1
                if traversing_idx_1 and not return_now:
                    if ((idx_1_iter * max_len) + idx_1_overflow) == orig_idx_1:  # Handle final iteration
                        start_index_1 += max_len
                        end_index_1 += max_len if idx_1_overflow == 0 else idx_1_overflow
                        idx_1_iter = 1  # Reset idx_1 iteration counter
                        idx_1_traversed, idx_2_traversed = True, False  # Maintain both indices final walk states
                        traversing_idx_1, traversing_idx_2 = False, True  # Stop walking idx_1 and begin walking idx_2
                    else:  # Traverse idx_1 in step lengths of max_len
                        start_index_1 += max_len
                        end_index_1 += max_len
                        traversing_idx_1, traversing_idx_2 = False, True

        # Process the most recent iteration on an index
        return_now = False if idx_2_traversed else return_now  # Force a return-like action in an intermediate walk
        # Insert slices logits into their corresponding indices in the new interaction tensor being populated
        interact_tensor[:, :, start_index_1: end_index_1, start_index_2: end_index_2] = logits

        # Shift idx_2 back to support idx_1 for the next idx_1-columnwise "batch"
        if just_finished_idx_2:
            start_index_2, end_index_2 = 0, min(max_len, orig_idx_2)
            just_finished_idx_2 = False

    return interact_tensor


def substitute_missing_atoms(struct_df: pd.DataFrame, all_atom_struct_df: pd.DataFrame, atom_names: list):
    """Substitute missing backbone atoms in a Pandas DataFrame when they are found."""
    for ca_atom_idx, ca_atom in struct_df.iterrows():
        ca_atom_support_atoms = all_atom_struct_df[(all_atom_struct_df['model'] == ca_atom['model']) &
                                                   (all_atom_struct_df['chain'] == ca_atom['chain']) &
                                                   (all_atom_struct_df['residue'] == ca_atom['residue'])]

        # Check if at least one missing backbone atom was found
        num_atoms_missing = 4 - len(ca_atom_support_atoms)
        if num_atoms_missing > 0:
            # Replace all missing atoms sequentially
            for _ in range(num_atoms_missing):
                # Find which atom is missing
                if len(ca_atom_support_atoms[ca_atom_support_atoms['atom_name'] == 'N']) == 0:
                    missing_atom_key = 'N'
                    missing_atom_atom_id = ca_atom['aid'] - 1
                elif len(ca_atom_support_atoms[ca_atom_support_atoms['atom_name'] == 'C']) == 0:
                    missing_atom_key = 'C'
                    missing_atom_atom_id = ca_atom['aid'] + 1
                elif len(ca_atom_support_atoms[ca_atom_support_atoms['atom_name'] == 'O']) == 0:
                    missing_atom_key = 'O'
                    missing_atom_atom_id = ca_atom['aid'] + 2
                else:
                    raise NotImplementedError('Error: A missing atom was found, and it is not possible to process it.')

                # Choose a replacement for the missing atom
                available_atom_keys = set(atom_names) - {missing_atom_key}
                replacement_atom_name = available_atom_keys.pop()  # Choose the first available atom as the substitute
                replacement_atom = ca_atom_support_atoms[ca_atom_support_atoms['atom_name'] == replacement_atom_name]
                logging.info(f'Found a missing {missing_atom_key} atom for row number {ca_atom_idx} -'
                             f' Replaced it with {replacement_atom_name}')

                # Construct a new substitute atom via random coordinate shifts and value replacements
                coord_shift_range = -1, 1
                missing_atom_x_coord = replacement_atom['x'].values[0] + random.uniform(*coord_shift_range)
                missing_atom_y_coord = replacement_atom['y'].values[0] + random.uniform(*coord_shift_range)
                missing_atom_z_coord = replacement_atom['z'].values[0] + random.uniform(*coord_shift_range)

                # Assemble replacement atom's components and append the full atom
                replacement_atom = pd.DataFrame({
                    'pdb_name': replacement_atom['pdb_name'],
                    'model': replacement_atom['model'],
                    'chain': replacement_atom['chain'],
                    'residue': replacement_atom['residue'],
                    'resname': replacement_atom['resname'],
                    'ss_value': replacement_atom['ss_value'],
                    'rsa_value': replacement_atom['rsa_value'],
                    'rd_value': replacement_atom['rd_value'],
                    'avg_cx': replacement_atom['avg_cx'],
                    's_avg_cx': replacement_atom['s_avg_cx'],
                    's_ch_avg_cx': replacement_atom['s_ch_avg_cx'],
                    's_ch_s_avg_cx': replacement_atom['s_ch_s_avg_cx'],
                    'max_cx': replacement_atom['max_cx'],
                    'min_cx': replacement_atom['min_cx'],
                    'hsaac': replacement_atom['hsaac'],
                    'cn_value': replacement_atom['cn_value'],
                    'sequence_feats': replacement_atom['sequence_feats'],
                    'amide_norm_vec': replacement_atom['amide_norm_vec'],
                    'x': [missing_atom_x_coord],
                    'y': [missing_atom_y_coord],
                    'z': [missing_atom_z_coord],
                    'element': [missing_atom_key],
                    'atom_name': [missing_atom_key],
                    'aid': [missing_atom_atom_id]
                })
                all_atom_struct_df = all_atom_struct_df.append(replacement_atom, ignore_index=True)

                # Update support atoms collection after adding an atom
                ca_atom_support_atoms = all_atom_struct_df[(all_atom_struct_df['model'] == ca_atom['model']) &
                                                           (all_atom_struct_df['chain'] == ca_atom['chain']) &
                                                           (all_atom_struct_df['residue'] == ca_atom['residue'])]
    # Correct the ordering of each residue's atoms (inconsistency caused by arbitrarily appending new atoms)
    return all_atom_struct_df.sort_values(by='aid')


def convert_df_to_dgl_graph(df: pd.DataFrame, input_file: str, knn: int,
                            geo_nbrhd_size: int, self_loops: bool) -> dgl.DGLGraph:
    r""" Transform a given DataFrame of residues into a corresponding DGL graph.

    Parameters
    ----------
    df : pandas.DataFrame
    input_file : str
    knn : int
    geo_nbrhd_size : int
    self_loops : bool

    Returns
    -------
    :class:`dgl.DGLGraph`

        Graph structure, feature tensors for each node and edge.

...     node_feats = graph.ndata['f']
...     node_coords = graph.ndata['x']
...     edge_feats = graph.edata['f']

        - ``ndata['f']``: feature tensors of the nodes
                            Indices:
                            'node_pos_enc': 0,
                            'node_geo_feats_start': 1,
                            'node_geo_feats_end': 7,
                            'node_dips_plus_feats_start': 7,
                            'node_dips_plus_feats_end': 113,
        - ``ndata['x']:`` Cartesian coordinate tensors of the nodes
        - ``edata['f']``: feature tensors of the edges
                            Indices:
                            'edge_pos_enc': 0,
                            'edge_weights': 1,
                            'edge_dist_feats_start': 2,
                            'edge_dist_feats_end': 20,
                            'edge_dir_feats_start': 20,
                            'edge_dir_feats_end': 23,
                            'edge_orient_feats_start': 23,
                            'edge_orient_feats_end': 27,
                            'edge_amide_angles': 27
        - ``edata['src_nbr_e_ids']``: For edge e, integer IDs of incident edges connected to e's source node
        - ``edata['dst_nbr_e_ids']``: For edge e, integer IDs of incident edges connected to e's destination node
    """
    # Derive node features, with edges being defined via a k-nearest neighbors approach and a maximum distance threshold
    backbone_atom_names = ['N', 'CA', 'C', 'O']
    all_atom_struct_df = df[df['atom_name'].isin(backbone_atom_names)]  # Cache backbone atoms
    struct_df = df[df['atom_name'] == 'CA']
    graph, pairwise_dists, node_coords, node_feats = prot_df_to_dgl_graph_feats(
        struct_df,  # Only use CA atoms when constructing the initial graph
        FEAT_COLS,
        ALLOWABLE_FEATS,
        knn
    )

    # Retrieve src and destination node IDs
    srcs = graph.edges()[0]
    dsts = graph.edges()[1]

    # Remove self-loops (if requested)
    if not self_loops:
        graph = dgl.remove_self_loop(graph)
        srcs = graph.edges()[0]
        dsts = graph.edges()[1]

    # Manually add isolated nodes (i.e. those with no connected edges) to the graph
    if len(node_feats) > graph.number_of_nodes():
        num_of_isolated_nodes = len(node_feats) - graph.number_of_nodes()
        raise Exception(f'{num_of_isolated_nodes} isolated node(s) detected in {input_file}')

    """Derive geometric node and edge features for the input graph"""
    # Construct quaternions for each residue to capture a detailed geometric view of each residue-residue interaction
    edges = graph.edges()
    try:
        all_atom_coords = all_atom_struct_df[['x', 'y', 'z']].to_numpy().reshape(1, len(struct_df), 4, 3)
    except ValueError:
        # Coerce atom ID columns to be of numeric type for future sorting
        struct_df.loc[:, 'aid'] = pd.to_numeric(struct_df["aid"])
        all_atom_struct_df.loc[:, 'aid'] = pd.to_numeric(all_atom_struct_df["aid"])
        all_atom_struct_df = substitute_missing_atoms(struct_df, all_atom_struct_df, backbone_atom_names)
        # Retry the reshape procedure from above after replacing missing atoms
        all_atom_coords = all_atom_struct_df[['x', 'y', 'z']].to_numpy().reshape(1, len(struct_df), 4, 3)
    all_atom_coords = torch.from_numpy(all_atom_coords).to(dtype=torch.float32)
    # Mask NaN coordinates with zero
    coord_is_nan = torch.isnan(all_atom_coords)
    mask = torch.isfinite(torch.sum(all_atom_coords, (2, 3)))
    all_atom_coords[coord_is_nan] = 0.
    # Derive 'full' geometric features for each residue and features describing each residue-residue neighborhood
    num_rbf, features_type = 18, 'full'
    gen_geo_prot_feats = GeometricProteinFeatures(num_rbf=num_rbf, features_type=features_type)
    edges_transformed = edges[1].reshape(1, len(struct_df), knn)
    geo_node_feats, geo_edge_feats = gen_geo_prot_feats(all_atom_coords, pairwise_dists, edges_transformed, mask)
    geo_node_feats, geo_edge_feats = geo_node_feats.squeeze(), geo_edge_feats.squeeze()
    # Restructure derived geometric edge features to match DGL-expected format
    geo_edge_feats = geo_edge_feats.reshape(-1, geo_edge_feats.shape[2])
    # Parse out collected features for the 'full' feature set - other feature sets would need to be handled similarly
    full = features_type == 'full'
    if full:
        edge_dist_feats = geo_edge_feats[:, :num_rbf]
        edge_dir_feats = geo_edge_feats[:, num_rbf:num_rbf + 3]
        edge_orient_feats = geo_edge_feats[:, num_rbf + 3:]
    else:  # Default back to the full geometric feature set
        edge_dist_feats = geo_edge_feats[:, :num_rbf]
        edge_dir_feats = geo_edge_feats[:, num_rbf:num_rbf + 3]
        edge_orient_feats = geo_edge_feats[:, num_rbf + 3:]

    """Encode node features and labels in graph"""
    # Positional encoding for each node (used for Transformer-like GNNs)
    graph.ndata['f'] = min_max_normalize_tensor(graph.nodes()).reshape(-1, 1)  # [num_res_in_struct_df, 1]
    # Geometric node features derived above
    graph.ndata['f'] = torch.cat((graph.ndata['f'], geo_node_feats), dim=1)  # [num_res_in_struct_df, num_geo_node_feat]
    # One-hot features for each residue
    graph.ndata['f'] = torch.cat((graph.ndata['f'], node_feats), dim=1)  # [num_res_in_struct_df, num_node_feats]
    # Cartesian coordinates for each residue
    graph.ndata['x'] = node_coords  # [num_res_in_struct_df, 3]

    """Encode edge features and labels in graph"""
    # Positional encoding for each edge (used for sequentially-ordered inputs like proteins)
    graph.edata['f'] = torch.sin((graph.edges()[0] - graph.edges()[1]).float()).reshape(-1, 1)  # [num_edges, 1]
    # Normalized edge weights (according to Euclidean distance)
    edge_weights = min_max_normalize_tensor(torch.sum(node_coords[srcs] - node_coords[dsts] ** 2, 1)).reshape(-1, 1)
    graph.edata['f'] = torch.cat((graph.edata['f'], edge_weights), dim=1)  # [num_edges, 1]
    # Geometric edge features derived above
    graph.edata['f'] = torch.cat((graph.edata['f'], edge_dist_feats), dim=1)  # Distance: [num_edges, num_rbf] if full
    graph.edata['f'] = torch.cat((graph.edata['f'], edge_dir_feats), dim=1)  # Direction: [num_edges, 3] if full
    graph.edata['f'] = torch.cat((graph.edata['f'], edge_orient_feats), dim=1)  # Orientation: [num_edges, 4] if full

    # Angle between the two amide normal vectors for a pair of residues, for all edge-connected residue pairs
    plane1 = struct_df[['amide_norm_vec']].iloc[dsts]
    plane2 = struct_df[['amide_norm_vec']].iloc[srcs]
    plane1.columns = ['amide_norm_vec']
    plane2.columns = ['amide_norm_vec']
    plane1 = torch.from_numpy(np.stack(plane1['amide_norm_vec'].values).astype('float32'))
    plane2 = torch.from_numpy(np.stack(plane2['amide_norm_vec'].values).astype('float32'))
    angles = np.array([
        torch.acos(torch.dot(vec1, vec2) / (torch.linalg.norm(vec1) * torch.linalg.norm(vec2)))
        for vec1, vec2 in zip(plane1, plane2)
    ])
    # Ensure amide plane normal vector angles on each edge are zeroed out rather than being left as NaN (in some cases)
    np.nan_to_num(angles, copy=False, nan=0.0, posinf=None, neginf=None)
    amide_angles = torch.from_numpy(np.nan_to_num(
        min_max_normalize_tensor(torch.from_numpy(angles)).cpu().numpy(),
        copy=True, nan=0.0, posinf=None, neginf=None
    )).reshape(-1, 1)  # [num_edges, 1]
    graph.edata['f'] = torch.cat((graph.edata['f'], amide_angles), dim=1)  # Amide-amide angles: [num_edges, 1]

    """Define edge neighborhoods: For edge e, retrieve a given number of indices
     for edges directed towards the source and destination nodes of e, respectively."""
    src_node_in_edges, dst_node_in_edges = graph.in_edges(edges[0]), graph.in_edges(edges[1])
    src_node_in_edges = torch.cat((src_node_in_edges[0].reshape(-1, 1), src_node_in_edges[1].reshape(-1, 1)), dim=1)
    dst_node_in_edges = torch.cat((dst_node_in_edges[0].reshape(-1, 1), dst_node_in_edges[1].reshape(-1, 1)), dim=1)
    src_node_in_edges, dst_node_in_edges = src_node_in_edges.reshape(-1, knn, 2), dst_node_in_edges.reshape(-1, knn, 2)
    # Shuffle each KNN edge batch uniquely
    for batch_idx, knn_edge_batch in enumerate(src_node_in_edges):
        src_shuffled_edge_idx = torch.randperm(knn_edge_batch.size()[0])
        src_node_in_edges[batch_idx] = src_node_in_edges[batch_idx, src_shuffled_edge_idx]
    for batch_idx, knn_edge_batch in enumerate(dst_node_in_edges):
        dst_shuffled_edge_idx = torch.randperm(knn_edge_batch.size()[0])
        dst_node_in_edges[batch_idx] = dst_node_in_edges[batch_idx, dst_shuffled_edge_idx]
    src_node_in_edges = src_node_in_edges[:, :geo_nbrhd_size]
    dst_node_in_edges = dst_node_in_edges[:, :geo_nbrhd_size]
    # Derive edge IDs for randomly-selected neighboring edges
    src_e_ids = graph.edge_ids(torch.flatten(src_node_in_edges[:, :, 0]), torch.flatten(src_node_in_edges[:, :, 1]))
    dst_e_ids = graph.edge_ids(torch.flatten(dst_node_in_edges[:, :, 0]), torch.flatten(dst_node_in_edges[:, :, 1]))
    src_e_ids, dst_e_ids = src_e_ids.reshape(-1, geo_nbrhd_size), dst_e_ids.reshape(-1, geo_nbrhd_size)
    # Both the following edge features are of shape [num_edges, geo_nbrhd_size]
    graph.edata['src_nbr_e_ids'] = src_e_ids  # For edge e, store IDs of incident edges connected to e's src node
    graph.edata['dst_nbr_e_ids'] = dst_e_ids  # For edge e, store IDs of incident edges connected to e's dst node

    return graph


def build_examples_matrix_using_multi_indexing(array: np.ndarray, columns: List[str]):
    """Construct a new examples matrix using multi-indexing."""
    # Credit: https://stackoverflow.com/questions/46134827/how-to-recover-original-indices-for-a-flattened-numpy-array
    shape = array.shape
    index = pd.MultiIndex.from_product([range(s) for s in shape], names=columns)
    df = pd.DataFrame({'labels': array.flatten()}, index=index).reset_index()  # Flatten labels in a row-major fashion
    return df


def build_examples_tensor(df0: pd.DataFrame, df1: pd.DataFrame, pos_idx: pd.Series):
    """Construct the labels matrix for a given protein complex and mode (e.g. train, val, or test)."""
    # Derive inter-chain node-node (i.e., residue-residue) interaction matrix (Interacting = 1 and Non-Interacting = 0)
    labels = np.zeros((len(df0), len(df1)), dtype=np.int32)
    for idx in pos_idx:
        row0_row_num = df0.loc[[idx[0]]].index[0]
        row1_row_num = df1.loc[[idx[1]]].index[0]
        idx0 = df0.index.get_loc(row0_row_num)
        idx1 = df1.index.get_loc(row1_row_num)
        labels[idx0][idx1] = 1

    # Use multi-indexing to flatten and populate examples of shape (chain_0_res_id, chain_1_res_id, interaction_label)
    examples = build_examples_matrix_using_multi_indexing(array=labels, columns=['chain_0_res_id', 'chain_1_res_id'])

    # Return new examples tensor
    return torch.from_numpy(examples.to_numpy())  # Convert examples to NumPy array format and then to a PyTorch tensor


def create_input_dir_struct(input_dataset_dir: str, pdb_code: str):
    """Create directory structure for inputs."""
    dir_struct_create_cmd = f'mkdir -p {os.path.join(input_dataset_dir, "raw")}' \
                            f' {os.path.join(input_dataset_dir, "raw", pdb_code)}' \
                            f' {os.path.join(input_dataset_dir, "interim")}' \
                            f' {os.path.join(input_dataset_dir, "interim", "external_feats")}' \
                            f' {os.path.join(input_dataset_dir, "interim", "external_feats", "PSAIA")}' \
                            f' {os.path.join(input_dataset_dir, "interim", "external_feats", "PSAIA", "INPUT")}' \
                            f' {os.path.join(input_dataset_dir, "final")}' \
                            f' {os.path.join(input_dataset_dir, "final", "raw")}' \
                            f' {os.path.join(input_dataset_dir, "final", "processed")}'
    dir_struct_create_proc = subprocess.Popen(dir_struct_create_cmd.split(), stdout=subprocess.PIPE, cwd=os.getcwd())
    _, _ = dir_struct_create_proc.communicate()  # Wait until the directory structure creation cmd is finished


def copy_input_to_raw_dir(input_dataset_dir: str, pdb_filepath: str, pdb_code: str, chain_indic: str):
    """Make a copy of the input PDB file in the newly-created raw directory."""
    filename = db.get_pdb_code(pdb_filepath) + f'_{chain_indic}.pdb' \
        if chain_indic not in pdb_filepath else db.get_pdb_name(pdb_filepath)
    new_filepath = os.path.join(input_dataset_dir, "raw", pdb_code, filename)
    input_copy_cmd = f'cp {pdb_filepath} {os.path.join(input_dataset_dir, "raw", pdb_code, filename)}'
    input_copy_proc = subprocess.Popen(input_copy_cmd.split(), stdout=subprocess.PIPE, cwd=os.getcwd())
    _, _ = input_copy_proc.communicate()  # Wait until the input copy cmd is finished
    return new_filepath


def make_dataset(input_dataset_dir='datasets/Input/raw', output_dir='datasets/Input/interim', num_cpus=1,
                 neighbor_def='non_heavy_res', cutoff=6, source_type='input', unbound=True):
    """Make interim data set from raw data."""
    logger = logging.getLogger(__name__)
    logger.info('Making interim data set from raw data')

    parsed_dir = os.path.join(output_dir, 'parsed')
    pa.parse_all(input_dataset_dir, parsed_dir, num_cpus)

    complexes_dill_filepath = os.path.join(output_dir, 'complexes/complexes.dill')
    if os.path.exists(complexes_dill_filepath):
        os.remove(complexes_dill_filepath)  # Ensure that pairs are made everytime this function is called
    comp.complexes(parsed_dir, complexes_dill_filepath, source_type)
    complexes = comp.read_complexes(complexes_dill_filepath)
    pairs_dir = os.path.join(output_dir, 'pairs')
    get_neighbors = nb.build_get_neighbors(neighbor_def, cutoff)
    get_pairs = pair.build_get_pairs(neighbor_def, source_type, unbound, get_neighbors, False)
    pair.all_complex_to_pairs(complexes, source_type, get_pairs, pairs_dir, num_cpus)


def recover_any_missing_chain_ids(interim_dataset_dir: str, new_pdb_filepath: str,
                                  orig_pdb_filepath: str, pdb_code: str, chain_number: int):
    """Restore any missing chain IDs for the chain represented by the corresponding Pandas DataFrame."""
    orig_pdb_chain_id = '_'  # Default value for missing chain IDs
    new_pdb_code = db.get_pdb_code(new_pdb_filepath)
    orig_pdb_name = db.get_pdb_name(orig_pdb_filepath)
    new_pdb_obj = PandasPdb().read_pdb(new_pdb_filepath)
    unique_chain_ids = np.unique(new_pdb_obj.df['ATOM']['chain_id'].values)

    """Ascertain the chain ID corresponding to the original PDB file, using one of two available methods.
      Method 1: Used with datasets such as EVCoupling adopting .atom filename extensions (e.g., 4DI3C.atom)
      Method 2: Used with datasets such as DeepHomo adopting regular .pdb filename extensions (e.g., 2FNUA.pdb)"""
    if len(unique_chain_ids) == 1 and unique_chain_ids[0].strip() == '':  # Method 1: Try to use filename differences
        # No chain IDs were found, so we instead need to look to the original PDB filename to get the orig. chain ID
        pdb_code_diffs = difflib.ndiff(new_pdb_code, orig_pdb_name)
        for i, s in enumerate(pdb_code_diffs):
            if s[0] == '+':
                orig_pdb_chain_id = s[1:].strip()[0]
                break
    else:  # Method 2: Try to use unique chain IDs
        # Assume the first/second index is the first non-empty chain ID (e.g., 'A')
        orig_pdb_chain_id = unique_chain_ids[0] if (unique_chain_ids[0] != '') else unique_chain_ids[1]

    # Update version of the input PDB file copied to input_dataset_dir
    new_pdb_obj.df['ATOM']['chain_id'] = orig_pdb_chain_id
    new_pdb_obj.df['HETATM']['chain_id'] = orig_pdb_chain_id
    new_pdb_obj.df['ANISOU']['chain_id'] = orig_pdb_chain_id
    new_pdb_obj.df['OTHERS']['chain_id'] = orig_pdb_chain_id
    new_pdb_obj.to_pdb(new_pdb_filepath, records=None, gz=False, append_newline=True)

    # Update existing parsed chains to contain the newly-recovered chain ID
    parsed_dir = os.path.join(interim_dataset_dir, 'parsed', pdb_code)
    parsed_filenames = [
        os.path.join(parsed_dir, filename) for filename in os.listdir(parsed_dir) if new_pdb_code in filename
    ]
    parsed_filenames.sort()
    # Load in the existing Pair
    chain_df = pd.read_pickle(parsed_filenames[chain_number - 1])
    # Update the corresponding chain ID
    chain_df.chain = orig_pdb_chain_id
    # Save the updated Pair
    chain_df.to_pickle(parsed_filenames[chain_number - 1])

    # Update the existing Pair to contain the newly-recovered chain ID
    pair_dir = os.path.join(interim_dataset_dir, 'pairs', pdb_code)
    pair_filenames = [os.path.join(pair_dir, filename) for filename in os.listdir(pair_dir) if new_pdb_code in filename]
    pair_filenames.sort()
    # Load in the existing Pair
    with open(pair_filenames[0], 'rb') as f:
        pair = dill.load(f)
    # Update the corresponding chain ID
    pair.df0.chain = orig_pdb_chain_id if chain_number == 1 else pair.df0.chain
    pair.df1.chain = orig_pdb_chain_id if chain_number == 2 else pair.df1.chain
    # Save the updated Pair
    with open(pair_filenames[0], 'wb') as f:
        dill.dump(pair, f)


def generate_psaia_features(psaia_dir='~/Programs/PSAIA_1.0_source/bin/linux/psa',
                            psaia_config='datasets/builder/psaia_config_file_input.txt',
                            pdb_dataset='datasets/Input/raw', pkl_dataset='datasets/Input/interim/parsed',
                            pruned_dataset='datasets/Input/interim/parsed',
                            output_dir='datasets/Input/interim/external_feats', source_type='input'):
    """Generate PSAIA features from PDB files."""
    logger = logging.getLogger(__name__)
    logger.info(f'Generating PSAIA features from PDB files in {pkl_dataset}')

    # Generate protrusion indices
    con.map_all_protrusion_indices(psaia_dir, psaia_config, pdb_dataset, pkl_dataset,
                                   pruned_dataset, output_dir, source_type)


def generate_hhsuite_features(pkl_dataset='datasets/Input/interim/parsed',
                              pruned_dataset='datasets/Input/interim/parsed',
                              hhsuite_db='~/Data/Databases/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt',
                              output_dir='datasets/Input/interim/external_feats',
                              num_cpu_jobs=1,
                              num_cpus_per_job=8,
                              num_iter=2,
                              source_type='input'):
    """Generate PSAIA features from PDB files."""
    logger = logging.getLogger(__name__)
    logger.info(f'Generating profile HMM features from PDB files in {pkl_dataset}')

    # Generate protrusion indices
    con.map_all_profile_hmms(pkl_dataset, pruned_dataset, output_dir, hhsuite_db,
                             num_cpu_jobs, num_cpus_per_job, source_type, num_iter, 0, 1, write_file=True)


def launch_postprocessing_of_pruned_pairs(raw_pdb_dir='datasets/Input/raw',
                                          pruned_pairs_dir='datasets/Input/interim/pairs',
                                          external_feats_dir='datasets/Input/interim/external_feats',
                                          output_dir='datasets/Input/final/raw',
                                          num_cpus=1,
                                          source_type='input',
                                          pdb_code=''):
    """Run postprocess_pruned_pairs() on all provided complexes."""
    logger = logging.getLogger(__name__)
    logger.info(f'Starting postprocessing for all unprocessed pairs in {pruned_pairs_dir}')

    # Make sure the output_dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get work filenames
    logger.info(f'Looking for all pairs in {pruned_pairs_dir}')
    requested_filenames = db.get_structures_filenames(pruned_pairs_dir, extension='.dill')
    requested_filenames = [filename for filename in requested_filenames]
    requested_keys = [db.get_pdb_name(x) for x in requested_filenames]
    produced_filenames = db.get_structures_filenames(output_dir, extension='.dill')
    produced_keys = [db.get_pdb_name(x) for x in produced_filenames]
    work_keys = [key for key in requested_keys if key not in produced_keys]
    input_work_keys = [key for key in requested_keys]
    rscb_pruned_pair_ext = '.dill' if source_type.lower() in ['rcsb', 'evcoupling', 'casp_capri'] else ''
    work_filenames = [os.path.join(pruned_pairs_dir, db.get_pdb_code(work_key)[1:3], work_key + rscb_pruned_pair_ext)
                      for work_key in work_keys]
    input_work_filenames = [os.path.join(pruned_pairs_dir, db.get_pdb_code(work_key)[1:3],
                                         work_key + rscb_pruned_pair_ext) for work_key in input_work_keys
                            if pdb_code in work_key]
    logger.info(f'Found {len(work_keys)} work pair(s) in {pruned_pairs_dir}')

    # Remove any duplicate filenames
    work_filenames = list(set(work_filenames))

    # Get filenames in which our threads will store output
    output_filenames = []
    work_filenames_to_iter = input_work_filenames if source_type.lower() == 'input' else work_filenames
    for pdb_filename in work_filenames_to_iter:
        sub_dir = output_dir + '/' + db.get_pdb_code(pdb_filename)[1:3]
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        new_output_filename = sub_dir + '/' + db.get_pdb_name(pdb_filename) + ".dill" if \
            source_type in ['rcsb', 'evcoupling', 'casp_capri'] else \
            sub_dir + '/' + db.get_pdb_name(pdb_filename)
        output_filenames.append(new_output_filename)

    # Collect thread inputs
    inputs = [(raw_pdb_dir, external_feats_dir, i, o, source_type)
              for i, o in zip(work_filenames, output_filenames)]
    par.submit_jobs(postprocess_pruned_pairs, inputs, num_cpus)
    return output_filenames


def impute_missing_feature_values(output_dir='datasets/Input/final/raw',
                                  impute_atom_features=False,
                                  advanced_logging=False,
                                  num_cpus=1):
    """Impute missing feature values."""
    logger = logging.getLogger(__name__)
    logger.info('Imputing missing feature values for given inputs')

    # Make sure the output_dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Collect thread inputs
    inputs = [(pair_filename.as_posix(), pair_filename.as_posix(), impute_atom_features, advanced_logging)
              for pair_filename in Path(output_dir).rglob('*.dill')]
    # Without impute_atom_features set to True, non-CA atoms will be filtered out after writing updated pairs
    par.submit_jobs(impute_postprocessed_missing_feature_values, inputs, num_cpus)


def convert_input_pdb_files_to_pair(left_pdb_filepath: str, right_pdb_filepath: str, input_dataset_dir: str,
                                    psaia_dir: str, psaia_config: str, hhsuite_db: str):
    """Convert a pair of input PDB files into two DeepInteract feature set-filled DataFrames."""
    # Ascertain the input PDB files' shared PDB code
    pdb_code = db.get_pdb_group(list(ca.get_complex_pdb_codes([left_pdb_filepath, right_pdb_filepath]))[0])
    # Iteratively execute the PDB file feature generation process
    create_input_dir_struct(input_dataset_dir, pdb_code)
    new_l_u_filepath = copy_input_to_raw_dir(input_dataset_dir, left_pdb_filepath, pdb_code, 'l_u')
    new_r_u_filepath = copy_input_to_raw_dir(input_dataset_dir, right_pdb_filepath, pdb_code, 'r_u')
    make_dataset(os.path.join(input_dataset_dir, 'raw'), os.path.join(input_dataset_dir, 'interim'))
    recover_any_missing_chain_ids(os.path.join(input_dataset_dir, 'interim'),
                                  new_l_u_filepath, left_pdb_filepath, pdb_code, 1)
    recover_any_missing_chain_ids(os.path.join(input_dataset_dir, 'interim'),
                                  new_r_u_filepath, right_pdb_filepath, pdb_code, 2)
    generate_psaia_features(psaia_dir=psaia_dir,
                            psaia_config=psaia_config,
                            pdb_dataset=os.path.join(input_dataset_dir, 'raw'),
                            pkl_dataset=os.path.join(input_dataset_dir, 'interim', 'parsed'),
                            pruned_dataset=os.path.join(input_dataset_dir, 'interim', 'parsed'),
                            output_dir=os.path.join(input_dataset_dir, 'interim', 'external_feats'))
    # Allow the user to specify an alternative to the BFD for searches
    generate_hhsuite_features(pkl_dataset=os.path.join(input_dataset_dir, 'interim', 'parsed'),
                              pruned_dataset=os.path.join(input_dataset_dir, 'interim', 'parsed'),
                              hhsuite_db=hhsuite_db,
                              output_dir=os.path.join(input_dataset_dir, 'interim', 'external_feats'))
    # Postprocess any pruned pairs that have not already been postprocessed
    pair_filepaths = launch_postprocessing_of_pruned_pairs(
        raw_pdb_dir=os.path.join(input_dataset_dir, 'raw'),
        pruned_pairs_dir=os.path.join(input_dataset_dir, 'interim', 'pairs'),
        external_feats_dir=os.path.join(input_dataset_dir, 'interim', 'external_feats'),
        output_dir=os.path.join(input_dataset_dir, 'final', 'raw'),
        pdb_code=pdb_code
    )
    if len(pair_filepaths) > 0:
        # Retrieve the filepath of the single input pair produced in this case
        pair_filepath = pair_filepaths[0]
    else:
        # Manually construct the already-postprocessed input pair's filepath since no pairs needed postprocessing
        output_dir = os.path.join(input_dataset_dir, 'final', 'raw')
        produced_filenames = db.get_structures_filenames(output_dir, extension='.dill')
        produced_keys = [db.get_pdb_name(x) for x in produced_filenames
                         if db.get_pdb_code(x).upper() in db.get_pdb_code(new_l_u_filepath).upper()]
        pair_filepath = [os.path.join(output_dir, db.get_pdb_code(key)[1:3], key)
                         for key in produced_keys][0]
    # Impute any missing feature values in the postprocessed input pairs
    impute_missing_feature_values(output_dir=os.path.join(input_dataset_dir, 'final', 'raw'))
    # Load preprocessed pair
    with open(pair_filepath, 'rb') as f:
        input_pair = dill.load(f)
    return input_pair


def process_pdb_into_graph(left_pdb_filepath: str, right_pdb_filepath: str, input_dataset_dir: str, psaia_dir: str,
                           psaia_config: str, hhsuite_db: str, knn: int, geo_nbrhd_size: int, self_loops: bool):
    """Process PDB file into a DGLGraph containing DeepInteract feature set."""
    input_pair = convert_input_pdb_files_to_pair(left_pdb_filepath, right_pdb_filepath,
                                                 input_dataset_dir, psaia_dir, psaia_config, hhsuite_db)
    # Convert the input DataFrame into its DGLGraph representations, using all atoms to generate geometric features
    graph1 = convert_df_to_dgl_graph(input_pair.df0, left_pdb_filepath, knn, geo_nbrhd_size, self_loops)
    graph2 = convert_df_to_dgl_graph(input_pair.df1, right_pdb_filepath, knn, geo_nbrhd_size, self_loops)
    return graph1, graph2


def all_equal(items):
    """Return True iff all items are equal."""
    first = items[0]
    return all(x == first for x in items)


def compute_match(aligned_sequences):
    """Compute the percent identity between two aligned sequences."""
    match_count = sum(1 for chars in zip(*aligned_sequences) if all_equal(chars))
    total = len(aligned_sequences[0])
    # mismatch_count = total - match_count
    percent_identity = match_count / total
    return percent_identity


def calculate_percent_identity(seq1: str, seq2: str):
    """Determine the percent identity for a pair of sequences by first aligning them."""
    alignment = pairwise2.align.globalxx(seq1, seq2)
    percent_identity = 0
    for align in alignment:
        seqA, seqB = SeqRecord(Seq(align.seqA)), SeqRecord(Seq(align.seqB))
        aligned_sequences = MultipleSeqAlignment([seqA, seqB], annotations={"tool": "demo"})
        percent_identity += compute_match(aligned_sequences)
    percent_identity /= len(alignment)  # Average percent identity across all alignments
    return percent_identity


def check_percent_identity(input_filename: str, compare_filenames: List[str], percent_identity_threshold: int, logger):
    """Determine the identity percentage for each of the four possible sequence pairs for a given complex."""
    with open(input_filename, 'rb') as i_f:
        input_complex = dill.load(i_f)
    for compare_filename in compare_filenames:
        with open(compare_filename, 'rb') as c_f:
            compare_complex = dill.load(c_f)
        l_b_l_b_per_id = calculate_percent_identity(input_complex.sequences['l_b'], compare_complex.sequences['l_b'])
        l_b_r_b_per_id = calculate_percent_identity(input_complex.sequences['l_b'], compare_complex.sequences['r_b'])
        r_b_l_b_per_id = calculate_percent_identity(input_complex.sequences['r_b'], compare_complex.sequences['l_b'])
        r_b_r_b_per_id = calculate_percent_identity(input_complex.sequences['r_b'], compare_complex.sequences['r_b'])
        # Report percent identity exceeding threshold (if applicable)
        if l_b_l_b_per_id > percent_identity_threshold:
            logger.info(f'L_b chain in {input_filename} {l_b_l_b_per_id - percent_identity_threshold}'
                        f' above percent identity threshold w.r.t to comparison complexes\' l_b chains')
            return
        elif l_b_r_b_per_id > percent_identity_threshold:
            logger.info(f'L_b chain in {input_filename} {l_b_r_b_per_id - percent_identity_threshold}'
                        f' above percent identity threshold w.r.t to comparison complexes\' r_b chains')
            return
        elif r_b_l_b_per_id > percent_identity_threshold:
            logger.info(f'R_b chain in {input_filename} {r_b_l_b_per_id - percent_identity_threshold}'
                        f' above percent identity threshold w.r.t to comparison complexes\' l_b chains')
            return
        elif r_b_r_b_per_id > percent_identity_threshold:
            logger.info(f'R_b chain in {input_filename} {r_b_r_b_per_id - percent_identity_threshold}'
                        f' above percent identity threshold w.r.t to comparison complexes\' r_b chains')
            return
    logger.info(f'All chains in {input_filename} are below percent identity threshold'
                f' w.r.t all chains in comparison files')


def process_complex_into_dict(raw_filepath: str, processed_filepath: str, knn: int,
                              geo_nbrhd_size: int, self_loops: bool, check_sequence: bool):
    """Process protein complex into a dictionary representing both structures and ready for a given mode (e.g. val)."""
    # Retrieve specified DIPS+ (RCSB) complex
    bound_complex: pa.Pair = pd.read_pickle(raw_filepath)

    # Isolate CA atoms in each structure's DataFrame
    df0 = bound_complex.df0[bound_complex.df0['atom_name'] == 'CA']
    df1 = bound_complex.df1[bound_complex.df1['atom_name'] == 'CA']

    # Ensure that the sequence of each DataFrame's residues matches its original FASTA sequence, character-by-character
    if check_sequence:
        df0_sequence = bound_complex.sequences['l_b']
        for i, (df_res_name, orig_res) in enumerate(zip(df0['resname'].values, df0_sequence)):
            if D3TO1[df_res_name] != orig_res:
                raise Exception(f'DataFrame 0 residue sequence does not match original FASTA sequence at position {i}')
        df1_sequence = bound_complex.sequences['r_b']
        for i, (df_res_name, orig_res) in enumerate(zip(df1['resname'].values, df1_sequence)):
            if D3TO1[df_res_name] != orig_res:
                raise Exception(f'DataFrame 1 residue sequence does not match original FASTA sequence at position {i}')

    # Convert each DataFrame into its DGLGraph representation, using all atoms to generate geometric features
    all_atom_df0, all_atom_df1 = bound_complex.df0, bound_complex.df1
    graph1 = convert_df_to_dgl_graph(all_atom_df0, raw_filepath, knn, geo_nbrhd_size, self_loops)
    graph2 = convert_df_to_dgl_graph(all_atom_df1, raw_filepath, knn, geo_nbrhd_size, self_loops)

    # Assemble the examples (containing labels) for the complex
    examples = build_examples_tensor(df0, df1, bound_complex.pos_idx)

    # Represent each complex as a pair of DGL graphs stored in a dictionary
    processed_complex = {
        'graph1': graph1,
        'graph2': graph2,
        'examples': examples,
        'complex': bound_complex.complex
    }

    # Write into processed_filepath
    processed_file_dir = os.path.join(*processed_filepath.split(os.sep)[: -1])
    os.makedirs(processed_file_dir, exist_ok=True)
    with open(processed_filepath, 'wb') as f:
        pickle.dump(processed_complex, f)


def zero_out_complex_features(cmplx: dict):
    """Zero-out the input features for a given protein complex dictionary (for an input-independent baseline)."""
    cmplx['graph1'].ndata['f'] = torch.zeros_like(cmplx['graph1'].ndata['f'])
    cmplx['graph1'].edata['f'] = torch.zeros_like(cmplx['graph1'].edata['f'])
    cmplx['graph2'].ndata['f'] = torch.zeros_like(cmplx['graph2'].ndata['f'])
    cmplx['graph2'].edata['f'] = torch.zeros_like(cmplx['graph2'].edata['f'])
    return cmplx


def calculate_top_k_prec(sorted_pred_indices: torch.Tensor, labels: torch.Tensor, k: int):
    """Calculate the top-k interaction precision."""
    num_interactions_to_score = k
    selected_pred_indices = sorted_pred_indices[:num_interactions_to_score]
    true_labels = labels[selected_pred_indices]
    num_correct = torch.sum(true_labels).item()
    prec = num_correct / num_interactions_to_score
    return prec


def extract_object(obj: any):
    """If incoming object is of type torch.Tensor, convert it to a NumPy array. If it is a scalar, simply return it."""
    return obj.cpu().numpy() if type(obj) == torch.Tensor else obj


def collect_args():
    """Collect all arguments required for training/testing."""
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # -----------------
    # Model arguments
    # -----------------
    parser.add_argument('--model_name', type=str, default='GINI', help='Default option is GINI')
    parser.add_argument('--num_gnn_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--num_interact_layers', type=int, default=14, help='Number of layers in interaction module')
    parser.add_argument('--metric_to_track', type=str, default='val_ce', help='Scheduling and early stop')

    # -----------------
    # Data arguments
    # -----------------
    parser.add_argument('--knn', type=int, default=20, help='Number of nearest neighbor edges for each node')
    parser.add_argument('--self_loops', action='store_true', dest='self_loops', help='Allow node self-loops')
    parser.add_argument('--no_self_loops', action='store_false', dest='self_loops', help='Disable self-loops')
    parser.add_argument('--pn_ratio', type=float, default=0.1,
                        help='Positive-negative class ratio to instate during training with DIPS-Plus')
    parser.add_argument('--dips_percent_to_use', type=float, default=1.00,
                        help='Fraction of DIPS-Plus dataset splits to use')
    parser.add_argument('--dips_data_dir', type=str, default='datasets/DIPS/final/raw', help='Path to DIPS')
    parser.add_argument('--casp_capri_data_dir', type=str, default='datasets/CASP_CAPRI/final/raw', help='CAPRI path')
    parser.add_argument('--casp_capri_percent_to_use', type=float, default=1.0, help='Fraction of CASP-CAPRI to use')
    parser.add_argument('--process_complexes', action='store_true', dest='process_complexes',
                        help='Check if all complexes for a dataset are processed and, if not, process those remaining')
    parser.add_argument('--testing_with_casp_capri', action='store_true', dest='testing_with_casp_capri',
                        help='Test on the 13th and 14th CASP-CAPRI\'s dataset of homo and heterodimers')
    parser.add_argument('--input_dataset_dir', type=str, default='datasets/Input',
                        help='Path to directory in which to generate features and outputs for the given inputs')
    parser.add_argument('--psaia_dir', type=str, default='~/Programs/PSAIA_1.0_source/bin/linux/psa',
                        help='Path to locally-compiled copy of PSAIA (i.e., to PSA, one of its CLIs)')
    parser.add_argument('--psaia_config', type=str, default='datasets/builder/psaia_config_file_input.txt',
                        help='Path to input config file for PSAIA')
    parser.add_argument('--hhsuite_db', type=str,
                        default='~/Data/Databases/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt',
                        help='Path to downloaded and extracted HH-suite3-compatible database (e.g., BFD or Uniclust30)')

    # -----------------
    # Logging arguments
    # -----------------
    parser.add_argument('--logger_name', type=str, default='TensorBoard', help='Which logger to use for experiments')
    parser.add_argument('--experiment_name', type=str, default=None, help='Logger experiment name')
    parser.add_argument('--project_name', type=str, default='DeepInteract', help='Logger project name')
    parser.add_argument('--entity', type=str, default='bml-lab', help='Logger entity (i.e. team) name')
    parser.add_argument('--run_id', type=str, default='', help='Logger run ID')
    parser.add_argument('--offline', action='store_true', dest='offline', help='Whether to log locally or remotely')
    parser.add_argument('--online', action='store_false', dest='offline', help='Whether to log locally or remotely')
    parser.add_argument('--tb_log_dir', type=str, default='tb_logs', help='Where to store TensorBoard log files')
    parser.set_defaults(offline=False)  # Default to using online logging mode

    # -----------------
    # Seed arguments
    # -----------------
    parser.add_argument('--seed', type=int, default=None, help='Seed for NumPy and PyTorch')

    # -----------------
    # Meta-arguments
    # -----------------
    parser.add_argument('--batch_size', type=int, default=1, help='Number of samples included in each data batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Decay rate of optimizer weight')
    parser.add_argument('--num_epochs', type=int, default=50, help='Maximum number of epochs to run for training')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout (forget) rate')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait until early stopping')
    parser.add_argument('--pad', action='store_true', dest='pad', help='Whether to zero pad interaction tensors')

    # -----------------
    # Miscellaneous
    # -----------------
    parser.add_argument('--max_hours', type=int, default=1, help='Maximum number of hours to allot for training')
    parser.add_argument('--max_minutes', type=int, default=55, help='Maximum number of minutes to allot for training')
    parser.add_argument('--multi_gpu_backend', type=str, default='ddp', help='Multi-GPU backend for training')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use (e.g. -1 = all available GPUs)')
    parser.add_argument('--auto_choose_gpus', action='store_true', dest='auto_choose_gpus', help='Auto-select GPUs')
    parser.add_argument('--num_compute_nodes', type=int, default=1, help='Number of compute nodes to use')
    parser.add_argument('--gpu_precision', type=int, default=32, help='Bit size used during training (e.g. 16-bit)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU threads for loading data')
    parser.add_argument('--profiler_method', type=str, default=None, help='PL profiler to use (e.g. simple)')
    parser.add_argument('--ckpt_dir', type=str, default=f'{os.path.join(os.getcwd(), "checkpoints")}',
                        help='Directory in which to save checkpoints')
    parser.add_argument('--ckpt_name', type=str, default='', help='Filename of best checkpoint')
    parser.add_argument('--min_delta', type=float, default=5e-6, help='Minimum percentage of change required to'
                                                                      ' "metric_to_track" before early stopping'
                                                                      ' after surpassing patience')
    parser.add_argument('--accum_grad_batches', type=int, default=1, help='Norm over which to clip gradients')
    parser.add_argument('--grad_clip_val', type=float, default=0.5, help='Norm over which to clip gradients')
    parser.add_argument('--grad_clip_algo', type=str, default='norm', help='Algorithm with which to clip gradients')
    parser.add_argument('--stc_weight_avg', action='store_true', dest='stc_weight_avg', help='Smooth loss landscape')
    parser.add_argument('--find_lr', action='store_true', dest='find_lr', help='Find an optimal learning rate a priori')
    parser.add_argument('--input_indep', action='store_true', dest='input_indep', help='Whether to zero input for test')

    return parser


def process_args(args):
    """Process all arguments required for training/testing."""
    # ---------------------------------------
    # Seed fixing for random numbers
    # ---------------------------------------
    if not args.seed:
        args.seed = 42  # np.random.randint(100000)
    logging.info(f'Seeding everything with random seed {args.seed}')
    pl.seed_everything(args.seed)
    dgl.seed(args.seed)

    return args


def construct_pl_logger(args):
    """Return a specific Logger instance requested by the user."""
    if args.logger_name.lower() == 'wandb':
        return construct_wandb_pl_logger(args)
    else:  # Default to using TensorBoard
        return construct_tensorboard_pl_logger(args)


def construct_wandb_pl_logger(args):
    """Return an instance of WandbLogger with corresponding project and name strings."""
    return WandbLogger(name=args.experiment_name,
                       offline=args.offline,
                       project=args.project_name,
                       log_model=True,
                       entity=args.entity)


def construct_tensorboard_pl_logger(args):
    """Return an instance of TensorBoardLogger with corresponding project and experiment name strings."""
    return TensorBoardLogger(save_dir=args.tb_log_dir,
                             name=args.experiment_name)
