import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import CaPPBuilder

# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------

# Dataset-global node count limits to restrict computational learning complexity
ATOM_COUNT_LIMIT = 2048  # Default atom count filter for DIPS-Plus when encoding complexes at an atom-based level
RESIDUE_COUNT_LIMIT = 256  # Default residue count limit for DIPS-Plus (empirically determined for smoother training)
NODE_COUNT_LIMIT = 2304  # An upper-bound on the node count limit for Geometric Transformers - equal to 9-sized batch
KNN = 20  # Default number of nearest neighbors to query for during graph message passing

# Complexes excluded due to their non-trivially corrected backbone atom structures (for geometric feature generation)
EXCLUDED_COMPLEX_PAIR_FILENAMES = ['../DIPS/final/raw/ep/1epb.pdb1_0.dill']

# Postprocessing logger dictionary
DEFAULT_DATASET_STATISTICS = dict(num_of_processed_complexes=0, num_of_df0_residues=0, num_of_df1_residues=0,
                                  num_of_df0_interface_residues=0, num_of_df1_interface_residues=0,
                                  num_of_pos_res_pairs=0, num_of_neg_res_pairs=0, num_of_res_pairs=0,
                                  num_of_valid_df0_ss_values=0, num_of_valid_df1_ss_values=0,
                                  num_of_valid_df0_rsa_values=0, num_of_valid_df1_rsa_values=0,
                                  num_of_valid_df0_rd_values=0, num_of_valid_df1_rd_values=0,
                                  num_of_valid_df0_protrusion_indices=0, num_of_valid_df1_protrusion_indices=0,
                                  num_of_valid_df0_hsaacs=0, num_of_valid_df1_hsaacs=0,
                                  num_of_valid_df0_cn_values=0, num_of_valid_df1_cn_values=0,
                                  num_of_valid_df0_sequence_feats=0, num_of_valid_df1_sequence_feats=0,
                                  num_of_valid_df0_amide_normal_vecs=0, num_of_valid_df1_amide_normal_vecs=0)

# Parsing utilities for PDB files (i.e. relevant for sequence and structure analysis)
PDB_PARSER = PDBParser()
CA_PP_BUILDER = CaPPBuilder()

# PSAIA features to encode as DataFrame columns
PSAIA_COLUMNS = ['avg_cx', 's_avg_cx', 's_ch_avg_cx', 's_ch_s_avg_cx', 'max_cx', 'min_cx']

# Constants for calculating half sphere exposure statistics
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY-'
AMINO_ACID_IDX = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS))))

# Default fill values for missing features
HSAAC_DIM = 42  # We have 2 + (2 * 20) HSAAC values from the two instances of the unknown residue symbol '-'
DEFAULT_MISSING_FEAT_VALUE = np.nan
DEFAULT_MISSING_SS = '-'
DEFAULT_MISSING_RSA = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_RD = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_PROTRUSION_INDEX = [DEFAULT_MISSING_FEAT_VALUE for _ in range(6)]
DEFAULT_MISSING_HSAAC = [DEFAULT_MISSING_FEAT_VALUE for _ in range(HSAAC_DIM)]
DEFAULT_MISSING_CN = DEFAULT_MISSING_FEAT_VALUE
DEFAULT_MISSING_SEQUENCE_FEATS = np.array([DEFAULT_MISSING_FEAT_VALUE for _ in range(27)])
DEFAULT_MISSING_NORM_VEC = [DEFAULT_MISSING_FEAT_VALUE for _ in range(3)]

# Default number of NaN values allowed in a specific column before imputing missing features of the column with zero
NUM_ALLOWABLE_NANS = 5

# Dict for converting three letter codes to one letter codes
D3TO1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

# Features to be one-hot encoded during graph processing and what their values could be
FEAT_COLS = [
    'resname',  # [7:27]
    'ss_value',  # [27:35]
    'rsa_value',  # [35:36]
    'rd_value'  # [36:37]
]
FEAT_COLS.extend(
    PSAIA_COLUMNS +  # [37:43]
    ['hsaac',  # [43:85]
     'cn_value',  # [85:86]
     'sequence_feats',  # [86:113]
     'amide_norm_vec',  # [Stored separately]
     # 'element'  # For atom-level learning only
     ])

ALLOWABLE_FEATS = [
    ["TRP", "PHE", "LYS", "PRO", "ASP", "ALA", "ARG", "CYS", "VAL", "THR",
     "GLY", "SER", "HIS", "LEU", "GLU", "TYR", "ILE", "ASN", "MET", "GLN"],
    ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-'],  # Populated 1D list means restrict column feature values by list values
    [],  # Empty list means take scalar value as is
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [[]],  # Doubly-nested, empty list means take first-level nested list as is
    [],
    [[]],
    [[]],
    # ['C', 'O', 'N', 'S']  # For atom-level learning only
]

# A schematic of which tensor indices correspond to which node and edge features
FEATURE_INDICES = {
    # Node feature indices
    'node_pos_enc': 0,
    'node_geo_feats_start': 1,
    'node_geo_feats_end': 7,
    'node_dips_plus_feats_start': 7,
    'node_dips_plus_feats_end': 113,
    # Edge feature indices
    'edge_pos_enc': 0,
    'edge_weights': 1,
    'edge_dist_feats_start': 2,
    'edge_dist_feats_end': 20,
    'edge_dir_feats_start': 20,
    'edge_dir_feats_end': 23,
    'edge_orient_feats_start': 23,
    'edge_orient_feats_end': 27,
    'edge_amide_angles': 27
}
