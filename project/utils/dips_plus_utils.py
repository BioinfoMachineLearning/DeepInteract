import logging
import os
import pickle
from pathlib import Path
from typing import List

import atom3.database as db
import atom3.pair as pa
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.PDB import Selection
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, DSSP
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.vectors import Vector
from Bio.SCOP.Raf import protein_letters_3to1
from scipy import spatial
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from project.utils.deepinteract_constants import NUM_ALLOWABLE_NANS, DEFAULT_DATASET_STATISTICS, PSAIA_COLUMNS, \
    DEFAULT_MISSING_NORM_VEC, DEFAULT_MISSING_SEQUENCE_FEATS, DEFAULT_MISSING_CN, DEFAULT_MISSING_HSAAC, \
    DEFAULT_MISSING_PROTRUSION_INDEX, DEFAULT_MISSING_RD, DEFAULT_MISSING_RSA, DEFAULT_MISSING_SS, AMINO_ACIDS, \
    AMINO_ACID_IDX, HSAAC_DIM, ATOM_COUNT_LIMIT, PDB_PARSER


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code derived from PAIRpred (https://combi.cs.colostate.edu/supplements/pairpred/):
# -------------------------------------------------------------------------------------------------------------------------------------
def get_coords(residues):
    """
    Get atom coordinates given a list of biopython residues
    """
    Coords = []
    for (idx, r) in enumerate(residues):
        v = [ak.get_coord() for ak in r.get_list()]
        Coords.append(v)
    return Coords


def get_res_letter(residue):
    """
    Get the letter code for a biopython residue object
    """
    r2name = residue.get_resname()
    if r2name in protein_letters_3to1:
        scode = protein_letters_3to1[r2name]
    else:
        scode = '-'
    return scode


def get_side_chain_vector(residue):
    """
    Find the average of the unit vectors to different atoms in the side chain
    from the c-alpha atom. For glycine the average of the N-Ca and C-Ca is
    used.
    Returns (C-alpha coordinate vector, side chain unit vector) for residue r
    """
    u = None
    gly = 0
    if is_aa(residue) and residue.has_id('CA'):
        ca = residue['CA'].get_coord()
        dv = np.array([ak.get_coord() for ak in residue.get_unpacked_list()[4:]])
        if len(dv) < 1:
            if residue.has_id('N') and residue.has_id('C'):
                dv = [residue['C'].get_coord(), residue['N'].get_coord()]
                dv = np.array(dv)
                gly = 1
            else:
                return None
        dv = dv - ca
        if gly:
            dv = -dv
        n = np.sum(np.abs(dv) ** 2, axis=-1) ** (1. / 2)
        v = dv / n[:, np.newaxis]
        v = v.mean(axis=0)
        u = (Vector(ca), Vector(v))
    return u


def get_similarity_matrix(coords, sg=2.0, thr=1e-3):
    """
    Instantiates the distance based similarity matrix (S). S is a tuple of
    lists (I,V). |I|=|V|=|R|. Each I[r] refers to the indices
    of residues in R which are "close" to the residue indexed by r in R, and V[r]
    contains a list of the similarity scores for the corresponding residues.
    The distance between two residues is defined to be the minimum distance of
    any of their atoms. The similarity score is evaluated as
        s = exp(-d^2/(2*sg^2))
    This ensures that the range of similarity values is 0-1. sg (sigma)
    determines the extent of the neighborhood.
    Two residues are defined to be close to one another if their similarity
    score is greater than a threshold (thr).
    Residues (or ligands) for which DSSP features are not available are not
    included in the distance calculations.
    """
    sg = 2 * (sg ** 2)
    I = [[] for k in range(len(coords))]
    V = [[] for k in range(len(coords))]
    for i in range(len(coords)):
        for j in range(i, len(coords)):
            d = spatial.distance.cdist(coords[i], coords[j]).min()
            s = np.exp(-(d ** 2) / sg)
            if s > thr:  # and not np.isnan(self.Phi[i]) and not np.isnan(self.Phi[j])
                I[i].append(j)
                V[i].append(s)
                if i != j:
                    I[j].append(i)
                    V[j].append(s)
    similarity_matrix = (I, V)
    coordinate_numbers = np.array([len(a) for a in similarity_matrix[0]])
    return similarity_matrix, coordinate_numbers


def get_hsacc(residues, similarity_matrix, raw_pdb_filename):
    """
    Compute the Half sphere exposure statistics
    The up direction is defined as the direction of the side chain and is
    calculated by taking average of the unit vectors to different side chain
    atoms from the C-alpha atom
    Anything within the up half sphere is counted as up and the rest as
    down
    """
    N = len(residues)
    Na = len(AMINO_ACIDS)
    UN = np.zeros(N)
    DN = np.zeros(N)
    UC = np.zeros((Na, N))
    DC = np.zeros((Na, N))
    for (i, r) in enumerate(residues):
        u = get_side_chain_vector(r)
        if u is None:
            UN[i] = np.nan
            DN[i] = np.nan
            UC[:, i] = np.nan
            DC[:, i] = np.nan
            logging.info(f'No side chain vector found for residue #{i} in PDB file {raw_pdb_filename}')
        else:
            idx = AMINO_ACID_IDX[get_res_letter(r)]
            UC[idx, i] = UC[idx, i] + 1
            DC[idx, i] = DC[idx, i] + 1
            n = similarity_matrix[0][i]
            for j in n:
                r2 = residues[j]
                if is_aa(r2) and r2.has_id('CA'):
                    v2 = r2['CA'].get_vector()
                    scode = get_res_letter(r2)
                    idx = AMINO_ACID_IDX[scode]
                    angle = u[1].angle((v2 - u[0]))
                    if angle < np.pi / 2.0:
                        UN[i] = UN[i] + 1
                        UC[idx, i] = UC[idx, i] + 1
                    else:
                        DN[i] = DN[i] + 1
                        DC[idx, i] = DC[idx, i] + 1
    UC = UC / (1.0 + UN)
    DC = DC / (1.0 + DN)
    return UC, DC


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated in DIPS-Plus (https://github.com/BioinfoMachineLearning/DIPS-Plus):
# -------------------------------------------------------------------------------------------------------------------------------------

def find_fasta_sequences_for_pdb_file(sequences: dict, pdb_filename: str, external_feats_dir: str,
                                      struct_idx: int, is_rcsb_complex: bool, original_pair: pd.DataFrame):
    """Extract from previously-generated FASTA files the residue sequences for a given PDB file."""
    # Extract required paths, file lists, and sequences
    pdb_code = db.get_pdb_code(pdb_filename)[1:3]
    pdb_full_name = db.get_pdb_name(pdb_filename)
    external_feats_subdir = os.path.join(external_feats_dir, pdb_code, 'work')
    if is_rcsb_complex:  # Construct RCSB FASTA file names indirectly out of necessity
        df = original_pair.df0 if struct_idx == 0 else original_pair.df1
        raw_rcsb_filename = os.path.split(pdb_filename)
        rcsb_fasta_filename = os.path.join(external_feats_subdir,
                                           raw_rcsb_filename[1] +
                                           f'-{df.iloc[0]["model"]}' + f'-{df.iloc[0]["chain"]}.fa')
        fasta_files = [rcsb_fasta_filename]
    else:  # Construct non-DIPS FASTA file names more directly using raw PDB filenames
        fasta_files = [os.path.join(external_feats_subdir, file) for file in os.listdir(external_feats_subdir)
                       if pdb_full_name in file and '.fa' in file]

    # Get only the first sequence from each FASTA sequence object
    sequence_list = [sequence.seq._data for fasta_file in fasta_files for sequence in SeqIO.parse(fasta_file, 'fasta')]

    # Give each sequence a left or right-bound key
    for fasta_file, sequence in zip(fasta_files, sequence_list):
        if struct_idx == 0:
            sequences['l_b'] = sequence
        else:
            sequences['r_b'] = sequence
    return sequences


def min_max_normalize_feature_array(features):
    """Independently for each column, normalize feature array values to be in range [0, 1]."""
    scaler = MinMaxScaler()
    scaler.fit(features)
    features_scaled = scaler.transform(features)
    return features_scaled


def min_max_normalize_feature_tensor(features):
    """Normalize provided feature tensor to have its values be in range [0, 1]."""
    min_value = min(features)
    max_value = max(features)
    features_std = torch.tensor([(value - min_value) / (max_value - min_value) for value in features])
    features_scaled = features_std * (max_value - min_value) + min_value
    return features_scaled


def get_dssp_dict_for_pdb_file(pdb_filename):
    """Run DSSP to calculate secondary structure features for a given PDB file."""
    dssp_dict = {}  # Initialize to default DSSP dict value
    try:
        dssp_tuple = dssp_dict_from_pdb_file(pdb_filename)
        dssp_dict = dssp_tuple[0]
    except Exception:
        logging.info("No DSSP features found for {:}".format(pdb_filename))
    return dssp_dict


def get_dssp_dict_for_pdb_model(pdb_model, raw_pdb_filename):
    """Run DSSP to calculate secondary structure features for a given PDB file."""
    dssp_dict = {}  # Initialize to default DSSP dict value
    try:
        dssp_dict = DSSP(pdb_model, raw_pdb_filename)
    except Exception:
        logging.info("No DSSP features found for {:}".format(pdb_model))
    return dssp_dict


def get_msms_rd_dict_for_pdb_model(pdb_model):
    """Run MSMS to calculate residue depth model for a given PDB model."""
    rd_dict = {}  # Initialize to default RD dict value
    try:
        rd_dict = ResidueDepth(pdb_model)
    except Exception:
        logging.info("No MSMS residue depth model found for {:}".format(pdb_model))
    return rd_dict


# Following function adapted from PAIRPred (https://combi.cs.colostate.edu/supplements/pairpred/)
def get_df_from_psaia_tbl_file(psaia_filename):
    """Parse through a given PSAIA .tbl output file to construct a new Pandas DataFrame."""
    psaia_dict = {}  # Initialize to default PSAIA DataFrame
    psaia_df = pd.DataFrame(columns=['average CX', 's_avg CX', 's-ch avg CX', 's-ch s_avg CX', 'max CX', 'min CX'])
    # Attempt to parse all the lines of a single PSAIA .tbl file for residue-level protrusion values
    try:
        stnxt = 0
        ln = 0
        for l in open(psaia_filename, "r"):
            ln = ln + 1
            ls = l.split()
            if stnxt:
                cid = ls[0]
                if cid == '*':  # PSAIA replaces cid ' ' in pdb files with *
                    cid = ' '
                resid = (cid, ls[1])  # cid, resid, resname is ignored
                rcx = tuple(map(float, ls[3:9]))
                psaia_dict[resid] = rcx
            elif len(ls) and ls[0] == 'chain':  # the line containing 'chain' is the last line before real data starts
                stnxt = 1
        # Construct a new DataFrame from the parsed dictionary
        psaia_df = pd.DataFrame.from_dict(psaia_dict).T
        psaia_df.columns = PSAIA_COLUMNS
    except Exception:
        logging.info("Error in parsing PSAIA .tbl file {:}".format(psaia_filename))
    return psaia_df


def get_hsaac_for_pdb_residues(residues, similarity_matrix, raw_pdb_filename):
    """Run BioPython to calculate half-sphere amino acid composition (HSAAC) for a given list of PDB residues."""
    hsaacs = np.array([DEFAULT_MISSING_HSAAC for _ in range(len(residues))])  # Initialize to default HSAACs value
    try:
        UC, DC = get_hsacc(residues, similarity_matrix, raw_pdb_filename)
        hsaacs = np.concatenate((UC, DC))  # Concatenate to get HSAAC
    except Exception:
        logging.info("No half-sphere amino acid compositions (HSAACs) found for PDB file {:}".format(raw_pdb_filename))
    return hsaacs


def get_dssp_value_for_residue(dssp_dict: dict, feature: str, chain: str, residue: int):
    """Return a secondary structure (SS) value or a relative solvent accessibility (RSA) value for a given chain-residue pair."""
    dssp_value = DEFAULT_MISSING_SS if feature == 'SS' else DEFAULT_MISSING_RSA  # Initialize to default DSSP feature
    try:
        if feature == 'SS':
            dssp_values = dssp_dict[chain, (' ', residue, ' ')]
            dssp_value = dssp_values[2]
        else:  # feature == 'RSA'
            dssp_values = dssp_dict[chain, (' ', residue, ' ')]
            dssp_value = dssp_values[3]
    except Exception:
        logging.info("No DSSP entry found for {:}".format((chain, (' ', residue, ' '))))
    return dssp_value


def get_msms_rd_value_for_residue(rd_dict: dict, chain: str, residue: int):
    """Return an alpha-carbon residue depth (RD) value for a given chain-residue pair."""
    ca_depth_value = DEFAULT_MISSING_RD  # Initialize to default RD value
    try:
        rd_value, ca_depth_value = rd_dict[chain, (' ', residue, ' ')]
    except Exception:
        logging.info("No MSMS residue depth entry found for {:}".format((chain, (' ', residue, ' '))))
    return ca_depth_value[0] if type(ca_depth_value) == list else ca_depth_value


def get_protrusion_index_for_residue(psaia_df: pd.DataFrame, chain: str, residue: int):
    """Return a protrusion index for a given chain-residue pair."""
    protrusion_index = DEFAULT_MISSING_PROTRUSION_INDEX  # Initialize to default protrusion index
    try:
        protrusion_index = psaia_df.loc[(chain, str(residue))].to_list()
    except Exception:
        logging.info("No protrusion index entry found for {:}".format((chain, (' ', residue, ' '))))
    return protrusion_index


def get_hsaac_for_residue(hsaac_matrix: np.array, residue_counter: int, chain: str, residue_id: int):
    """Return a half-sphere amino acid composition (HSAAC) for a given chain-residue pair."""
    hsaac = np.array(DEFAULT_MISSING_HSAAC)  # Initialize to default HSAAC value
    try:
        hsaac = hsaac_matrix[:, residue_counter]
    except Exception:
        logging.info(
            "No half-sphere amino acid composition entry found for {:}".format((chain, (' ', residue_id, ' '))))
    return np.array(DEFAULT_MISSING_HSAAC) if len(hsaac) > HSAAC_DIM else hsaac  # Handle HSAAC parsing edge case


def get_cn_value_for_residue(cn_values: np.array, residue_counter: int, chain: str, residue_id: int):
    """Return a coordinate number value for a given chain-residue pair."""
    cn_value = DEFAULT_MISSING_CN  # Initialize to default HSAAC value
    try:
        cn_value = cn_values[residue_counter]
    except Exception:
        logging.info("No coordinate number entry found for {:}".format((chain, (' ', residue_id, ' '))))
    return cn_value


def get_sequence_feats_for_residue(sequence_feats_df: pd.DataFrame, chain: str, residue_id: int):
    """Return all pre-generated sequence features (from profile HMM) for a given chain-residue pair."""
    sequence_feats = DEFAULT_MISSING_SEQUENCE_FEATS  # Initialize to default sequence features
    try:
        # Sequence features start at the 5th column and end at the third-to-last column
        sequence_feats = sequence_feats_df[sequence_feats_df['chain'].apply(
            lambda x: x.strip() == chain) & sequence_feats_df['residue'].apply(
            lambda x: x.strip() == str(residue_id))].to_numpy()[0, 4:-3]  # Grab first matching chain-residue pair
        # First twenty feature entries are emission probabilities for the residue, and the last seven are its transition probabilities
    except Exception:
        logging.info("No sequence feature entries found for {:}".format((chain, (' ', residue_id, ' '))))
    return sequence_feats


def get_norm_vec_for_residue(df: pd.DataFrame, ca_atom: pd.Series, chain: str, residue_id: int):
    """Return a normal vector for a given residue."""
    norm_vec = DEFAULT_MISSING_NORM_VEC  # Initialize to default norm vec value
    try:
        # Calculate normal vector for each residue's amide plane using relative coords of each Ca-Cb and Cb-N bond
        cb_atom = df[(df.chain == ca_atom.chain) &
                     (df.residue == ca_atom.residue) &
                     (df.atom_name == 'CB')]
        n_atom = df[(df.chain == ca_atom.chain) &
                    (df.residue == ca_atom.residue) &
                    (df.atom_name == 'N')]
        vec1 = ca_atom[['x', 'y', 'z']].to_numpy() - cb_atom[['x', 'y', 'z']].to_numpy()
        vec2 = cb_atom[['x', 'y', 'z']].to_numpy() - n_atom[['x', 'y', 'z']].to_numpy()
        norm_vec = np.cross(vec1, vec2)
    except Exception:
        logging.info("No normal vector entry found for {:}".format(chain, (' ', residue_id, ' ')))
    if len(norm_vec) == 0:  # Catch a missing normal vector, possibly from the residue missing a CB atom (e.g. Glycine)
        norm_vec = DEFAULT_MISSING_NORM_VEC
    return norm_vec


def get_raw_pdb_filename_from_interim_filename(interim_filename: str, raw_pdb_dir: str, source_type: str):
    """Get raw pdb filename from interim filename."""
    pdb_name = interim_filename
    slash_tokens = pdb_name.split(os.path.sep)
    slash_dot_tokens = slash_tokens[-1].split(".")
    raw_pdb_filename = os.path.join(raw_pdb_dir, slash_tokens[-2], slash_dot_tokens[0]) + '.' + slash_dot_tokens[1] if \
        source_type.lower() in ['rcsb', 'evcoupling', 'casp_capri', 'input'] else \
        os.path.join(raw_pdb_dir, slash_dot_tokens[0].split('_')[0], slash_dot_tokens[0]) + '.' + slash_dot_tokens[1]
    return raw_pdb_filename


def __should_keep_postprocessed(raw_pdb_dir: str, pair_filename: str, source_type: str):
    """Determine if given pair filename corresponds to a pair of structures, both with DSSP-derivable secondary structure features."""
    # pair_name example: 20gs.pdb1_0
    raw_pdb_filenames = []
    pair = pd.read_pickle(pair_filename)
    for i, interim_filename in enumerate(pair.srcs.values()):  # Unbound source filenames to be converted to bound ones
        # Identify if a given complex contains DSSP-derivable secondary structure features
        raw_pdb_filenames.append(get_raw_pdb_filename_from_interim_filename(interim_filename, raw_pdb_dir, source_type))
        pair_dssp_dict = get_dssp_dict_for_pdb_file(raw_pdb_filenames[i])
        if source_type.lower() not in ['input'] and not pair_dssp_dict:
            return pair, raw_pdb_filenames[i], False  # Discard pair missing DSSP-derivable secondary structure features
        if source_type.lower() not in ['input'] \
                and (pair.df0.shape[0] > ATOM_COUNT_LIMIT or pair.df1.shape[0] > ATOM_COUNT_LIMIT):
            return pair, raw_pdb_filenames[i], False  # Discard pair exceeding atom count limit to reduce comp. complex.
    return pair, raw_pdb_filenames, True


def postprocess_pruned_pairs(raw_pdb_dir: str, external_feats_dir: str, pair_filename: str,
                             output_filename: str, source_type: str):
    """Check if underlying PDB file for pair_filename contains DSSP-derivable features. If yes, postprocess its derived features and write them into three separate output_filenames. Otherwise, delete it if it is already in output_filename."""
    output_file_exists = os.path.exists(output_filename)
    pair, raw_pdb_filenames, should_keep = __should_keep_postprocessed(raw_pdb_dir, pair_filename, source_type)
    if should_keep:
        postprocessed_pair = postprocess_pruned_pair(raw_pdb_filenames, external_feats_dir, pair, source_type)
        if not output_file_exists:
            # Write into output_filenames if not exist
            with open(output_filename, 'wb') as f:
                pickle.dump(postprocessed_pair, f)
    else:
        if output_file_exists:
            # Delete the output_filenames
            os.remove(output_filename)


def postprocess_pruned_pair(raw_pdb_filenames: List[str], external_feats_dir: str, original_pair, source_type: str):
    """Construct a new Pair consisting of residues of structures with DSSP-derivable features and append DSSP secondary structure (SS) features to each protein structure dataframe as well."""
    chains_selected = [original_pair.df0['chain'][0], original_pair.df1['chain'][0]]
    df0_ss_values, df0_rsa_values, df0_rd_values, df0_protrusion_indices, \
    df0_hsaacs, df0_cn_values, df0_sequence_feats, df0_amide_norm_vecs, \
    df1_ss_values, df1_rsa_values, df1_rd_values, df1_protrusion_indices, \
    df1_hsaacs, df1_cn_values, df1_sequence_feats, df1_amide_norm_vecs, = [], [], [], [], [], [], [], [], \
                                                                          [], [], [], [], [], [], [], []
    single_raw_pdb_file_provided = len(list(set(raw_pdb_filenames))) == 1

    # Collect sequence and structure based features for each provided pair file (e.g. left-bound and right-bound files)
    sequences = {}
    dssp_dicts, rd_dicts, psaia_dfs, coordinate_numbers_list, hsaac_matrices, sequence_feats_dfs = [], [], [], \
                                                                                                   [], [], []
    raw_pdb_filenames.sort()  # Ensure the left input PDB is processed first
    for struct_idx, raw_pdb_filename in enumerate(raw_pdb_filenames):
        is_rcsb_complex = source_type.lower() in ['rcsb', 'evcoupling', 'casp_capri']

        # Extract the FASTA sequence(s) for a given PDB file
        sequences = find_fasta_sequences_for_pdb_file(sequences,
                                                      raw_pdb_filename,
                                                      external_feats_dir,
                                                      struct_idx,
                                                      is_rcsb_complex,
                                                      original_pair)

        # Avoid redundant feature fetching for RCSB complexes
        if (is_rcsb_complex and struct_idx == 0) or not is_rcsb_complex:
            # Derive BioPython structure and residues for the given PDB file
            structure = PDB_PARSER.get_structure(original_pair.complex, raw_pdb_filename)  # PDB structure
            # Filter out all hetero residues including waters to leave only amino and nucleic acids
            residues = [residue for residue in Selection.unfold_entities(structure, 'R')
                        if residue.get_id()[0] == ' ' and residue.get_parent().id in chains_selected]

            # Extract DSSP secondary structure (SS) and relative solvent accessibility (RSA) values for the 1st model
            dssp_dict = get_dssp_dict_for_pdb_model(structure[0], raw_pdb_filename)  # Only for 1st model

            # Extract residue depth (RD) values for each PDB model using MSMS
            rd_dict = get_msms_rd_dict_for_pdb_model(structure[0])  # RD only retrieved for first model

            # Get protrusion indices using PSAIA
            pdb_code = db.get_pdb_code(raw_pdb_filename)
            psaia_filenames = [path for path in Path(external_feats_dir).rglob(f'{pdb_code}*.tbl')]
            psaia_filenames.sort()  # Ensure the left input PDB is processed first
            psaia_df = get_df_from_psaia_tbl_file(psaia_filenames[struct_idx])

            # Extract half-sphere exposure (HSE) statistics for each PDB model (including HSAAC and CN values)
            similarity_matrix, coordinate_numbers = get_similarity_matrix(get_coords(residues))
            hsaac_matrix = get_hsaac_for_pdb_residues(residues, similarity_matrix, raw_pdb_filename)

            # Retrieve pre-generated sequence features (i.e. transition and emission probabilities via HH-suite3)
            seq_file_index = 'src0' if struct_idx == 0 else 'src1'
            file = os.path.split(original_pair.srcs[seq_file_index])[-1]
            sequence_feats_filepath = os.path.join(external_feats_dir, db.get_pdb_code(file)[1:3], file)
            sequence_feats_df = pd.read_pickle(sequence_feats_filepath)

            # Collect gathered features for later postprocessing below
            dssp_dicts.append(dssp_dict)
            rd_dicts.append(rd_dict)
            psaia_dfs.append(psaia_df)
            coordinate_numbers_list.append(coordinate_numbers)
            hsaac_matrices.append(hsaac_matrix)
            sequence_feats_dfs.append(sequence_feats_df)

    # -------------
    # DataFrame 0
    # -------------

    # Determine which feature data structures to pull features out of for the first DataFrame
    df0_dssp_dict = dssp_dicts[0]
    df0_rd_dict = rd_dicts[0]
    df0_hsaac_matrix = hsaac_matrices[0]
    df0_coordinate_numbers = coordinate_numbers_list[0]
    df0_raw_pdf_filename = raw_pdb_filenames[0]
    df0_psaia_df = psaia_dfs[0]
    df0_sequence_feats_df = sequence_feats_dfs[0]

    # Add SS and RSA values to the residues in the first dataframe, df0, of a pair of dataframes
    df0: pd.DataFrame = original_pair.df0

    # Iterate through each residue in the first structure and collect extracted features for training and reporting
    residue_counter = 0
    for index, row in df0.iterrows():
        # Aggregate features for each residues' alpha-carbon (CA) atom
        is_ca_atom = 'CA' in row.atom_name

        # Parse information from residue ID
        residue_id = row.residue.strip().lstrip("-+")
        residue_is_inserted = not residue_id.isdigit()
        residue_id = int(residue_id) if not residue_is_inserted else residue_id

        # Collect features for each residue
        dssp_ss_value_for_atom = get_dssp_value_for_residue(df0_dssp_dict, 'SS', row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_SS
        dssp_rsa_value_for_atom = get_dssp_value_for_residue(df0_dssp_dict, 'RSA', row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_RSA
        msms_rd_value_for_atom = get_msms_rd_value_for_residue(df0_rd_dict, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_RD
        protrusion_index_for_atom = get_protrusion_index_for_residue(df0_psaia_df, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_PROTRUSION_INDEX
        hsaac_for_atom = get_hsaac_for_residue(df0_hsaac_matrix, residue_counter, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_HSAAC
        cn_value_for_atom = get_cn_value_for_residue(
            df0_coordinate_numbers, residue_counter, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_CN
        sequence_feats_for_atom = get_sequence_feats_for_residue(
            df0_sequence_feats_df, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_SEQUENCE_FEATS
        norm_vec_for_atom = get_norm_vec_for_residue(original_pair.df0, row, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_NORM_VEC

        # Handle missing normal vectors
        if is_ca_atom and np.nan in norm_vec_for_atom:
            logging.info(f'Normal vector missing for df0 residue {row.residue}'
                         f'in chain {row.chain} in file {df0_raw_pdf_filename}')
            df0_amide_norm_vecs.append(np.array(norm_vec_for_atom))
        elif is_ca_atom:  # Normal vector was found successfully
            df0_amide_norm_vecs.append(norm_vec_for_atom[0])  # 2D array with a single inner array -> 1D array
        else:
            df0_amide_norm_vecs.append(np.array(norm_vec_for_atom))

        # Aggregate feature values
        df0_ss_values += dssp_ss_value_for_atom
        df0_rsa_values.append(dssp_rsa_value_for_atom)
        df0_rd_values.append(msms_rd_value_for_atom)
        df0_protrusion_indices.append(protrusion_index_for_atom)
        df0_hsaacs.append(hsaac_for_atom)
        df0_cn_values.append(cn_value_for_atom)
        df0_sequence_feats.append(sequence_feats_for_atom)

        # Report presence of inserted residues
        if is_ca_atom and residue_is_inserted:
            logging.info('Found inserted df0 residue entry for residue ' + row.resname + ': '
                         + '(\'' + row.chain + '\', \'' + row.residue + '\')')

        # Increment residue counter for each alpha-carbon encountered
        if 'CA' in row.atom_name:
            residue_counter += 1

    # Normalize df0 residue features to be in range [0, 1]
    df0_rd_values = min_max_normalize_feature_array(np.array(df0_rd_values).reshape(-1, 1))
    df0_protrusion_indices = min_max_normalize_feature_array(np.array(df0_protrusion_indices))
    df0_cn_values = min_max_normalize_feature_array(np.array(df0_cn_values).reshape(-1, 1))

    # Insert new df0 features
    df0.insert(5, 'ss_value', df0_ss_values, False)
    df0.insert(6, 'rsa_value', df0_rsa_values, False)
    df0.insert(7, 'rd_value', df0_rd_values, False)
    # Insert all protrusion index fields sequentially
    df0_col_idx = 8
    for struct_idx, col_name in enumerate(PSAIA_COLUMNS):
        df0.insert(df0_col_idx, col_name, df0_protrusion_indices[:, struct_idx], False)
        df0_col_idx += 1
    df0.insert(14, 'hsaac', df0_hsaacs, False)
    df0.insert(15, 'cn_value', df0_cn_values, False)
    df0.insert(16, 'sequence_feats', df0_sequence_feats, False)
    df0.insert(17, 'amide_norm_vec', df0_amide_norm_vecs, False)

    # -------------
    # DataFrame 1
    # -------------

    # Determine which feature data structures to pull features out of for the second DataFrame
    df1_dssp_dict = dssp_dicts[0] if single_raw_pdb_file_provided else dssp_dicts[1]
    df1_rd_dict = rd_dicts[0] if single_raw_pdb_file_provided else rd_dicts[1]
    df1_hsaac_matrix = hsaac_matrices[0] if single_raw_pdb_file_provided else hsaac_matrices[1]
    df1_coordinate_numbers = coordinate_numbers_list[0] if single_raw_pdb_file_provided else coordinate_numbers_list[1]
    df1_raw_pdf_filename = raw_pdb_filenames[0] if single_raw_pdb_file_provided else raw_pdb_filenames[1]
    df1_psaia_df = psaia_dfs[0] if single_raw_pdb_file_provided else psaia_dfs[1]
    df1_sequence_feats_df = sequence_feats_dfs[0] if single_raw_pdb_file_provided else sequence_feats_dfs[1]

    # Add SS and RSA values to the residues in the second dataframe, df1, of a pair of dataframes
    df1: pd.DataFrame = original_pair.df1

    # Iterate through each residue in the second structure and collect extracted features for training and reporting
    residue_counter = 0
    for index, row in df1.iterrows():
        # Aggregate features for each residues' alpha-carbon (CA) atom
        is_ca_atom = 'CA' in row.atom_name

        # Parse information from residue ID
        residue_id = row.residue.strip().lstrip("-+")
        residue_is_inserted = not residue_id.isdigit()
        residue_id = int(residue_id) if not residue_is_inserted else residue_id

        # Collect features for each residue
        dssp_ss_value_for_atom = get_dssp_value_for_residue(df1_dssp_dict, 'SS', row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_SS
        dssp_rsa_value_for_atom = get_dssp_value_for_residue(df1_dssp_dict, 'RSA', row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_RSA
        msms_rd_value_for_atom = get_msms_rd_value_for_residue(df1_rd_dict, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_RD
        protrusion_index_for_atom = get_protrusion_index_for_residue(df1_psaia_df, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_PROTRUSION_INDEX
        hsaac_for_atom = get_hsaac_for_residue(df1_hsaac_matrix, residue_counter, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_HSAAC
        cn_value_for_atom = get_cn_value_for_residue(
            df1_coordinate_numbers, residue_counter, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_CN
        sequence_feats_for_atom = get_sequence_feats_for_residue(
            df1_sequence_feats_df, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_SEQUENCE_FEATS
        norm_vec_for_atom = get_norm_vec_for_residue(original_pair.df1, row, row.chain.strip(), residue_id) \
            if is_ca_atom else DEFAULT_MISSING_NORM_VEC

        # Handle missing normal vectors
        if is_ca_atom and np.nan in norm_vec_for_atom:
            logging.info(f'Normal vector missing for df1 residue {row.residue}'
                         f'in chain {row.chain} in file {df1_raw_pdf_filename}')
            df1_amide_norm_vecs.append(np.array(norm_vec_for_atom))
        elif is_ca_atom:  # Normal vector was found successfully
            df1_amide_norm_vecs.append(norm_vec_for_atom[0])  # 2D array with a single inner array -> 1D array
        else:
            df1_amide_norm_vecs.append(norm_vec_for_atom)

        # Aggregate feature values
        df1_ss_values += dssp_ss_value_for_atom
        df1_rsa_values.append(dssp_rsa_value_for_atom)
        df1_rd_values.append(msms_rd_value_for_atom)
        df1_protrusion_indices.append(protrusion_index_for_atom)
        df1_hsaacs.append(hsaac_for_atom)
        df1_cn_values.append(cn_value_for_atom)
        df1_sequence_feats.append(sequence_feats_for_atom)

        # Report presence of inserted residues
        if is_ca_atom and residue_is_inserted:
            logging.info('Found inserted df1 residue entry for residue ' + row.resname + ': '
                         + '(\'' + row.chain + '\', \'' + row.residue + '\')')

        # Increment residue counter for each alpha-carbon encountered
        if is_ca_atom:
            residue_counter += 1

    # Normalize df1 residue features to be in range [0, 1]
    df1_rd_values = min_max_normalize_feature_array(np.array(df1_rd_values).reshape(-1, 1))
    df1_protrusion_indices = min_max_normalize_feature_array(np.array(df1_protrusion_indices))
    df1_cn_values = min_max_normalize_feature_array(np.array(df1_cn_values).reshape(-1, 1))

    # Insert new df1 features
    df1.insert(5, 'ss_value', df1_ss_values, False)
    df1.insert(6, 'rsa_value', df1_rsa_values, False)
    df1.insert(7, 'rd_value', df1_rd_values, False)
    # Insert all protrusion index fields sequentially
    df1_col_idx = 8
    for struct_idx, col_name in enumerate(PSAIA_COLUMNS):
        df1.insert(df1_col_idx, col_name, df1_protrusion_indices[:, struct_idx], False)
        df1_col_idx += 1
    df1.insert(14, 'hsaac', df1_hsaacs, False)
    df1.insert(15, 'cn_value', df1_cn_values, False)
    df1.insert(16, 'sequence_feats', df1_sequence_feats, False)
    df1.insert(17, 'amide_norm_vec', df1_amide_norm_vecs, False)

    # Reconstruct a Pair representing a complex of interacting proteins
    pair = pa.Pair(complex=original_pair.complex, df0=df0, df1=df1,
                   pos_idx=original_pair.pos_idx, neg_idx=original_pair.neg_idx,
                   srcs=original_pair.srcs, id=original_pair.id, sequences=sequences)
    return pair


def collect_dataset_statistics(output_dir: str):
    """Aggregate statistics for a postprocessed dataset."""
    dataset_statistics = DEFAULT_DATASET_STATISTICS
    # Look at each .dill file in the given output directory
    pair_filenames = [pair_filename.as_posix() for pair_filename in Path(output_dir).rglob('*.dill')]
    for i in tqdm(range(len(pair_filenames))):
        postprocessed_pair: pa.Pair = pd.read_pickle(pair_filenames[i])

        # Keep track of how many complexes have already been postprocessed
        dataset_statistics['num_of_processed_complexes'] += 1

        # -------------
        # DataFrame 0
        # -------------

        # Grab first structure's DataFrame (CA atoms by default)
        df0: pd.DataFrame = postprocessed_pair.df0[postprocessed_pair.df0['atom_name'].apply(lambda x: x == 'CA')]

        # Increment feature counters
        dataset_statistics['num_of_df0_interface_residues'] += len(postprocessed_pair.pos_idx[:, 0])
        dataset_statistics['num_of_valid_df0_ss_values'] += len(df0[df0['ss_value'] != '-'])
        dataset_statistics['num_of_valid_df0_rsa_values'] += len(df0[~df0['rsa_value'].isna()])
        dataset_statistics['num_of_valid_df0_rd_values'] += len(df0[~df0['rd_value'].isna()])
        num_nonzero_protrusion_indices = len(df0[(~df0['avg_cx'].isna()) & (~df0['s_avg_cx'].isna())
                                                 & (~df0['s_ch_avg_cx'].isna()) & (~df0['s_ch_s_avg_cx'].isna())
                                                 & (~df0['max_cx'].isna()) & (~df0['min_cx'].isna())])
        dataset_statistics['num_of_valid_df0_protrusion_indices'] += num_nonzero_protrusion_indices
        for hsaac_array in df0['hsaac']:
            if np.sum(np.isnan(hsaac_array)) == 0:
                dataset_statistics['num_of_valid_df0_hsaacs'] += 1
        dataset_statistics['num_of_valid_df0_cn_values'] += len(df0[~df0['cn_value'].isna()])
        for sequence_array in df0['sequence_feats']:
            if np.sum(np.isnan(sequence_array.astype(np.float))) == 0:
                dataset_statistics['num_of_valid_df0_sequence_feats'] += 1
        for amide_normal_vec in df0['amide_norm_vec']:
            if np.sum(np.isnan(amide_normal_vec.astype(np.float))) == 0:
                dataset_statistics['num_of_valid_df0_amide_normal_vecs'] += 1

        # Increment total residue count for first structure
        dataset_statistics['num_of_df0_residues'] += len(df0)

        # -------------
        # DataFrame 1
        # -------------

        # Grab second structure's DataFrame (CA atoms by default)
        df1: pd.DataFrame = postprocessed_pair.df1[postprocessed_pair.df1['atom_name'].apply(lambda x: x == 'CA')]

        # Increment feature counters
        dataset_statistics['num_of_df1_interface_residues'] += len(postprocessed_pair.pos_idx[:, 1])
        dataset_statistics['num_of_valid_df1_ss_values'] += len(df1[df1['ss_value'] != '-'])
        dataset_statistics['num_of_valid_df1_rsa_values'] += len(df1[~df1['rsa_value'].isna()])
        dataset_statistics['num_of_valid_df1_rd_values'] += len(df1[~df1['rd_value'].isna()])
        num_nonzero_protrusion_indices = len(df1[(~df1['avg_cx'].isna()) & (~df1['s_avg_cx'].isna())
                                                 & (~df1['s_ch_avg_cx'].isna()) & (~df1['s_ch_s_avg_cx'].isna())
                                                 & (~df1['max_cx'].isna()) & (~df1['min_cx'].isna())])
        dataset_statistics['num_of_valid_df1_protrusion_indices'] += num_nonzero_protrusion_indices
        for hsaac_array in df1['hsaac']:
            if np.sum(np.isnan(hsaac_array)) == 0:
                dataset_statistics['num_of_valid_df1_hsaacs'] += 1
        dataset_statistics['num_of_valid_df1_cn_values'] += len(df1[~df1['cn_value'].isna()])
        for sequence_array in df1['sequence_feats']:
            if np.sum(np.isnan(sequence_array.astype(np.float))) == 0:
                dataset_statistics['num_of_valid_df1_sequence_feats'] += 1
        for amide_normal_vec in df1['amide_norm_vec']:
            if np.sum(np.isnan(amide_normal_vec.astype(np.float))) == 0:
                dataset_statistics['num_of_valid_df1_amide_normal_vecs'] += 1

        # Increment total residue count for second structure
        dataset_statistics['num_of_df1_residues'] += len(df1)

        # Aggregate pair counts for logging final statistics
        num_unique_res_pairs = len(df0) * len(df1)
        dataset_statistics['num_of_pos_res_pairs'] += len(postprocessed_pair.pos_idx)
        dataset_statistics['num_of_neg_res_pairs'] += (num_unique_res_pairs - len(postprocessed_pair.pos_idx))
        dataset_statistics['num_of_res_pairs'] += num_unique_res_pairs

    return dataset_statistics


def log_dataset_statistics(logger, dataset_statistics: dict):
    """Output pair postprocessing statistics."""
    logger.info(f'{dataset_statistics["num_of_processed_complexes"]} complexes copied')

    logger.info(f'{dataset_statistics["num_of_df0_residues"]} residues found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_df1_residues"]} residues found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_df0_interface_residues"]} residues found in df0 interfaces in total')
    logger.info(f'{dataset_statistics["num_of_df1_interface_residues"]} residues found in df1 interfaces in total')

    logger.info(f'{dataset_statistics["num_of_df0_interface_residues"] / dataset_statistics["num_of_df0_residues"]}'
                f' percent of total df0 residues found in interfaces on average')
    logger.info(f'{dataset_statistics["num_of_df1_interface_residues"] / dataset_statistics["num_of_df1_residues"]}'
                f' percent of total df1 residues found in interfaces on average')

    logger.info(f'{dataset_statistics["num_of_pos_res_pairs"]} positive residue pairs found in total')
    logger.info(f'{dataset_statistics["num_of_neg_res_pairs"]} negative residue pairs found in total')

    logger.info(f'{dataset_statistics["num_of_pos_res_pairs"] / dataset_statistics["num_of_res_pairs"]}'
                f' positive residue pairs found in complexes on average')
    logger.info(f'{dataset_statistics["num_of_neg_res_pairs"] / dataset_statistics["num_of_res_pairs"]}'
                f' negative residue pairs found in complexes on average')

    logger.info(f'{dataset_statistics["num_of_valid_df0_ss_values"]}'
                f' valid secondary structure (SS) values found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_ss_values"]}'
                f' valid secondary structure (SS) values found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_rsa_values"]}'
                f' valid relative solvent accessibility (RSA) values found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_rsa_values"]}'
                f' valid relative solvent accessibility (RSA) values found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_rd_values"]}'
                f' valid residue depth (RD) values found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_rd_values"]}'
                f' valid residue depth (RD) values found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_protrusion_indices"]}'
                f' valid protrusion indices found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_protrusion_indices"]}'
                f' valid protrusion indices found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_hsaacs"]}'
                f' valid half-sphere amino acid compositions (HSAACs) found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_hsaacs"]}'
                f' valid half-sphere amino acid compositions (HSAACs) found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_cn_values"]}'
                f' valid coordinate number (CN) values found in df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_cn_values"]}'
                f' valid coordinate number (CN) values found in df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_sequence_feats"]}'
                f' valid sequence feature arrays found for df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_sequence_feats"]}'
                f' valid sequence feature arrays found for df1 structures in total')

    logger.info(f'{dataset_statistics["num_of_valid_df0_amide_normal_vecs"]}'
                f' valid amide normal vectors found for df0 structures in total')
    logger.info(f'{dataset_statistics["num_of_valid_df1_amide_normal_vecs"]}'
                f' valid amide normal vectors found for df1 structures in total')


def determine_nan_fill_value(column: pd.Series, imputation_method='median'):
    """Determine whether to fill NaNs in a given column with the column's requested fill value or instead with zero."""
    cleaned_imputation_method = imputation_method.strip().lower()
    # Determine user-requested imputation (fill) value
    if cleaned_imputation_method == 'mean':
        imputation_value = column.mean()
    elif cleaned_imputation_method == 'min':
        imputation_value = column.min()
    elif cleaned_imputation_method == 'max':
        imputation_value = column.max()
    elif cleaned_imputation_method == 'zero':
        imputation_value = 0
    else:  # Default to replacing NaNs with the column's median value
        imputation_value = column.median()
    return imputation_value if column.isna().sum().sum() <= NUM_ALLOWABLE_NANS else 0


def impute_postprocessed_missing_feature_values(input_pair_filename: str, output_pair_filename: str,
                                                impute_atom_features: bool, advanced_logging: bool):
    """Impute missing feature values in a postprocessed dataset."""
    # Look at a .dill file in the given output directory
    postprocessed_pair: pa.Pair = pd.read_pickle(input_pair_filename)

    # -------------
    # DataFrames
    # -------------
    updated_dfs = []
    for df_idx, df in enumerate([postprocessed_pair.df0, postprocessed_pair.df1]):
        # Collect indices of requested atoms - all atom indices if atom feature imputation requested and only CAs if not
        atom_indices = df[df['atom_name'].apply(lambda x: x == 'CA')].index if not impute_atom_features else df.index

        # Grab structure's feature columns
        df.iloc[:, 6].replace('NA', np.nan, inplace=True)  # Replace NA (i.e. DSSP-stringified NaN value) with np.nan
        numeric_feat_cols = df.iloc[atom_indices, 6:14]
        hsaacs = []
        for hsaac in df.iloc[atom_indices, 14]:
            hsaacs.append(hsaac)
        hsaacs = pd.DataFrame(np.array(hsaacs))
        cns = df.iloc[atom_indices, 15]
        sequence_feats = pd.DataFrame(np.array([sequence_feats for sequence_feats in df.iloc[atom_indices, 16]]))
        amide_normal_vecs = pd.DataFrame(np.array([sequence_feats for sequence_feats in df.iloc[atom_indices, 17]]))

        # Initially inspect whether there are missing features in the structure
        numeric_feat_cols_have_nan = numeric_feat_cols.isna().values.any()
        hsaacs_have_nan = hsaacs.isna().values.any()
        cns_have_nan = cns.isna().values.any()
        sequence_feats_have_nan = sequence_feats.isna().values.any()
        amide_normal_vecs_have_nan = amide_normal_vecs.isna().values.any()
        nan_found = numeric_feat_cols_have_nan or \
                    hsaacs_have_nan or \
                    cns_have_nan or \
                    sequence_feats_have_nan or \
                    amide_normal_vecs_have_nan
        if nan_found:
            if advanced_logging:
                logging.info(f"""Before Feature Imputation:
                                | df{df_idx} (from {input_pair_filename}) contained at least one NaN value |
                                df{df_idx}_numeric_feat_cols_have_nan: {numeric_feat_cols_have_nan}
                                df{df_idx}_hsaacs_have_nan: {hsaacs_have_nan}
                                df{df_idx}_cns_have_nan: {cns_have_nan}
                                df{df_idx}_sequence_feats_have_nan: {sequence_feats_have_nan}
                                df{df_idx}_amide_normal_vecs_have_nan: {amide_normal_vecs_have_nan}""")

        # Impute structure's missing feature values uniquely for each column
        numeric_feat_cols = numeric_feat_cols.apply(lambda col: col.fillna(determine_nan_fill_value(col)), axis=0)
        df.iloc[atom_indices, 6:14] = numeric_feat_cols

        hsaacs = hsaacs.apply(lambda col: col.fillna(determine_nan_fill_value(col)), axis=0)
        for atom_index, df_hsaac in zip(atom_indices, hsaacs.values.tolist()):
            df.at[atom_index, 'hsaac'] = df_hsaac

        cns = cns.fillna(determine_nan_fill_value(cns))
        df.iloc[atom_indices, 15] = cns

        sequence_feats = sequence_feats.apply(lambda col: col.fillna(determine_nan_fill_value(col)), axis=0)
        for atom_index, seq_feats in zip(atom_indices, [np.array(lst) for lst in sequence_feats.values.tolist()]):
            df.at[atom_index, 'sequence_feats'] = seq_feats

        amide_normal_vecs = amide_normal_vecs.apply(lambda col: col.fillna(
            determine_nan_fill_value(col, imputation_method='zero')), axis=0)
        for atom_index, nv in zip(atom_indices, [np.array(lst) for lst in amide_normal_vecs.values.tolist()]):
            df.at[atom_index, 'amide_norm_vec'] = nv

        numeric_feat_cols_have_nan = numeric_feat_cols.isna().values.any()
        hsaacs_have_nan = hsaacs.isna().values.any()
        cns_have_nan = cns.isna().values.any()
        sequence_feats_have_nan = sequence_feats.isna().values.any()
        amide_normal_vecs_have_nan = amide_normal_vecs.isna().values.any()
        nan_found = numeric_feat_cols_have_nan or \
                    hsaacs_have_nan or \
                    cns_have_nan or \
                    sequence_feats_have_nan or \
                    amide_normal_vecs_have_nan
        if nan_found:
            raise Exception(f"""After Feature Imputation:
                            | df{df_idx} (from {input_pair_filename}) contained at least one NaN value |
                            df{df_idx}_numeric_feat_cols_have_nan: {numeric_feat_cols_have_nan}
                            df{df_idx}_hsaacs_have_nan: {hsaacs_have_nan}
                            df{df_idx}_cns_have_nan: {cns_have_nan}
                            df{df_idx}_sequence_feats_have_nan: {sequence_feats_have_nan}
                            df{df_idx}_amide_normal_vecs_have_nan: {amide_normal_vecs_have_nan}""")

        updated_dfs.append(df)

    # Reconstruct a feature-imputed Pair representing a complex of interacting proteins
    df0, df1 = updated_dfs[0], updated_dfs[1]
    feature_imputed_pair = pa.Pair(complex=postprocessed_pair.complex, df0=df0, df1=df1,
                                   pos_idx=postprocessed_pair.pos_idx, neg_idx=postprocessed_pair.neg_idx,
                                   srcs=postprocessed_pair.srcs, id=postprocessed_pair.id,
                                   sequences=postprocessed_pair.sequences)

    # Write into current pair_filename
    with open(output_pair_filename, 'wb') as f:
        pickle.dump(feature_imputed_pair, f)
