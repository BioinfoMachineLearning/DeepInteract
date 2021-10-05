import logging
import os
import random
from pathlib import Path

import atom3.pair as pa
import click
import pandas as pd
from tqdm import tqdm

from project.utils.deepinteract_constants import RESIDUE_COUNT_LIMIT, KNN, \
    EXCLUDED_COMPLEX_PAIR_FILENAMES


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------


@click.command()
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
@click.option('--source_type', default='rcsb', type=click.Choice(['rcsb']))
@click.option('--filter_by_atom_count', '-f', default=True)
@click.option('--min_atom_count', '-l', default=KNN)
@click.option('--max_atom_count', '-l', default=RESIDUE_COUNT_LIMIT)
@click.option('--atom_filter_type', '-t', default='ca', type=click.Choice(['ca', 'all']))
def main(output_dir: str, source_type: str, filter_by_atom_count: bool,
         min_atom_count: int, max_atom_count: int, atom_filter_type: str):
    """Partition dataset filenames."""
    logger = logging.getLogger(__name__)
    logger.info(f'Writing filename DataFrames to their respective text files')

    # Make sure the output_dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pairs_postprocessed_txt = os.path.join(output_dir, 'pairs-postprocessed.txt')
    open(pairs_postprocessed_txt, 'w').close()  # Create pairs-postprocessed.txt from scratch each run

    # Record dataset filenames conditionally by sequence length (if requested - otherwise, record all)
    pair_filenames = [pair_filename for pair_filename in Path(output_dir).rglob('*.dill')]
    for pair_filename in tqdm(pair_filenames):
        struct_id = pair_filename.as_posix().split(os.sep)[-2]
        if filter_by_atom_count and source_type.lower() == 'rcsb':
            postprocessed_pair: pa.Pair = pd.read_pickle(pair_filename)
            # Retrieve requested type of atoms from each structure's DataFrame
            if atom_filter_type == 'ca':
                df0_atoms = postprocessed_pair.df0[postprocessed_pair.df0['atom_name'] == 'CA']
                df1_atoms = postprocessed_pair.df1[postprocessed_pair.df1['atom_name'] == 'CA']
            else:
                df0_atoms, df1_atoms = postprocessed_pair.df0, postprocessed_pair.df1
            complex_num_interactions = len(df0_atoms) * len(df1_atoms)
            complex_meets_lower_size_bound = len(df0_atoms) > min_atom_count and len(df1_atoms) > min_atom_count
            complex_meets_upper_size_bound = complex_num_interactions < (max_atom_count ** 2)  # e.g., 256^2 = 65,536
            complex_not_excluded = pair_filename not in EXCLUDED_COMPLEX_PAIR_FILENAMES
            if complex_meets_lower_size_bound and complex_meets_upper_size_bound and complex_not_excluded:
                with open(pairs_postprocessed_txt, 'a') as f:
                    path, filename = os.path.split(pair_filename.as_posix())
                    filename = os.path.join(struct_id, filename)
                    f.write(filename + '\n')  # Pair file was copied
        else:
            with open(pairs_postprocessed_txt, 'a') as f:
                path, filename = os.path.split(pair_filename.as_posix())
                filename = os.path.join(struct_id, filename)
                f.write(filename + '\n')  # Pair file was copied

    # Separate training samples from validation samples
    if source_type.lower() == 'rcsb':
        # Prepare files
        pairs_postprocessed_train_txt = os.path.join(output_dir, 'pairs-postprocessed-train.txt')
        if not os.path.exists(pairs_postprocessed_train_txt):  # Create train data list if not already existent
            open(pairs_postprocessed_train_txt, 'w').close()
        pairs_postprocessed_val_txt = os.path.join(output_dir, 'pairs-postprocessed-val.txt')
        if not os.path.exists(pairs_postprocessed_val_txt):  # Create val data list if not already existent
            open(pairs_postprocessed_val_txt, 'w').close()
        pairs_postprocessed_test_txt = os.path.join(output_dir, 'pairs-postprocessed-test.txt')
        if not os.path.exists(pairs_postprocessed_test_txt):  # Create test data list if not already existent
            open(pairs_postprocessed_test_txt, 'w').close()
        # Write out training-validation-testing partitions for DIPS
        output_dirs = [filename
                       for filename in os.listdir(output_dir)
                       if os.path.isdir(os.path.join(output_dir, filename))]
        # Get training and validation directories separately
        num_train_dirs = int(0.8 * len(output_dirs))
        num_val_dirs = int(0.25 * num_train_dirs)
        train_dirs = random.sample(output_dirs, num_train_dirs)
        test_dirs = list(set(output_dirs) - set(train_dirs))
        val_dirs = random.sample(train_dirs, num_val_dirs)
        train_dirs = list(set(train_dirs) - set(val_dirs))
        # Ascertain training and validation filename separately
        filenames_frame = pd.read_csv(pairs_postprocessed_txt, header=None)
        train_filenames = [os.path.join(train_dir, filename)
                           for train_dir in train_dirs
                           for filename in os.listdir(os.path.join(output_dir, train_dir))
                           if os.path.join(train_dir, filename) in filenames_frame.values]
        val_filenames = [os.path.join(val_dir, filename)
                         for val_dir in val_dirs
                         for filename in os.listdir(os.path.join(output_dir, val_dir))
                         if os.path.join(val_dir, filename) in filenames_frame.values]
        test_filenames = [os.path.join(test_dir, filename)
                          for test_dir in test_dirs
                          for filename in os.listdir(os.path.join(output_dir, test_dir))
                          if os.path.join(test_dir, filename) in filenames_frame.values]
        # Create separate .txt files to describe the training list and validation list, respectively
        train_filenames_frame, val_filenames_frame = pd.DataFrame(train_filenames), pd.DataFrame(val_filenames)
        test_filenames_frame = pd.DataFrame(test_filenames)
        # Create separate .txt files to describe the training list and validation list, respectively
        train_filenames_frame.to_csv(pairs_postprocessed_train_txt, header=None, index=None, sep=' ', mode='a')
        val_filenames_frame.to_csv(pairs_postprocessed_val_txt, header=None, index=None, sep=' ', mode='a')
        test_filenames_frame.to_csv(pairs_postprocessed_test_txt, header=None, index=None, sep=' ', mode='a')


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
