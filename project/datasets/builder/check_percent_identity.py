import logging
import os

import click
import pandas as pd
from atom3.database import get_pdb_name, get_pdb_code
from parallel import submit_jobs
from project.utils.deepinteract_utils import check_percent_identity, construct_filenames_frame_txt_filenames


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------


@click.command()
@click.argument('input_dir', default='../DIPS/final/raw', type=click.Path())
@click.argument('compare_dir', default='../CASP_CAPRI/final/raw', type=click.Path())
@click.option('--mode', '-m', default='train', type=str)
@click.option('--percent_identity_threshold', '-p', default=0.3, type=float)
@click.option('--num_cpus', '-c', default=1)
def main(input_dir: str, compare_dir: str, mode: str, percent_identity_threshold: int, num_cpus: int):
    """Check percent identity of the given input dataset's chains and those of a comparison dataset."""
    logger = logging.getLogger(__name__)
    logger.info('Checking percent identity of the given input dataset\'s chains and those of a comparison dataset')

    # Get training, validation, or test filenames
    base_txt_filename, filenames_frame_txt_filename, filenames_frame_txt_filepath = \
        construct_filenames_frame_txt_filenames(mode, 1.0, False, input_dir)
    input_filenames = pd.read_csv(os.path.join(input_dir, filenames_frame_txt_filename), header=None).values

    base_txt_filename, filenames_frame_txt_filename, filenames_frame_txt_filepath = \
        construct_filenames_frame_txt_filenames('test', 1.0, False, input_dir)
    compare_filenames = pd.read_csv(os.path.join(compare_dir, filenames_frame_txt_filename), header=None).values

    # Get work filenames
    logger.info(f'Looking for all complexes in {input_dir}')
    requested_filenames = [os.path.join(input_dir, filename[0]) for filename in input_filenames]
    requested_keys = [get_pdb_name(x) for x in requested_filenames]
    work_keys = [key for key in requested_keys]
    work_filenames = [
        os.path.join(input_dir, get_pdb_code(work_key)[1:3], work_key + '.dill') for work_key in work_keys
    ]
    logger.info(f'Found {len(work_keys)} work complex(es) in {input_dir}')
    logger.info(f'Found {len(compare_filenames)} comparison complex(es) in {compare_dir}')

    # Process comparison filenames
    compare_filenames = [os.path.join(compare_dir, compare_filename[0]) for compare_filename in compare_filenames]

    # Collect thread inputs
    inputs = [(pair_filename, compare_filenames, percent_identity_threshold, logger)
              for pair_filename in work_filenames]
    submit_jobs(check_percent_identity, inputs, num_cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
