import logging
import os

import click
import pandas as pd
from atom3.database import get_structures_filenames, get_pdb_name, get_pdb_code
from parallel import submit_jobs
from project.utils.deepinteract_utils import process_complex_into_dict, construct_filenames_frame_txt_filenames


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------


@click.command()
@click.argument('input_dir', default='../DIPS/final/raw', type=click.Path())
@click.argument('output_dir', default='../DIPS/final/processed', type=click.Path())
@click.option('--source_type', '-t', default='rcsb', type=click.Choice(['rcsb', 'casp_capri']))
@click.option('--knn', '-k', default=20, type=int)
@click.option('--geo_nbrhd_size', '-n', default=2, type=int)
@click.option('--self_loops', '-s', default=True, type=bool)
@click.option('--mode', '-m', default='train', type=str)
@click.option('--check_sequence', '-v', default=False, type=bool)
@click.option('--percent_to_use', '-p', default=1.00)
@click.option('--num_cpus', '-c', default=1)
def main(input_dir: str, output_dir: str, source_type: str, knn: int, geo_nbrhd_size: int, self_loops: bool,
         mode: str, check_sequence: bool, percent_to_use: float, num_cpus: int):
    """Process complexes into dictionaries ready for training, validation, and testing."""
    logger = logging.getLogger(__name__)
    logger.info('Processing complexes into dictionaries for the given dataset')

    # Make sure the output_dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get training, validation, or test filenames
    filename_sampling = 0.0 < percent_to_use < 1.0
    base_txt_filename, filenames_frame_txt_filename, filenames_frame_txt_filepath = \
        construct_filenames_frame_txt_filenames(mode, percent_to_use, filename_sampling, input_dir)
    filenames = pd.read_csv(os.path.join(input_dir, filenames_frame_txt_filename), header=None).values

    # Get work filenames
    ext = '.dill' if source_type.lower() in ['rcsb', 'casp_capri'] else ''
    logger.info(f'Looking for all complexes in {input_dir}')
    requested_filenames = [os.path.join(input_dir, filename[0]) for filename in filenames]
    requested_keys = [get_pdb_name(x) for x in requested_filenames]
    produced_filenames = get_structures_filenames(output_dir, extension='.dill')
    produced_keys = [get_pdb_name(x) for x in produced_filenames]
    work_keys = [key for key in requested_keys if key not in produced_keys]
    work_filenames = [
        os.path.join(input_dir, get_pdb_code(work_key)[1:3], work_key + ext) for work_key in work_keys
    ]
    logger.info(f'Found {len(work_keys)} work complex(es) in {input_dir}')

    # Collect thread inputs
    inputs = []
    for pair_filename in work_filenames:
        pair_filename_splits = pair_filename.split(os.sep)
        trimmed_filename = os.path.join(pair_filename_splits[4], pair_filename_splits[5])
        if trimmed_filename in filenames:
            input_filename = pair_filename
            output_filename = os.path.join(output_dir, trimmed_filename)
            inputs.append((input_filename, output_filename, knn, geo_nbrhd_size, self_loops, check_sequence))
    submit_jobs(process_complex_into_dict, inputs, num_cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
