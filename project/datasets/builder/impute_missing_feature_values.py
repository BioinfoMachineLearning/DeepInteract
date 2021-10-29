import logging
import os
from pathlib import Path

import click
from parallel import submit_jobs
from project.utils.dips_plus_utils import impute_postprocessed_missing_feature_values


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------

@click.command()
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
@click.option('--impute_atom_features', '-a', default=False)
@click.option('--advanced_logging', '-l', default=False)
@click.option('--num_cpus', '-c', default=1)
def main(output_dir: str, impute_atom_features: bool, advanced_logging: bool, num_cpus: int):
    """Impute missing feature values."""
    logger = logging.getLogger(__name__)
    logger.info('Imputing missing feature values for given dataset')

    # Make sure the output_dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Collect thread inputs
    inputs = [(pair_filename.as_posix(), pair_filename.as_posix(), impute_atom_features, advanced_logging)
              for pair_filename in Path(output_dir).rglob('*.dill')]
    # Without impute_atom_features set to True, non-CA atoms will be filtered out after writing updated pairs
    submit_jobs(impute_postprocessed_missing_feature_values, inputs, num_cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
