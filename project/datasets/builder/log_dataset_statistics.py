import logging
import os

import click
from project.utils.dips_plus_utils import log_dataset_statistics, DEFAULT_DATASET_STATISTICS


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------

@click.command()
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
def main(output_dir: str):
    """Log all collected dataset statistics."""
    logger = logging.getLogger(__name__)

    # Make sure the output_dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Create dataset statistics CSV if not already existent
    dataset_statistics_csv = os.path.join(output_dir, 'dataset_statistics.csv')
    if not os.path.exists(dataset_statistics_csv):
        # Reset dataset statistics CSV
        with open(dataset_statistics_csv, 'w') as f:
            for key in DEFAULT_DATASET_STATISTICS.keys():
                f.write("%s, %s\n" % (key, DEFAULT_DATASET_STATISTICS[key]))

    with open(dataset_statistics_csv, 'r') as f:
        # Read-in existing dataset statistics
        dataset_statistics = {}
        for line in f.readlines():
            dataset_statistics[line.split(',')[0].strip()] = int(line.split(',')[1].strip())

    # Log dataset statistics in a readable fashion
    if dataset_statistics is not None:
        log_dataset_statistics(logger, dataset_statistics)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
