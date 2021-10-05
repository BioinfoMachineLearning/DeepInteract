import logging
import os

import click
from project.utils.dips_plus_utils import collect_dataset_statistics, DEFAULT_DATASET_STATISTICS


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------


@click.command()
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
def main(output_dir: str):
    """Collect all dataset statistics."""
    logger = logging.getLogger(__name__)
    logger.info('Aggregating statistics for given dataset')

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

    # Aggregate dataset statistics in a readable fashion
    dataset_statistics = collect_dataset_statistics(output_dir)

    # Write out updated dataset statistics
    with open(dataset_statistics_csv, 'w') as f:
        for key in dataset_statistics.keys():
            f.write("%s, %s\n" % (key, dataset_statistics[key]))


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
