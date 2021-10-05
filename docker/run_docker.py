# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from AlphaFold (https://github.com/deepmind/alphafold):
# -------------------------------------------------------------------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2021 University of Missouri-Columbia Bioinformatics & Machine Learning (BML) Lab.

"""Docker launch script for DeepInteract docker image."""

import os
import signal
from pathlib import Path
from typing import Tuple

import docker
from absl import app
from absl import flags
from absl import logging
from docker import types

flags.DEFINE_bool('use_gpu', True, 'Enable NVIDIA runtime to run with GPUs.')
flags.DEFINE_string('gpu_devices', 'all', 'Comma separated list of devices to '
                                          'pass to NVIDIA_VISIBLE_DEVICES.')
flags.DEFINE_string('left_pdb_filepath', None, 'A filepath to the left input PDB chain.')
flags.DEFINE_string('right_pdb_filepath', None, 'A filepath to the right input PDB chain.')
flags.DEFINE_string('input_dataset_dir', None, 'Directory in which to store generated features and outputs for inputs.')
flags.DEFINE_string('ckpt_name', None, 'Directory from which to load checkpoints.')
flags.DEFINE_string('psaia_dir', '/home/Programs/PSAIA_1.0_source/bin/linux/psa',
                    'Path to locally-compiled copy of PSAIA (i.e., to PSA, one of its CLIs)')
flags.DEFINE_string('psaia_config', '/app/DeepInteract/project/datasets/builder/psaia_config_file_input_docker.txt',
                    'Path to input config file for PSAIA')
flags.DEFINE_string('hhsuite_db', None, 'Path to downloaded and extracted HH-suite3-compatible database'
                                        ' (e.g., BFD or Uniclust30')
flags.DEFINE_integer('num_gpus', 0, 'How many GPUs to use to make a prediction (num_gpus=0 means use CPU instead)')
flags.DEFINE_string('docker_image_name', 'deepinteract',
                    'Name of DeepInteract docker image.')

FLAGS = flags.FLAGS

_ROOT_MOUNT_DIRECTORY = '/mnt/'


def _create_mount(mount_name: str, path: str, execute=True,
                  type='bind', read_only=True, same_level=False) -> Tuple[types.Mount, str]:
    path = os.path.abspath(path)
    source_path = path if same_level else os.path.dirname(path)
    target_path = os.path.join(_ROOT_MOUNT_DIRECTORY, mount_name)
    logging.info('Mounting %s -> %s', source_path, target_path)
    mount = types.Mount(target_path, source_path, type=type, read_only=read_only) if execute else None
    return mount, os.path.join(target_path, os.path.basename(path))


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    #### USER CONFIGURATION ####

    # Paths to input PDB chains for a given complex.
    left_pdb_filepath = Path(FLAGS.left_pdb_filepath).absolute().as_posix()
    right_pdb_filepath = Path(FLAGS.right_pdb_filepath).absolute().as_posix()

    # Path to directory for storing generated features and outputs for input complexes.
    input_dataset_dir = Path(FLAGS.input_dataset_dir).absolute().as_posix()

    # Path to directory containing trained models (i.e., PyTorch LightningModule checkpoints).
    ckpt_filepath = Path(FLAGS.ckpt_name).absolute().as_posix()

    # Path to HH-suite3-compatible database files.
    hhsuite_db = Path(FLAGS.hhsuite_db).absolute().as_posix()

    # Number of GPUs to use for predictions (num_gpus=0 means use CPU instead)
    num_gpus = FLAGS.num_gpus

    #### END OF USER CONFIGURATION ####

    mounts = []
    command_args = []

    # Mount each PDB complex's chain filepaths as a unique target directory.
    mount, target_path = _create_mount(f'input_pdbs', left_pdb_filepath, read_only=False)
    mounts.append(mount)
    command_args.append(f'--left_pdb_filepath={target_path}')

    _, target_path = _create_mount(f'input_pdbs', right_pdb_filepath, execute=False, read_only=False)
    command_args.append(f'--right_pdb_filepath={target_path}')

    # Mount directory for storing generated features and outputs for the input chains
    mount, target_path = _create_mount('Input', input_dataset_dir, same_level=True, read_only=False)
    mounts.append(mount)
    command_args.append(f'--input_dataset_dir={os.path.dirname(target_path)}')

    # Mount directory for storing requested checkpoint to be used for prediction
    mount, target_path = _create_mount('checkpoints', ckpt_filepath)
    mounts.append(mount)
    command_args.append(f'--ckpt_dir={os.path.dirname(target_path)}')
    command_args.append(f'--ckpt_name={os.path.basename(target_path)}')

    # Mount directory for storing requested HH-suite3-compatible database for searches
    mount, target_path = _create_mount('hhsuite_db', hhsuite_db)
    mounts.append(mount)
    command_args.append(f'--hhsuite_db={target_path}')

    # Set number of GPUs to use for predictions
    command_args.append(f'--num_gpus={num_gpus}')

    client = docker.from_env()
    container = client.containers.run(
        image=FLAGS.docker_image_name,
        command=command_args,
        runtime='nvidia' if FLAGS.use_gpu else None,
        remove=True,
        detach=True,
        mounts=mounts,
        environment={
            'NVIDIA_VISIBLE_DEVICES': FLAGS.gpu_devices,
        })

    # Add signal handler to ensure CTRL+C also stops the running container.
    signal.signal(signal.SIGINT,
                  lambda unused_sig, unused_frame: container.kill())

    for line in container.logs(stream=True):
        logging.info(line.strip().decode('utf-8'))


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'left_pdb_filepath',
        'right_pdb_filepath',
        'input_dataset_dir',
        'ckpt_name',
        'hhsuite_db'
    ])
    app.run(main)
