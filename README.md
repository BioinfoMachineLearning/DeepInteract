<div align="center">

# Source code for Geometric Transformers for Protein Interface Contact Prediction (ICLR 2022)

[![Paper](http://img.shields.io/badge/paper-arxiv.2110.02423-B31B1B.svg)](https://openreview.net/forum?id=CS4463zx6Hi) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6671582.svg)](https://doi.org/10.5281/zenodo.6671582)

[<img src="https://twixes.gallerycdn.vsassets.io/extensions/twixes/pypi-assistant/1.0.3/1589834023190/Microsoft.VisualStudio.Services.Icons.Default" width="50"/>](https://pypi.org/project/DeepInteract/)

![DeepInteract Architecture](https://github.com/BioinfoMachineLearning/DeepInteract/blob/main/img/DeepInteract_Architecture.png)

![Geometric Transformer](https://github.com/BioinfoMachineLearning/DeepInteract/blob/main/img/Geometric_Transformer.png)

</div>

## Description

A geometric deep learning pipeline for predicting protein interface contacts.

## Citing this work

If you use the code or data associated with this package, please cite:

```bibtex
@inproceedings{morehead2022geometric,
  title={Geometric Transformers for Protein Interface Contact Prediction},
  author={Alex Morehead and Chen Chen and Jianlin Cheng},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=CS4463zx6Hi}
}
```

## First time setup

The following step is required in order to run DeepInteract:

### Genetic databases

This step requires `aria2c` to be installed on your machine.

DeepInteract needs only one of the following genetic (sequence) databases compatible with HH-suite3 to run:

* [BFD (Requires ~1.7TB of Space When Unextracted)](https://bfd.mmseqs.com/)
* [Small BFD (Requires ~17GB of Space When Unextracted)](https://storage.googleapis.com/alphafold-databases/reduced_dbs/bfd-first_non_consensus_sequences.fasta.gz)  
* [Uniclust30 (Requires ~86GB of Space When Unextracted)](https://uniclust.mmseqs.com/)

#### Install the BFD for HH-suite3

```bash
# Following script originally from AlphaFold2 (https://github.com/deepmind/alphafold):
DOWNLOAD_DIR="~/Data/Databases"
ROOT_DIR="${DOWNLOAD_DIR}/bfd"
mkdir "~/Data" "$DOWNLOAD_DIR" "$ROOT_DIR"
# Mirror of:
# https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz.
SOURCE_URL="https://storage.googleapis.com/alphafold-databases/casp14_versions/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz"
BASENAME=$(basename "${SOURCE_URL}")

mkdir --parents "${ROOT_DIR}"
aria2c "${SOURCE_URL}" --dir="${ROOT_DIR}"
tar --extract --verbose --file="${ROOT_DIR}/${BASENAME}" \
  --directory="${ROOT_DIR}"
rm "${ROOT_DIR}/${BASENAME}"

# The CLI argument --hhsuite_db for lit_model_predict.py
# should then become '~/Data/Databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'
```

#### (Smaller Alternative) Install the Small BFD for HH-suite3

```bash
# Following script originally from AlphaFold2 (https://github.com/deepmind/alphafold):
DOWNLOAD_DIR="~/Data/Databases"
ROOT_DIR="${DOWNLOAD_DIR}/small_bfd"
mkdir "~/Data" "$DOWNLOAD_DIR" "$ROOT_DIR"
SOURCE_URL="https://storage.googleapis.com/alphafold-databases/reduced_dbs/bfd-first_non_consensus_sequences.fasta.gz"
BASENAME=$(basename "${SOURCE_URL}")

mkdir --parents "${ROOT_DIR}"
aria2c "${SOURCE_URL}" --dir="${ROOT_DIR}"
pushd "${ROOT_DIR}"
gunzip "${ROOT_DIR}/${BASENAME}"
popd

# The CLI argument --hhsuite_db for lit_model_predict.py
# should then become '~/Data/Databases/small_bfd/bfd-first_non_consensus_sequences.fasta'
```

#### (Smaller Alternative) Install Uniclust30 for HH-suite3

```bash
# Following script originally from AlphaFold2 (https://github.com/deepmind/alphafold):
DOWNLOAD_DIR="~/Data/Databases"
ROOT_DIR="${DOWNLOAD_DIR}/uniclust30"
mkdir "~/Data" "$DOWNLOAD_DIR" "$ROOT_DIR"
# Mirror of:
# http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz
SOURCE_URL="https://storage.googleapis.com/alphafold-databases/casp14_versions/uniclust30_2018_08_hhsuite.tar.gz"
BASENAME=$(basename "${SOURCE_URL}")

mkdir --parents "${ROOT_DIR}"
aria2c "${SOURCE_URL}" --dir="${ROOT_DIR}"
tar --extract --verbose --file="${ROOT_DIR}/${BASENAME}" \
  --directory="${ROOT_DIR}"
rm "${ROOT_DIR}/${BASENAME}"

# The CLI argument --hhsuite_db for lit_model_predict.py
# should then become '~/Data/Databases/uniclust30/uniclust30_2018_08/uniclust30_2018_08'
```

## Repository Directory Structure

```
DeepInteract
│
└───docker
│
└───img
│
└───project
     │
     └───checkpoints
     │
     └───datasets
     │   │
     │   └───builder
     │   │
     │   └───DB5
     │   │   │
     │   │   └───final
     │   │   │   │
     │   │   │   └───processed
     │   │   │   │
     │   │   │   └───raw
     │   │   │
     │   │   db5_dgl_data_module.py
     │   │   db5_dgl_dataset.py
     │   │
     │   └───CASP_CAPRI
     │   │   │
     │   │   └───final
     │   │   │   │
     │   │   │   └───processed
     │   │   │   │
     │   │   │   └───raw
     │   │   │
     │   │   casp_capri_dgl_data_module.py
     │   │   casp_capri_dgl_dataset.py
     │   │
     │   └───DIPS
     │   │   │
     │   │   └───final
     │   │   │   │
     │   │   │   └───processed
     │   │   │   │
     │   │   │   └───raw
     │   │   │
     │   │   dips_dgl_data_module.py
     │   │   dips_dgl_dataset.py
     │   │
     │   └───Input
     │   │   │
     │   │   └───final
     │   │   │   │
     │   │   │   └───processed
     │   │   │   │
     │   │   │   └───raw
     │   │   │
     │   │   └───interim
     │   │   │   │
     │   │   │   └───complexes
     │   │   │   │
     │   │   │   └───external_feats
     │   │   │   │   │
     │   │   │   │   └───PSAIA
     │   │   │   │       │
     │   │   │   │       └───INPUT
     │   │   │   │
     │   │   │   └───pairs
     │   │   │   │
     │   │   │   └───parsed
     │   │   │
     │   │   └───raw
     │   │
     │   └───PICP
     │       picp_dgl_data_module.py
     │
     └───test_data
     │
     └───utils
     │   deepinteract_constants.py
     │   deepinteract_modules.py
     │   deepinteract_utils.py
     │   dips_plus_utils.py
     │   graph_utils.py
     │   protein_feature_utils.py
     │   vision_modules.py
     │
     lit_model_predict.py
     lit_model_predict_docker.py
     lit_model_train.py
.gitignore
CONTRIBUTING.md
environment.yml
LICENSE
README.md
requirements.txt
setup.cfg
setup.py
```

## Running DeepInteract via Docker

**The simplest way to run DeepInteract is using the provided Docker script.**

The following steps are required in order to ensure Docker is installed and working correctly:

1. Install [Docker](https://www.docker.com/).
    * Install
        [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
        for GPU support.
    * Setup running
        [Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

2. Check that DeepInteract will be able to use a GPU by running:

    ```bash
    docker run --rm --gpus all nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04 nvidia-smi
    ```

    The output of this command should show a list of your GPUs. If it doesn't,
    check if you followed all steps correctly when setting up the
    [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
    or take a look at the following
    [NVIDIA Docker issue](https://github.com/NVIDIA/nvidia-docker/issues/1447#issuecomment-801479573).

Now that we know Docker is functioning properly, we can begin building our Docker image for DeepInteract:

1. Clone this repository and `cd` into it.

    ```bash
    git clone https://github.com/BioinfoMachineLearning/DeepInteract
    cd DeepInteract/
    DI_DIR=$(pwd)
    ```
   
2. Download our trained model checkpoints.

    ```bash
    mkdir -p project/checkpoints
    wget -P project/checkpoints https://zenodo.org/record/6671582/files/LitGINI-GeoTran-DilResNet.ckpt
    wget -P project/checkpoints https://zenodo.org/record/6671582/files/LitGINI-GeoTran-DilResNet-DB5-Fine-Tuned.ckpt
    ```

3. Build the Docker image (Warning: Requires ~13GB of Space):

    ```bash
    docker build -f docker/Dockerfile -t deepinteract .
    ```

4. Install the `run_docker.py` dependencies. Note: You may optionally wish to
    create a
    [Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html)
    to prevent conflicts with your system's Python environment.

    ```bash
    pip3 install -r docker/requirements.txt
    ```
   
5. Create directory in which to generate input features and outputs:

    ```bash
    mkdir -p project/datasets/Input
    ```

6. Run `run_docker.py` pointing to two input PDB files containing the first and second chains
    of a complex for which you wish to predict the contact probability map.
    For example, for the DIPS-Plus test target with the PDB ID `4HEQ`:

    ```bash
    python3 docker/run_docker.py --left_pdb_filepath "$DI_DIR"/project/test_data/4heq_l_u.pdb --right_pdb_filepath "$DI_DIR"/project/test_data/4heq_r_u.pdb --input_dataset_dir "$DI_DIR"/project/datasets/Input --ckpt_name "$DI_DIR"/project/checkpoints/LitGINI-GeoTran-DilResNet.ckpt --hhsuite_db ~/Data/Databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt --num_gpus 0
    ```
   
    This script will generate and (as NumPy array files - e.g., `test_data/4heq_contact_prob_map.npy`)
    save to the given input directory the predicted interface contact map as well as the Geometric
    Transformer's learned node and edge representations for both chain graphs.

7. Note that by using the default

    ```bash
    --num_gpus 0
    ```

    flag when executing `run_docker.py`, the Docker container will only 
    make use of the system's available CPU(s) for prediction. However,
    by specifying

    ```bash
    --num_gpus 1
    ```
   
    when executing `run_docker.py`, the Docker container will then
    employ the first available GPU for prediction.

## Running DeepInteract via a Traditional Installation (for Linux-Based Operating Systems)

First, install and configure Conda environment:

```bash
# Clone this repository:
git clone https://github.com/BioinfoMachineLearning/DeepInteract

# Change to project directory:
cd DeepInteract
DI_DIR=$(pwd)

# Set up Conda environment locally
conda env create --name DeepInteract -f environment.yml

# Activate Conda environment located in the current directory:
conda activate DeepInteract

# (Optional) Perform a full install of the pip dependencies described in 'requirements.txt':
pip3 install -r requirements.txt

# (Optional) To remove the long Conda environment prefix in your shell prompt, modify the env_prompt setting in your .condarc file with:
conda config --set env_prompt '({name})'
 ```

### Installing PSAIA

Install GCC 10 for PSAIA:

```bash
# Install GCC 10 for Ubuntu 20.04
sudo apt install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/ppa
sudo apt update
sudo apt install gcc-10 g++-10

# Or install GCC 10 for Arch Linux/Manjaro
yay -S gcc10
```

Install QT4 for PSAIA:

```bash
# Install QT4 for Ubuntu 20.04:
sudo add-apt-repository ppa:rock-core/qt4
sudo apt update
sudo apt install libqt4* libqtcore4 libqtgui4 libqtwebkit4 qt4* libxext-dev

# Or install QT4 for Arch Linux/Manjaro
yay -S qt4
```

Compile PSAIA from source:

```bash
# Select the location to install the software:
MY_LOCAL=~/Programs

# Download and extract PSAIA's source code:
mkdir "$MY_LOCAL"
cd "$MY_LOCAL"
wget http://complex.zesoi.fer.hr/data/PSAIA-1.0-source.tar.gz
tar -xvzf PSAIA-1.0-source.tar.gz

# Compile PSAIA (i.e., a GUI for PSA):
cd PSAIA_1.0_source/make/linux/psaia/
qmake-qt4 psaia.pro
make

# Compile PSA (i.e., the protein structure analysis (PSA) program):
cd ../psa/
qmake-qt4 psa.pro
make

# Compile PIA (i.e., the protein interaction analysis (PIA) program):
cd ../pia/
qmake-qt4 pia.pro
make

# Test run any of the above-compiled programs:
cd "$MY_LOCAL"/PSAIA_1.0_source/bin/linux
# Test run PSA inside a GUI:
./psaia/psaia
# Test run PIA through a terminal:
./pia/pia
# Test run PSA through a terminal:
./psa/psa
```

**Finally, substitute your absolute filepath for DeepInteract**
(i.e., where on your local storage device you downloaded the
repository to) **anywhere DeepInteract's local repository is
referenced in `project/datasets/builder/psaia_config_file_input.txt`.**

## Training

### Download training and cross-validation DGLGraphs

To train, fine-tune, or test DeepInteract models using CASP-CAPRI, DB5-Plus, or DIPS-Plus targets, we first need to download the preprocessed DGLGraphs from Zenodo:

```bash
# Download and extract preprocessed DGLGraphs for CASP-CAPRI, DB5-Plus, and DIPS-Plus
# Requires ~55GB of free space
# Download CASP-CAPRI
mkdir -p project/datasets/CASP_CAPRI/final
cd project/datasets/CASP_CAPRI/final
wget https://zenodo.org/record/6671582/files/final_raw_casp_capri.tar.gz
wget https://zenodo.org/record/6671582/files/final_processed_casp_capri.tar.gz

# Extract CASP-CAPRI
tar -xzf final_raw_casp_capri.tar.gz
tar -xzf final_processed_casp_capri.tar.gz
rm final_raw_casp_capri.tar.gz final_processed_casp_capri.tar.gz

# Download DB5-Plus
mkdir -p ../../DB5/final
cd ../../DB5/final
wget https://zenodo.org/record/6671582/files/final_raw_db5.tar.gz
wget https://zenodo.org/record/6671582/files/final_processed_db5.tar.gz

# Extract DB5-Plus
tar -xzf final_raw_db5.tar.gz
tar -xzf final_processed_db5.tar.gz
rm final_raw_db5.tar.gz final_processed_db5.tar.gz

# Download DIPS-Plus
mkdir -p ../../DIPS/final
cd ../../DIPS/final
wget https://zenodo.org/record/6671582/files/final_raw_dips.tar.gz
wget https://zenodo.org/record/6671582/files/final_processed_dips.tar.gz.partaa
wget https://zenodo.org/record/6671582/files/final_processed_dips.tar.gz.partab

# First, reassemble all processed DGLGraphs
# We split the (tar.gz) archive into two separate parts with
# 'split -b 4096M final_processed_dips.tar.gz "final_processed_dips.tar.gz.part"'
# to upload it to Zenodo, so to recover the original archive:
cat final_processed_dips.tar.gz.parta* >final_processed_dips.tar.gz

# Extract DIPS-Plus
tar -xzf final_raw_dips.tar.gz
tar -xzf final_processed_dips.tar.gz
rm final_processed_dips.tar.gz.parta* final_raw_dips.tar.gz final_processed_dips.tar.gz
```

Navigate to the project directory and run the training script with the parameters desired:

 ```bash
# Hint: Run `python3 lit_model_train.py --help` to see all available CLI arguments
cd project
python3 lit_model_train.py --lr 1e-3 --weight_decay 1e-2
cd ..
```

## Inference

### Download trained model checkpoints

```bash
# Return to root directory of DeepInteract repository
cd "$DI_DIR"

# Download our trained model checkpoints
mkdir -p project/checkpoints
wget -P project/checkpoints https://zenodo.org/record/6671582/files/LitGINI-GeoTran-DilResNet.ckpt
wget -P project/checkpoints https://zenodo.org/record/6671582/files/LitGINI-GeoTran-DilResNet-DB5-Fine-Tuned.ckpt
```

### Predict interface contact probability maps

Navigate to the project directory and run the prediction script
with the filenames of the left and right PDB chains.

 ```bash
 # Hint: Run `python3 lit_model_predict.py --help` to see all available CLI arguments
cd project
python3 lit_model_predict.py --left_pdb_filepath "$DI_DIR"/project/test_data/4heq_l_u.pdb --right_pdb_filepath "$DI_DIR"/project/test_data/4heq_r_u.pdb --ckpt_dir "$DI_DIR"/project/checkpoints --ckpt_name LitGINI-GeoTran-DilResNet.ckpt --hhsuite_db ~/Data/Databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt
cd ..
```

This script  will generate and (as NumPy array files - e.g., `test_data/4heq_contact_prob_map.npy`)
save to the given input directory the predicted interface contact map as well as the
Geometric Transformer's learned node and edge representations for both chain graphs.

## Acknowledgements

DeepInteract communicates with and/or references the following separate libraries
and packages:

* [Abseil](https://github.com/abseil/abseil-py)
* [Biopython](https://biopython.org)
* [Docker](https://www.docker.com)
* [HH Suite](https://github.com/soedinglab/hh-suite)
* [NumPy](https://numpy.org)
* [pytorch](https://github.com/pytorch/pytorch)
* [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [SciPy](https://scipy.org)
* [tqdm](https://github.com/tqdm/tqdm)

We thank all their contributors and maintainers!

## License and Disclaimer

Copyright 2021 University of Missouri-Columbia Bioinformatics & Machine Learning (BML) Lab.

### DeepInteract Code License

Licensed under the GNU Public License, Version 3.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at https://www.gnu.org/licenses/gpl-3.0.en.html.

### Third-party software

Use of the third-party software, libraries or code referred to in the
[Acknowledgements](#acknowledgements) section above may be governed by separate
terms and conditions or license provisions. Your use of the third-party
software, libraries or code is subject to any such terms and you should check
that you can comply with any applicable restrictions or terms and conditions
before use.
