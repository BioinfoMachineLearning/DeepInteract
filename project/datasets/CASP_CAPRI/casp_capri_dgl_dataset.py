import logging
import os
import pickle

import pandas as pd
from dgl.data import DGLDataset, download, check_sha1

from project.utils.deepinteract_utils \
    import construct_filenames_frame_txt_filenames, build_filenames_frame_error_message, process_complex_into_dict, \
    zero_out_complex_features


class CASPCAPRIDGLDataset(DGLDataset):
    r"""Bound protein complex dataset for DGL with PyTorch.

    Statistics:

    - Test homodimers: 14
    - Test heterodimers: 5
    - Number of structures per complex: 2
    ----------------------
    - Total dimers: 19
    ----------------------

    Parameters
    ----------
    mode: str, optional
        Should be one of ['test']. Default: 'test'.
    raw_dir: str
        Raw file directory to download/contains the input data directory. Default: 'final/raw'.
    knn: int
        How many nearest neighbors to which to connect a given node. Default: 20.
    geo_nbrhd_size: int
        Size of each edge's neighborhood when updating geometric edge features. Default: 2.
    self_loops: bool
        Whether to connect a given node to itself. Default: True.
    percent_to_use: float
        How much of the dataset to load. Default: 1.00.
    process_complexes: bool
        Whether to process each unprocessed complex as we load in the dataset. Default: True.
    input_indep: bool
        Whether to zero-out each input node and edge feature for an input-independent baseline. Default: False.
    force_reload: bool
        Whether to reload the dataset. Default: False.
    verbose: bool
        Whether to print out progress information. Default: False.

    Notes
    -----
    All the samples will be loaded and preprocessed in the memory first.

    Examples
    --------
    >>> # Get dataset
    >>> test_data = CASPCAPRIDGLDataset(mode='test')
    >>>
    >>> len(test_data)
    19
    >>> test_data.num_chains
    2
    """

    def __init__(self,
                 mode='test',
                 raw_dir=f'final{os.sep}raw',
                 knn=20,
                 geo_nbrhd_size=2,
                 self_loops=True,
                 percent_to_use=1.00,
                 process_complexes=True,
                 input_indep=False,
                 force_reload=False,
                 verbose=False):
        assert mode in ['test']
        assert 0.0 < percent_to_use <= 1.0
        self.mode = mode
        self.root = raw_dir
        self.knn = knn
        self.geo_nbrhd_size = geo_nbrhd_size
        self.self_loops = self_loops
        self.percent_to_use = percent_to_use  # How much of the dataset (e.g. CASP-CAPRI training dataset) to use
        self.process_complexes = process_complexes  # Whether to process any unprocessed complexes before training
        self.input_indep = input_indep  # Whether to use an input-independent pipeline to train the model
        self.final_dir = os.path.join(*self.root.split(os.sep)[:-1])
        self.processed_dir = os.path.join(self.final_dir, 'processed')

        self.filename_sampling = 0.0 < self.percent_to_use < 1.0
        self.base_txt_filename, self.filenames_frame_txt_filename, self.filenames_frame_txt_filepath = \
            construct_filenames_frame_txt_filenames(self.mode, self.percent_to_use, self.filename_sampling, self.root)

        # Try to load the text file containing all CASP-CAPRI filenames, and alert the user if it is missing
        filenames_frame_to_be_written = not os.path.exists(self.filenames_frame_txt_filepath)

        # Randomly sample DataFrame of filenames with requested cross validation ratio
        if self.filename_sampling:
            if filenames_frame_to_be_written:
                try:
                    self.filenames_frame = pd.read_csv(
                        os.path.join(self.root, self.base_txt_filename + '.txt'), header=None)
                except Exception:
                    raise FileNotFoundError(
                        build_filenames_frame_error_message('CASP-CAPRI', 'load', self.filenames_frame_txt_filepath))
                self.filenames_frame = self.filenames_frame.sample(frac=self.percent_to_use).reset_index()
                try:
                    self.filenames_frame[0].to_csv(self.filenames_frame_txt_filepath, header=None, index=None)
                except Exception:
                    raise Exception(
                        build_filenames_frame_error_message('CASP-CAPRI', 'write', self.filenames_frame_txt_filepath))

        # Load in existing DataFrame of filenames as requested (or if a sampled DataFrame .txt has already been written)
        if not filenames_frame_to_be_written:
            try:
                self.filenames_frame = pd.read_csv(self.filenames_frame_txt_filepath, header=None)
            except Exception:
                raise FileNotFoundError(
                    build_filenames_frame_error_message('CASP-CAPRI', 'load', self.filenames_frame_txt_filepath))

        super(CASPCAPRIDGLDataset, self).__init__(name='CASP-CAPRI-Plus',
                                                  raw_dir=raw_dir,
                                                  force_reload=force_reload,
                                                  verbose=verbose)
        logging.info(f"Loaded CASP-CAPRI-Plus {mode}-set, source: {self.processed_dir}, length: {len(self)}")

    def download(self):
        """Download and extract a pre-packaged version of the raw pairs if 'self.raw_dir' is not already populated."""
        # Path to store the file
        gz_file_path = os.path.join(os.path.join(*self.raw_dir.split(os.sep)[:-1]), 'final_raw_casp_capri.tar.gz')

        # Download file
        download(self.url, path=gz_file_path)

        # Check SHA-1
        if not check_sha1(gz_file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(gz_file_path))

        # Remove existing raw directory to make way for the new archive to be extracted
        if os.path.exists(self.raw_dir):
            os.removedirs(self.raw_dir)

        # Extract archive to parent directory of `self.raw_dir`
        self._extract_gz(gz_file_path, os.path.join(*self.raw_dir.split(os.sep)[:-1]))

    def process(self):
        """Process each protein complex into a testing-ready dictionary representing both structures."""
        if self.process_complexes:
            # Ensure the directory of processed complexes is already created
            os.makedirs(self.processed_dir, exist_ok=True)
            # Process each unprocessed protein complex
            for (i, raw_path) in self.filenames_frame.iterrows():
                raw_filepath = os.path.join(self.raw_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
                processed_filepath = os.path.join(self.processed_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
                if not os.path.exists(processed_filepath):
                    processed_parent_dir_to_make = os.path.join(self.processed_dir, os.path.split(raw_path[0])[0])
                    os.makedirs(processed_parent_dir_to_make, exist_ok=True)
                    process_complex_into_dict(raw_filepath, processed_filepath, self.knn,
                                              self.geo_nbrhd_size, self.self_loops, check_sequence=False)

    def has_cache(self):
        """Check if each complex is downloaded and available for testing."""
        for (i, raw_path) in self.filenames_frame.iterrows():
            processed_filepath = os.path.join(self.processed_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
            if not os.path.exists(processed_filepath):
                logging.info(
                    f'Unable to load at least one processed CASP-CAPRI pair. '
                    f'Please make sure all processed pairs have been successfully downloaded and are not corrupted.')
                raise FileNotFoundError
        logging.info('CASP-CAPRI cache found')  # Otherwise, a cache was found!

    def __getitem__(self, idx):
        r""" Get feature dictionary by index of complex.

        Parameters
        ----------
        idx: int

        Returns
        -------
        :class:`dict`

    - ``complex['graph1']:`` DGLGraph (of length M) containing each of the first graph's encoded node and edge features
    - ``complex['graph2']:`` DGLGraph (of length N) containing each of the second graph's encoded node and edge features
    - ``complex['examples']:`` PyTorch Tensor (of shape (M x N) x 3) containing the labels for inter-graph node pairs
    - ``complex['complex']:`` Python string describing the complex's code and original pdb filename
    - ``complex['filepath']:`` Python string describing the complex's filepath
        """
        # Assemble filepath of processed protein complex
        complex_filepath = f'{os.path.splitext(self.filenames_frame[0][idx])[0]}.dill'
        processed_filepath = os.path.join(self.processed_dir, complex_filepath)

        # Load in processed complex
        with open(processed_filepath, 'rb') as f:
            processed_complex = pickle.load(f)

        # Optionally zero-out input data for an input-independent pipeline (per Karpathy's suggestion)
        if self.input_indep:
            processed_complex = zero_out_complex_features(processed_complex)
        processed_complex['filepath'] = complex_filepath  # Add filepath to each complex dictionary

        # Manually filter for desired node and edge features
        # n_feat_idx_1, n_feat_idx_2 = 43, 85  # HSAAC
        # processed_complex['graph1'].ndata['f'] = processed_complex['graph1'].ndata['f'][:, n_feat_idx_1: n_feat_idx_2]
        # processed_complex['graph2'].ndata['f'] = processed_complex['graph2'].ndata['f'][:, n_feat_idx_1: n_feat_idx_2]

        # g1_rsa = processed_complex['graph1'].ndata['f'][:, 35: 36].reshape(-1, 1)  # RSA
        # g1_psaia = processed_complex['graph1'].ndata['f'][:, 37: 43]  # PSAIA
        # g1_hsaac = processed_complex['graph1'].ndata['f'][:, 43: 85]  # HSAAC
        # processed_complex['graph1'].ndata['f'] = torch.cat((g1_rsa, g1_psaia, g1_hsaac), dim=1)
        #
        # g2_rsa = processed_complex['graph2'].ndata['f'][:, 35: 36].reshape(-1, 1)  # RSA
        # g2_psaia = processed_complex['graph2'].ndata['f'][:, 37: 43]  # PSAIA
        # g2_hsaac = processed_complex['graph2'].ndata['f'][:, 43: 85]  # HSAAC
        # processed_complex['graph2'].ndata['f'] = torch.cat((g2_rsa, g2_psaia, g2_hsaac), dim=1)

        # processed_complex['graph1'].edata['f'] = processed_complex['graph1'].edata['f'][:, 1].reshape(-1, 1)
        # processed_complex['graph2'].edata['f'] = processed_complex['graph2'].edata['f'][:, 1].reshape(-1, 1)

        # Return requested complex to DataLoader
        return processed_complex

    def __len__(self) -> int:
        r"""Number of graph batches in the dataset."""
        return len(self.filenames_frame)

    @property
    def num_chains(self) -> int:
        """Number of protein chains in each complex."""
        return 2

    @property
    def num_classes(self) -> int:
        """Number of possible classes for each inter-chain residue pair in each complex."""
        return 2

    @property
    def num_node_features(self) -> int:
        """Number of node feature values after encoding them."""
        return 113

    @property
    def num_edge_features(self) -> int:
        """Number of edge feature values after encoding them."""
        return 27

    @property
    def raw_path(self) -> str:
        """Directory in which to locate raw pairs."""
        return self.raw_dir

    @property
    def url(self) -> str:
        """URL with which to download TAR archive of preprocessed pairs."""
        return 'https://zenodo.org/record/6299835/files/final_processed_casp_capri.tar.gz'
