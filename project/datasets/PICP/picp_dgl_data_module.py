from typing import Optional

from project.datasets.CASP_CAPRI.casp_capri_dgl_dataset import CASPCAPRIDGLDataset
from project.datasets.DIPS.dips_dgl_dataset import DIPSDGLDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from project.utils.deepinteract_utils import dgl_picp_collate


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------


class PICPDGLDataModule(LightningDataModule):
    """Combined protein complex data module for DGL with PyTorch."""

    # Dataset partition instantiations
    dips_train = None
    dips_val = None
    dips_val_viz = None
    dips_test = None
    casp_capri_test = None

    def __init__(self, dips_data_dir: str, casp_capri_data_dir: str, batch_size: int, num_dataloader_workers: int,
                 knn: int, self_loops: bool, pn_ratio: float, dips_percent_to_use: float,
                 casp_capri_percent_to_use: float, testing_with_casp_capri: bool, use_dgl: bool,
                 process_complexes: bool, input_indep: bool):
        super().__init__()

        self.dips_data_dir = dips_data_dir
        self.casp_capri_data_dir = casp_capri_data_dir
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        # How many edges to allow per node in each graph (e.g. 5 nearest-neighbor edges per node)
        self.knn = knn
        # Whether to allow node self-loops
        self.self_loops = self_loops
        # Positive-negative class sampling ratio to use during training with DIPS
        self.pn_ratio = pn_ratio
        # Fraction of DIPS dataset splits to use
        self.dips_percent_to_use = dips_percent_to_use
        # Fraction of CASP-CAPRI dataset split to use
        self.casp_capri_percent_to_use = casp_capri_percent_to_use
        # Whether to test on the 13th and 14th CASP-CAPRI's dataset of homo and heterodimers
        self.testing_with_casp_capri = testing_with_casp_capri
        # Whether to process each complex into a pair of DGL graphs for its final representation
        self.use_dgl = use_dgl
        # Whether to process any unprocessed complexes before training
        self.process_complexes = process_complexes
        # Whether to use an input-independent pipeline to train the model, to see if the input data is actually helpful
        self.input_indep = input_indep
        # Which collation function to use
        self.collate_fn = dgl_picp_collate if self.use_dgl else lambda x: x

    def setup(self, stage: Optional[str] = None):
        # Assign training/validation/testing data set for use in DataLoaders - called on every GPU
        self.dips_train = DIPSDGLDataset(mode='train', raw_dir=self.dips_data_dir, knn=self.knn,
                                         self_loops=self.self_loops, pn_ratio=self.pn_ratio,
                                         percent_to_use=self.dips_percent_to_use, use_dgl=self.use_dgl,
                                         process_complexes=self.process_complexes, input_indep=self.input_indep)
        self.dips_val = DIPSDGLDataset(mode='val', raw_dir=self.dips_data_dir, knn=self.knn,
                                       self_loops=self.self_loops, pn_ratio=self.pn_ratio,
                                       percent_to_use=self.dips_percent_to_use, use_dgl=self.use_dgl,
                                       process_complexes=self.process_complexes, input_indep=self.input_indep)
        self.dips_val_viz = DIPSDGLDataset(mode='val', raw_dir=self.dips_data_dir, knn=self.knn,
                                           self_loops=self.self_loops, pn_ratio=self.pn_ratio,
                                           percent_to_use=self.dips_percent_to_use, use_dgl=self.use_dgl,
                                           process_complexes=self.process_complexes, input_indep=self.input_indep,
                                           train_viz=True)
        if self.testing_with_casp_capri:
            self.casp_capri_test = CASPCAPRIDGLDataset(mode='test', raw_dir=self.casp_capri_data_dir, knn=self.knn,
                                                       self_loops=self.self_loops,
                                                       percent_to_use=self.casp_capri_percent_to_use,
                                                       use_dgl=self.use_dgl, process_complexes=self.process_complexes,
                                                       input_indep=self.input_indep)
        else:  # Default to testing with DIPS-Plus
            self.dips_test = DIPSDGLDataset(mode='test', raw_dir=self.dips_data_dir, knn=self.knn,
                                            self_loops=self.self_loops, pn_ratio=self.pn_ratio,
                                            percent_to_use=self.dips_percent_to_use, use_dgl=self.use_dgl,
                                            process_complexes=self.process_complexes, input_indep=self.input_indep)

    def train_dataloader(self) -> DataLoader:
        # Ascertain which training dataset and batch size to use
        train_dataset = self.dips_train
        train_batch_size = self.batch_size

        # Curate data loader for training data
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                      num_workers=self.num_dataloader_workers, collate_fn=self.collate_fn,
                                      pin_memory=True, drop_last=True)  # drop_last=True to keep loss smooth each epoch

        # Curate dataset and data loader for validation data to be used for model inspection during training
        val_dataset = self.dips_val_viz
        val_dataloader = DataLoader(val_dataset, batch_size=1,
                                    shuffle=False, num_workers=1,
                                    collate_fn=self.collate_fn, drop_last=True)

        return {'train_batch': train_dataloader, 'val_batch': val_dataloader}

    def val_dataloader(self) -> DataLoader:
        # Ascertain which validation dataset and batch size to use
        val_dataset = self.dips_val
        val_batch_size = self.batch_size
        return DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=self.collate_fn,
                          pin_memory=True, drop_last=True)

    def test_dataloader(self) -> DataLoader:
        if self.testing_with_casp_capri:
            test_dataset = self.casp_capri_test
            test_batch_size = 1
        else:
            test_dataset = self.dips_test
            test_batch_size = 1
        return DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=self.collate_fn, pin_memory=True)
