from typing import Optional

from project.datasets.CASP_CAPRI.casp_capri_dgl_dataset import CASPCAPRIDGLDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from project.utils.deepinteract_utils import dgl_picp_collate


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------


class CASPCAPRIDGLDataModule(LightningDataModule):
    """Bound protein complex data module for DGL with PyTorch."""

    # Dataset partition instantiations
    casp_capri_test = None

    def __init__(self, data_dir: str, batch_size: int, num_dataloader_workers: int, knn: int, self_loops: bool,
                 pn_ratio: float, percent_to_use: float, use_dgl: bool, process_complexes: bool, input_indep: bool):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.knn = knn
        self.self_loops = self_loops
        self.pn_ratio = pn_ratio
        self.percent_to_use = percent_to_use  # Fraction of CASP-CAPRI dataset splits to use
        self.use_dgl = use_dgl  # Whether to process each complex into a pair of DGL graphs for its final representation
        self.process_complexes = process_complexes  # Whether to process any unprocessed complexes before training
        self.input_indep = input_indep  # Whether to use an input-independent pipeline to train the model

    def setup(self, stage: Optional[str] = None):
        # Assign testing dataset for use in DataLoaders - called on every GPU
        self.casp_capri_test = CASPCAPRIDGLDataset(mode='test', raw_dir=self.data_dir, knn=self.knn,
                                                   self_loops=self.self_loops, pn_ratio=self.pn_ratio,
                                                   percent_to_use=self.percent_to_use, use_dgl=self.use_dgl,
                                                   process_complexes=self.process_complexes,
                                                   input_indep=self.input_indep)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.casp_capri_test, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_dataloader_workers, collate_fn=dgl_picp_collate, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.casp_capri_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=dgl_picp_collate, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.casp_capri_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=dgl_picp_collate, pin_memory=True)
