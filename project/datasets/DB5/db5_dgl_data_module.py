from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from project.datasets.DB5.db5_dgl_dataset import DB5DGLDataset
from project.utils.deepinteract_utils import dgl_picp_collate


class DB5DGLDataModule(LightningDataModule):
    """Unbound protein complex data module for DGL with PyTorch."""

    # Dataset partition instantiations
    db5_train = None
    db5_val = None
    db5_test = None

    def __init__(self, data_dir: str, batch_size: int, num_dataloader_workers: int, knn: int,
                 self_loops: bool, percent_to_use: float, process_complexes: bool, input_indep: bool):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.knn = knn
        self.self_loops = self_loops
        self.percent_to_use = percent_to_use  # Fraction of DB5 dataset splits to use
        self.process_complexes = process_complexes  # Whether to process any unprocessed complexes before training
        self.input_indep = input_indep  # Whether to use an input-independent pipeline to train the model
        self.collate_fn = dgl_picp_collate  # Which collation function to use

    def setup(self, stage: Optional[str] = None):
        # Assign training/validation/testing dataset for use in DataLoaders - called on every GPU
        self.db5_train = DB5DGLDataset(mode='train', raw_dir=self.data_dir, knn=self.knn,
                                       geo_nbrhd_size=2, self_loops=self.self_loops,
                                       percent_to_use=self.percent_to_use,
                                       process_complexes=self.process_complexes,
                                       input_indep=self.input_indep)
        self.db5_val = DB5DGLDataset(mode='val', raw_dir=self.data_dir, knn=self.knn,
                                     geo_nbrhd_size=2, self_loops=self.self_loops,
                                     percent_to_use=self.percent_to_use,
                                     process_complexes=self.process_complexes,
                                     input_indep=self.input_indep)
        self.db5_test = DB5DGLDataset(mode='test', raw_dir=self.data_dir, knn=self.knn,
                                      geo_nbrhd_size=2, self_loops=self.self_loops,
                                      percent_to_use=self.percent_to_use,
                                      process_complexes=self.process_complexes,
                                      input_indep=self.input_indep)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.db5_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_dataloader_workers, collate_fn=self.collate_fn, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.db5_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=self.collate_fn, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.db5_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_dataloader_workers, collate_fn=self.collate_fn, pin_memory=True)
