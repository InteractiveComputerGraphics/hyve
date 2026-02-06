import os
import json
from typing import Optional
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from hyve.dataset.dynamic_dataset import DynamicDataset
from torch_geometric.loader import DataLoader  # type: ignore
import pytorch_lightning as pl
from tqdm import tqdm


class ShapeNetV2(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str, 
                 train_split: str, 
                 test_split: str, 
                 batch_size: int = 16, 
                 batch_size_test: int = 4, 
                 num_workers: int = 6,
                 knn_instead_of_mesh: Optional[int] = None,
                 subsample: Optional[int] = None
                 ) -> None:

        super().__init__() 
        self.data_dir = data_dir 
        self.train_split = train_split
        self.test_split = test_split

        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers

        self.knn_instead_of_mesh = knn_instead_of_mesh
        self.subsample = subsample

        with open(train_split, 'r') as f:
            train_dict = json.load(f)
            train_files_list = [os.path.join(self.data_dir, class_id, f"{file}.pt") for class_id in train_dict for file in train_dict[class_id]]
            self.train_files_list = [file for file in train_files_list if os.path.isfile(file)]

        with open(test_split, 'r') as f:
            test_dict = json.load(f)
            test_files_list = [os.path.join(self.data_dir, class_id, f"{file}.pt") for class_id in test_dict for file in test_dict[class_id]]
            self.test_files_list = [file for file in test_files_list if os.path.isfile(file)]
        
        print(f"Found {len(self.train_files_list)} train files\nFound {len(self.test_files_list)} test files.")

        self.save_hyperparameters()
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(DynamicDataset(self.train_files_list, knn_instead_of_mesh=self.knn_instead_of_mesh,subsample=self.subsample), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(DynamicDataset(self.test_files_list, knn_instead_of_mesh=self.knn_instead_of_mesh, subsample=self.subsample), batch_size=self.batch_size_test, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(DynamicDataset(self.test_files_list, knn_instead_of_mesh=self.knn_instead_of_mesh), batch_size=self.batch_size_test, num_workers=self.num_workers, pin_memory=True)



if __name__ == "__main__":
    data = ShapeNetV2("/clusterstorage/sjeske/data/ShapeNetCore.V2_processed/", "data/configs/shapenetv2/splits/planes_train.json", "data/configs/shapenetv2/splits/planes_test.json")

    train_data = data.train_dataloader()
    val_data = data.val_dataloader()

    for data in tqdm(train_data):
        pass

    for data in tqdm(val_data):
        pass