from .dynamic_dataset import DynamicDataset
from .ipc import IPC
from .shapenetv2 import ShapeNetV2
from .thingi10k import Thingi10k
from .objaverse import Objaverse
from .scannet import ScanNet
from .dFaust import DFaust
from .single import Single
from .sequence import Sequence

model_dependent_dataset_dict = {
}

dataset_dict = {
    "dynamic": DynamicDataset,
    "ipc": IPC,
	'shapenetv2': ShapeNetV2,
    'thingi10k': Thingi10k,
    'objaverse': Objaverse,
    'scannet': ScanNet,
    'dfaust': DFaust,
    'single': Single,
    'sequence': Sequence,
}