from numpy import ndarray
from torch_geometric.data import Data
from torch import Tensor
from typing import TypedDict, get_args, get_type_hints, List, Dict, Literal

class MetaData(TypedDict):
    raw: Dict[Literal['train', 'val', 'test'], str]
    processed: Dict[Literal['train', 'val', 'test'], str]

class ProcessedDataItem(TypedDict):
    # Data containing both surface and volume samples as input (deprecated and not used anymore)
    volume: Data

    # Data containing only surface samples as input
    surface: Data

    # Unique sample identifier
    id: int

class GeometricDataItem(TypedDict):
    # Positions (x, y, z) of surface (or volume) point samples that should be encoded
    pos: Tensor

    # Point normals for input samples (only used for when specified)
    x: Tensor

    # Data for pre-sampled sdf sample points
    # y[0:3] = sample positions (x, y, z)
    # y[3] = sdf value at sample position (although this may be erroneous due to sampling process)
    # y[4:7] = normal at sample position
    y: Tensor

    # Pair-wise point connectivity, based on k-NN here and typically generated on-the-fly
    batch: Tensor

class ProcessedData(TypedDict):
    train: List[ProcessedDataItem] | List[str]
    val: List[ProcessedDataItem] | List[str]
    test: List[ProcessedDataItem] | List[str]

class TemplateData(TypedDict):
    points: List[ndarray]
    corrections: List[ndarray]
    sample_points: List[ndarray]
    sd: List[ndarray]
    normals: List[ndarray]
    id: List[ndarray]
    cells: ndarray
    surface: ndarray

class TemplateDataItem(TypedDict):
    points: ndarray
    corrections: ndarray
    sample_points: ndarray
    sd: ndarray
    normals: ndarray
    id: ndarray
    cells: ndarray
    surface: ndarray


class RawOrSplitData(TypedDict):
    train: TemplateData
    val: TemplateData
    test: TemplateData

stages_list = [stage for stage in get_type_hints(ProcessedData)]

if __name__ == "__main__":
    pass