from .registry import DATASETS
from .xml_style import XMLDataset

@DATASETS.register_module
class HangkongbeiDataset(XMLDataset):
    CLASSES = (
        'Vehicle',
    )
    def __init__(self, **kwargs):
        super(HangkongbeiDataset, self).__init__(**kwargs)