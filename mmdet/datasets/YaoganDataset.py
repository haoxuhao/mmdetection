from .registry import DATASETS
from .coco import CocoDataset

@DATASETS.register_module
class YaoganDataset(CocoDataset):
    CLASSES = (
        'large-vehicle',
        'swimming-pool',
        'helicopter',
        'bridge',
        'plane',
        'ship',
        'soccer-ball-field',
        'basketball-court',
        'airport',
        'container-crane',
        'ground-track-field',
        'small-vehicle',
        'harbor',
        'baseball-diamond',
        'tennis-court',
        'roundabout',
        'storage-tank',
        'helipad',
    )
    def __init__(self, **kwargs):
        super(YaoganDataset, self).__init__(**kwargs)