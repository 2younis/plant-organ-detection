from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator

import os


class HrbEvaluator(COCOEvaluator):

    def __init__(self, dataset_name, cfg, distributed=True, output_dir='detections/evaluations/'):
        super().__init__(dataset_name, cfg, distributed, output_dir)
        self._metadata = MetadataCatalog.get(dataset_name)