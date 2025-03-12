import random
import os
import numpy as np
import paddle
import logging
import argparse

from cdsam.model import CDSam

from core.datasets.cdloader import CDReader
from core.cdmisc.predict import predict


dataset_name = "SYSU_CD"


dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

    
if __name__ == "__main__":
    print("main")
    
    dataset = CDReader(dataset_path, mode='test')
    model = CDSam(256)
    weight_path = r""
    predict(model, dataset, weight_path, dataset_name)
    


