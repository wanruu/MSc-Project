from .excel_module import xls_reader, xls_writer
from .data_tool import DataTool, custom_get_dataloader, custom_load_dataset
from .metrics import Metrics
from . import bio_13, bio_11


import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tag:
    def __init__(self, method):
        self.method = method
    
    @property
    def label2id(self):
        if self.method == "bio_13":
            return bio_13.label2id
        elif self.method == "bio_11":
            return bio_11.label2id
        return None
    
    @property
    def id2label(self):
        if self.method == "bio_13":
            return bio_13.id2label
        elif self.method == "bio_11":
            return bio_11.id2label
        return None
    
    @property
    def generate_func(self):
        if self.method == "bio_13":
            return bio_13.generate_tags
        elif self.method == "bio_11":
            return bio_11.generate_tags
        return None