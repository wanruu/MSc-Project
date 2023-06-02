from .excel_module import xls_reader, xls_writer
from .data_tool import DataTool, custom_get_dataloader, custom_load_dataset
from .metrics import Metrics
from . import bio_13, bio_13_split


import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tag:
    def __init__(self, method):
        self.method = method
    
    @property
    def label2id(self):
        if self.method == "bio_13":
            return bio_13.label2id
        elif self.method == "bio_13_split":
            return bio_13_split.label2id
        return None
    
    @property
    def id2label(self):
        if self.method == "bio_13":
            return bio_13.id2label
        elif self.method == "bio_13_split":
            return bio_13_split.id2label
        return None
    
    @property
    def generate_func(self):
        if self.method == "bio_13":
            return bio_13.generate_tags
        elif self.method == "bio_13_split":
            return bio_13_split.generate_tags
        return None