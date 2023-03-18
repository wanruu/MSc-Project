import config

import os
import xlrd
import torch
import numpy as np
from torch.utils.data import Dataset


class RecordsTool:
    def __init__(self, filename, headers, ratios):
        # read & clean
        records = self.read(filename, headers)
        records = self.clean(records)
        records_len = len(records)
        self.all = records
        # normalize ratios -> lengths
        ratios = np.array(ratios)
        ratios = ratios / np.sum(ratios)
        lengths = [int(ratio*records_len) for ratio in ratios[:-1]]
        lengths.append(records_len-sum(lengths))
        self.splited = self.split(records, lengths)        

    def read(self, filename, headers) -> list:
        """Read file [filename] with [data_header] and [label_headers], return a 2D list."""
        print(f"Reading records from {filename}...", end="")

        def read_xls():
            # read .xls file, read sheet
            sheet = xlrd.open_workbook(filename).sheet_by_index(0)

            # extract headers
            all_headers = sheet.row_values(0)
            valid_headers_idxs = [all_headers.index(header) for header in headers]

            # extract each row
            records = []
            for row_idx in range(1, sheet.nrows):
                cells = sheet.row(row_idx)
                valid_cells = [cells[idx] for idx in valid_headers_idxs]
                # this line convert float to str(int(_))
                record = [str(cell.value) if cell.ctype != 2 else str(int(cell.value)) for cell in valid_cells]
                records.append(record)
            return records
        
        filetype = os.path.splitext(filename)[-1]
        if filetype == ".xls":
            records = read_xls()
        else:
            records = []
        
        print(f"(size={len(records)})")
        return records

    def clean(self, records: list) -> list:
        print("Cleaning records...", end="")
        # remove invalid data (eg, no notation)
        clean_records = list(filter(lambda record: set(record[1:])-{""}, records))

        # convert full-width to half-width
        # delete space
        for row_idx, record in enumerate(clean_records):
            for col_idx, item in enumerate(record):
                tmp = item.replace("（", "(").replace("）", ")").replace(" ", "")
                clean_records[row_idx][col_idx] = tmp

        print(f"(size={len(clean_records)})")
        return clean_records

    def split(self, records, lengths) -> list:
        # deep copy and shuffle
        _records = records.copy()
        np.random.seed(10)
        np.random.shuffle(_records)

        # crop by length
        split_records = []
        start_idx = 0
        for length in lengths:
            end_idx = start_idx + length  # not include itself
            sub_records = _records[start_idx:end_idx]
            split_records.append(sub_records)
            start_idx = end_idx

        return split_records


class NERDataset(Dataset):
    def __init__(self, records, label_func, tokenizer):
        self.data = self.preprocess(records, label_func, tokenizer)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

    def preprocess(self, origin_records, label_func, tokenizer):
        data = []
        for record in origin_records:
            # process label
            addr, labels = record[0], label_func(record)
            if labels is None:
                continue
            # split addr by character, add [CLS] before addr, token_ids
            tokens = ["[CLS]"] + list(addr)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            # label_ids
            label_ids = [config.label2id.get(label) for label in labels]
            data.append((token_ids, label_ids))
        return data

    def collate_fn(self, batch):
        # data in batch
        sentences = [x[0] for x in batch]
        labels = [x[1] for x in batch]
    
        # size, length information
        batch_size = len(sentences)
        max_len = max([len(sentence) for sentence in sentences])
        max_label_len = max([len(label) for label in labels])

        # padding
        batch_data = np.ones((batch_size, max_len))
        batch_labels = np.ones((batch_size, max_label_len))
        for idx in range(batch_size):
            cur_len = len(sentences[idx])
            cur_label_len = len(labels[idx])
            batch_data[idx][:cur_len] = sentences[idx]
            batch_labels[idx][:cur_label_len] = labels[idx]

        # convert to tensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        batch_data, batch_labels = batch_data.to(config.device), batch_labels.to(config.device)

        return batch_data, batch_labels


# TODO: now only consider poi
def gen_bio(record: list):
    addr, poi, building, unit, floor, room = record
    # match poi
    idxs = [addr.index(symbol) if symbol in addr else -1 for symbol in poi]
    valid = -1 not in idxs

    if not valid:
        return None

    boi = ["O" for _ in addr]
    for idx in idxs:
        boi[idx] = "I-POI"
    for idx in range(0, len(boi)):
        if boi[idx] == "I-POI":
            if idx == 0 or "-POI" not in boi[idx-1]:
                boi[idx] = "B-POI"

    return boi


