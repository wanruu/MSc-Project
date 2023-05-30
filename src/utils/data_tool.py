import os
import xlrd
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import ClassLabel, load_dataset, Features, Sequence, Value



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataTool:
    def __init__(self, filename, headers, ratios):
        # read & clean
        records = self._read(filename, headers)
        records = self._clean(records)
        records_len = len(records)
        
        # normalize ratios -> lengths
        ratios = np.array(ratios)
        ratios = ratios / np.sum(ratios)
        lengths = [int(ratio*records_len) for ratio in ratios[:-1]]
        lengths.append(records_len-sum(lengths))
        self.split = self._split(records, lengths)

    def _read(self, filename, headers) -> list:
        """Read file [filename] with [data_header] and [label_headers], return a 2D list."""
        print(f"- Reading records from {filename}...")

        def _read_xls():
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
                record = [str(cell.value).lower() if cell.ctype != 2 else str(int(cell.value)) for cell in valid_cells]
                records.append(record)
            return records
        
        filetype = os.path.splitext(filename)[-1]
        if filetype == ".xls":
            records = _read_xls()
        else:
            records = []
        
        print(f"Final size: {len(records)}")
        return records

    def _clean(self, records: list) -> list:
        print("- Cleaning records...")
        # remove invalid data (ie, no tag)
        clean_records = list(filter(lambda record: set(record[1:])-{""}, records))
        print("Empty tagging:", len(records)-len(clean_records))

        # convert full-width to half-width
        for row_idx, record in enumerate(clean_records):
            for col_idx, item in enumerate(record):
                tmp = item.replace("（", "(").replace("）", ")")
                clean_records[row_idx][col_idx] = tmp
        
        # remove repeat records
        n = len(clean_records)
        poi_dict = dict()
        for record in clean_records:
            if record[0] not in poi_dict:
                poi_dict[record[0]] = record
        clean_records = [poi_dict[poi] for poi in poi_dict]
        print("Repeated POI:", n-len(clean_records))

        print(f"Final size: {len(clean_records)}")
        return clean_records

    def _split(self, records, lengths) -> list:
        print("- Splitting records...")
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

        print([len(sr) for sr in split_records])
        return split_records



def custom_load_dataset(features_file, train_file=None, test_file=None):
    data_files = {}
    if train_file is not None:
        data_files["train"] = train_file
        extension = train_file.split(".")[-1]
    if test_file is not None:
        data_files["test"] = test_file
        extension = test_file.split(".")[-1]
    with open(features_file, "r") as f:
        tags = json.load(f)["ner_tags"]
    
    datasets = load_dataset(
        extension,
        data_files=data_files,
        features=Features(
            {
                "raw": Sequence(feature=Value(dtype="string")),
                "tokens": Sequence(feature=Value(dtype="string")),
                "ner_tags": Sequence(feature=ClassLabel(names=tags)),
            }
        ),
    )

    return datasets


def custom_get_dataloader(dataset, batch_size):
    def collate_fn(batch):
        # data in batch
        input_ids = [x["token_ids"] for x in batch]  # [token_ids]
        labels = [x["ner_tags"] for x in batch]  # [label_ids]

        # size, length information
        batch_size = len(input_ids)
        max_len = max([len(sentence) for sentence in input_ids])
        max_label_len = max([len(label) for label in labels])

        # padding
        batch_data = np.zeros((batch_size, max_len))
        batch_labels = np.zeros((batch_size, max_label_len))
        for idx in range(batch_size):
            cur_len = len(input_ids[idx])
            cur_label_len = len(labels[idx])
            batch_data[idx][:cur_len] = input_ids[idx]
            batch_labels[idx][:cur_label_len] = labels[idx]

        # convert to tensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

        return batch_data, batch_labels


    # dataset = dataset.map(preprocess)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader