import config

import os
import xlrd
import torch
import numpy as np
from torch.utils.data import Dataset


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


class NERDataset(Dataset):
    def __init__(self, records, label_func, tokenizer):
        self.data = self.preprocess(records, label_func, tokenizer)  # (token_ids, label_ids, raw)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

    def preprocess(self, origin_records, label_func, tokenizer):
        print("- Parsing labels...")
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
        print(f"In: {len(origin_records)}, Out: {len(data)}")
        return data

    def collate_fn(self, batch):
        # data in batch
        sentences = [x[0] for x in batch]  # [token_ids]
        labels = [x[1] for x in batch]  # [label_ids]

        # size, length information
        batch_size = len(sentences)
        max_len = max([len(sentence) for sentence in sentences])
        max_label_len = max([len(label) for label in labels])

        # padding
        batch_data = np.zeros((batch_size, max_len))
        batch_labels = np.zeros((batch_size, max_label_len))
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


def gen_bio(record: list):

    def _add_prefix(labels: list):
        new_labels = labels[:]
        # check each label
        for idx in range(len(labels)):
            # ignore O
            if labels[idx] == "O":
                continue
            # add B-
            if idx == 0 or labels[idx-1] != labels[idx]:
                new_labels[idx] = "B-" + new_labels[idx]
            # add I-
            elif labels[idx-1] == labels[idx]:
                new_labels[idx] = "I-" + new_labels[idx]
        return new_labels

    def _match_entity(addr, entity, split=False):
        # full matching
        if entity in addr:
            start_idx = addr.index(entity)
            end_idx = start_idx + len(entity)
            return list(range(start_idx, end_idx))

        # split matching for POI
        elif split:
            res_idxs = []
            last_idx = -1
            for char in entity:
                sub_addr = addr[last_idx+1:]
                if char in sub_addr:
                    last_idx = sub_addr.index(char) + last_idx + 1
                    res_idxs.append(last_idx)
            return res_idxs
        return None

    # retrieve data
    addr = record[0]
    entities = record[1:6]
    labels = ["POI", "building", "unit", "floor", "room"]
    
    # init result
    bio = ["O" for _ in addr]

    # generate
    for entity, label in zip(entities, labels):
        match_res = _match_entity(addr, entity, split=label=="POI")
        if match_res is None:
            return None

        for idx in match_res:
            if bio[idx] == "floor" and label == "room":
                bio[idx] = "floor|room"
            else:
                bio[idx] = label

    # prefix
    bio = _add_prefix(bio)

    return bio


def test_bio(idx):
    tool = DataTool(config.data_file, config.data_headers, ratios=[1,1])
    train_records, test_records = tool.split
    r = train_records[idx]
    bio = gen_bio(r)
    for c, tag in zip(r[0], bio):
        print(c, tag)

# test_bio(10)
