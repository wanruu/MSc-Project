
import csv
import config
from metrics import Metrics
from data import NERDataset, gen_bio, DataTool
from model import BertNER

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer


def test_single(data, model):
    """
    Example:
        data(batch_size=1): tensor([[101,1298,2255,3255,1736,118,21129,128,3406,21128,
            113,1298,2255,3255,1736, 21129,128,3406,122,122,3517,114]], device='cuda:0')
        return: tensor([1, 6, 6, 6, 0, 2, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 9, 0, 0], device='cuda:0')
    """
    model.eval()
    mask = data.gt(0)
    output = model(data, token_type_ids=None, attention_mask=mask, labels=None)[0][0]
    bio_result = torch.argmax(output, dim=1)
    return bio_result


def manual_test(addr, tokenizer, model):
    tokens = ["[CLS]"] + list(addr)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = torch.tensor([token_ids], dtype=torch.long)
    bio_result = test_single(token_ids, model).detach().numpy()
    bio_result = [config.id2label[tag] for tag in bio_result]
    bio_result = [tag[2:] if tag != "O" else tag for tag in bio_result]

    poi_idx = [idx for idx, content in enumerate(bio_result) if content == "POI"]
    building_idx = [idx for idx, content in enumerate(bio_result) if content == "building"]
    unit_idx = [idx for idx, content in enumerate(bio_result) if content == "unit"]
    floor_idx = [idx for idx, content in enumerate(bio_result) if "floor" in content]
    room_idx = [idx for idx, content in enumerate(bio_result) if "room" in content]

    poi = [addr[idx] for idx in poi_idx]
    building = [addr[idx] for idx in building_idx]
    unit = [addr[idx] for idx in unit_idx]
    floor = [addr[idx] for idx in floor_idx]
    room = [addr[idx] for idx in room_idx]

    print(poi, building, unit, floor, room)


def test(dataloader, model):
    tmp=0
    M = Metrics()
    for data, labels in tqdm(dataloader):
        pred = test_single(data, model)
        M.append(data[0], pred, labels[0])
        # tmp+=1
        # if tmp==50:
        #     break

    bad_cases_data = []
    bad_cases = set()
    for label_id in config.id2label:
        acc, bad_case, note = M.with_target_full_match(label_id)
        print(acc, note, config.id2label[label_id])
        bad_cases = bad_cases | bad_case

    for idx in bad_cases:
        tokens = M.raws[idx].detach().cpu().numpy()
        addr = tokenizer.convert_ids_to_tokens(tokens)[1:]

        pred = M.preds[idx].detach().cpu().numpy()  # n
        pred_labels = [config.id2label[label_id].replace("I-","").replace("B-","") for label_id in pred]
        actual = M.actuals[idx].detach().cpu().numpy()  # n
        actual_labels = [config.id2label[label_id].replace("I-","").replace("B-","") for label_id in actual]

        labels = ["POI", "building", "unit", "floor", "room"]
        bad_case_dict = {"address": "".join(addr)}
        for label in labels:
            pred_idx = [idx for idx, pred_label in enumerate(pred_labels) if label in pred_label]
            pred_entity = [addr[idx] for idx in pred_idx]
            actual_idx = [idx for idx, actual_label in enumerate(actual_labels) if label in actual_label]
            actual_entity = [addr[idx] for idx in actual_idx]
            bad_case_dict["pred-"+label] = "".join(pred_entity)
            bad_case_dict["truth-"+label] = "".join(actual_entity)
        bad_cases_data.append(bad_case_dict)

    with open("badcases.csv", "w+", encoding="utf-8", newline="") as f:
        headers = bad_cases_data[0].keys()
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        for line in bad_cases_data:
            writer.writerow(line)


if __name__ == "__main__":
    print("* Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    tokenizer.add_tokens(config.new_tokens)
    
    model = BertNER.from_pretrained(config.bert_model, num_labels=len(config.label2id))
    model.resize_token_embeddings(len(tokenizer))
    state_dict = torch.load("checkpoints/bert-2023-04-06_11-28.pt")
    model.load_state_dict(state_dict)

    print("* Preparing data...")
    tool = DataTool(config.data_file, config.data_headers, ratios=[1,1])
    _, test_records = tool.split

    test_dataset = NERDataset(test_records, gen_bio, tokenizer)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, collate_fn=test_dataset.collate_fn)

    # print("* Starting testing...")
    model.to(config.device)
    test(test_dataloader, model)


    # manual_test("城建御府名筑花园 (1栋4B)", tokenizer, model)