
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


def test_single(input_ids, model):
    """
    Example:
        data(batch_size=1): tensor([[101,1298,2255,3255,1736,118,21129,128,3406,21128,
            113,1298,2255,3255,1736, 21129,128,3406,122,122,3517,114]], device='cuda:0')
        return: tensor([1, 6, 6, 6, 0, 2, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 9, 0, 0], device='cuda:0')
    """
    model.eval()
    mask = input_ids.gt(0)
    output = model(input_ids, token_type_ids=None, attention_mask=mask, labels=None)[0][0]
    labels = torch.argmax(output, dim=1)
    return labels


# def manual_test(addr, tokenizer, model):
#     tokens = ["[CLS]"] + list(addr)
#     token_ids = tokenizer.convert_tokens_to_ids(tokens)
#     token_ids = torch.tensor([token_ids], dtype=torch.long)
#     bio_result = test_single(token_ids, model).detach().numpy()
#     bio_result = [config.id2label[tag] for tag in bio_result]
#     bio_result = [tag[2:] if tag != "O" else tag for tag in bio_result]

#     poi_idx = [idx for idx, content in enumerate(bio_result) if content == "POI"]
#     building_idx = [idx for idx, content in enumerate(bio_result) if content == "building"]
#     unit_idx = [idx for idx, content in enumerate(bio_result) if content == "unit"]
#     floor_idx = [idx for idx, content in enumerate(bio_result) if "floor" in content]
#     room_idx = [idx for idx, content in enumerate(bio_result) if "room" in content]

#     poi = [addr[idx] for idx in poi_idx]
#     building = [addr[idx] for idx in building_idx]
#     unit = [addr[idx] for idx in unit_idx]
#     floor = [addr[idx] for idx in floor_idx]
#     room = [addr[idx] for idx in room_idx]

#     print(poi, building, unit, floor, room)


def test(dataloader, model):
    print("* Testing...")
    print("token_level: accuracy, precision, recall, f1")
    print("entity_level: accuracy")
    M = Metrics()
    for input_ids, labels in tqdm(dataloader):
        pred = test_single(input_ids, model)
        M.append(input_ids[0], labels[0], pred)

    for label_id in config.id2label:
        print(label_id, config.id2label[label_id])
        accuracy, precision, recall, f1 = M.token_level(label_id)
        print(accuracy, precision, recall, f1)
        accuracy = M.entity_level(label_id)
        print(accuracy)
        print()
    
    print("sentence_level: accuracy")
    accuracy = M.sentence_level()
    print(accuracy)


if __name__ == "__main__":
    model_name = config.roberta_model

    print("* Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(config.new_tokens)
    
    model = BertNER.from_pretrained(model_name, num_labels=len(config.label2id))
    model.resize_token_embeddings(len(tokenizer))
    state_dict = torch.load("../checkpoints/roberta-2023-05-17_20-44.pt")
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