

import config
from metrics import Metrics
from data import NERDataset, gen_bio, RecordsTool
from train import train
from model import BertNER

import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW


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


def test(dataloader, model):
    M = Metrics()
    for data, labels in tqdm(dataloader):
        pred = test_single(data, model)
        M.append(pred, labels[0])
    print(f"Accuracy: {M.acc}")


if __name__ == "__main__":
    print("* Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    tokenizer.add_tokens(config.new_tokens)
    
    model = BertNER.from_pretrained(config.bert_model, num_labels=len(config.label2id))
    model.resize_token_embeddings(len(tokenizer))
    state_dict = torch.load("checkpoints/test.pt")
    model.load_state_dict(state_dict)

    print("* Preparing data...")
    tool = RecordsTool(config.data_file, config.data_headers, ratios=[1,1])
    train_records, test_records = tool.splited

    train_dataset = NERDataset(train_records, gen_bio, tokenizer)
    test_dataset = NERDataset(test_records, gen_bio, tokenizer)

    train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=False, collate_fn=train_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, collate_fn=test_dataset.collate_fn)
    print(len(train_dataset), len(test_dataset), len(train_dataset)+len(test_dataset))


    model.to(config.device)
    test(test_dataloader, model)