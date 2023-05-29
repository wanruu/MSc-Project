import config
from utils import Metrics


import csv
import torch
from tqdm import tqdm


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


def test(dataloader, model, id2label):
    M = Metrics()
    for input_ids, label_ids in tqdm(dataloader):
        labels = [id2label[label] for label in label_ids[0].cpu().numpy()]
        preds = [id2label[pred] for pred in test_single(input_ids, model).cpu().numpy()]
        M.append(labels, preds)

    return M.compute()
