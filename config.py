import torch

# data source
data_file = "./data/20230315.xls"
data_headers = ["用户地址", "POI", "楼", "单元", "层", "房间"]

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# label
label2id = {
    "O": 0,
    "B-POI": 1,
    "B-building": 2,
    "B-unit": 3,
    "B-floor": 4,
    "B-room": 5,
    "I-POI": 6,
    "I-building": 7,
    "I-unit": 8,
    "I-floor": 9,
    "I-room": 10,
    "B-floor|room": 11,
    "I-floor|room": 12,
}
id2label = {label2id[label]: label for label in label2id}

# model
bert_model = "bert-base-chinese"
roberta_model = "hfl/chinese-roberta-wwm-ext-large"

# new tokens
new_tokens = [" "] + [chr(_) for _ in range(97,123)] + [chr(_) for _ in range(65,91)]  # space, a-z, A-Z

# hyperparameter
learning_rate = 3e-5
weight_decay = 0.01
batch_size = 32
epoch_num = 10