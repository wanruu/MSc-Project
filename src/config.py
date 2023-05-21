import torch

# data source
data_file = "../data/20230412.xls"
data_headers = ["用户地址", "POI", "楼", "单元", "层", "房间"]

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
bert_model = "bert-base-chinese"
roberta_model = "hfl/chinese-roberta-wwm-ext"#-large"
bert_wwm_model = "hfl/chinese-bert-wwm"

# new tokens
new_tokens = [" "] # + [chr(_) for _ in range(97,123)] + [chr(_) for _ in range(65,91)]  # space, a-z, A-Z

# hyperparameter
learning_rate = 3e-5
weight_decay = 0.01
batch_size = 32
epoch_num = 10