import config
from data import NERDataset, gen_bio, DataTool
from test import test
from train import train
from model import BertNER

import time
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

model_name = config.roberta_model
print(config.data_file)
print(model_name)

print("* Loading BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertNER.from_pretrained(model_name, num_labels=len(config.label2id))
num_added_toks = tokenizer.add_tokens(config.new_tokens)
model.resize_token_embeddings(len(tokenizer))


print("* Preparing data...")
tool = DataTool(config.data_file, config.data_headers, ratios=[1,1])
train_records, test_records = tool.split

train_dataset = NERDataset(train_records, gen_bio, tokenizer)
test_dataset = NERDataset(test_records, gen_bio, tokenizer)

train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=False, collate_fn=train_dataset.collate_fn)
test_dataloader = DataLoader(test_dataset, 1, shuffle=False, collate_fn=test_dataset.collate_fn)


print("* Preparing optimizer & scheduler...")
bert_optimizer = list(model.bert.named_parameters())
classifier_optimizer = list(model.classifier.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
        "weight_decay": config.weight_decay},
    {"params": [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0},
    {"params": [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
        "lr": config.learning_rate * 5, "weight_decay": config.weight_decay},
    {"params": [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
        "lr": config.learning_rate * 5, "weight_decay": 0.0},
    {"params": model.crf.parameters(), "lr": config.learning_rate * 5}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
train_steps_per_epoch = len(train_dataset) // config.batch_size
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                            num_training_steps=config.epoch_num * train_steps_per_epoch)


print("* Starting training...")
model.to(config.device)
train(train_dataloader, model, optimizer, scheduler, epochs=config.epoch_num)
cur_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime()) 
torch.save(model.state_dict(), f"../checkpoints/{model_name}-{cur_time}.pt")


print("* Starting testing...")
test(test_dataloader, model)