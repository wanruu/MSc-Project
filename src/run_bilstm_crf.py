# customized
from utils import device, custom_load_dataset, custom_get_dataloader, Metrics

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# transformer
from transformers import TrainingArguments, HfArgumentParser

import re
import os
import json
import logging
from tqdm import tqdm
from torchcrf import CRF
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, num_layers=1, bidirectional=True)
        self.classifier = nn.Linear(2*hidden_dim, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, lengths, labels=None):
        emb = self.embedding(input_ids)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        logits = self.classifier(lstm_out)

        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs

        return outputs


def test(dataloader, model, id2label):
    M = Metrics()
    model.eval()
    for input_ids, label_ids in tqdm(dataloader):
        # preds
        lengths = torch.tensor([torch.count_nonzero(tokens) for tokens in input_ids])
        logits = model(input_ids=input_ids, lengths=lengths)[0]  # [1, 32, 13]
        preds = torch.argmax(logits, dim=2)  # [1, 31, 13]

        # to cpu numpy
        labels = [id2label[label] for label in label_ids[0].cpu().numpy()]
        preds = [id2label[pred] for pred in preds[0].cpu().numpy()]
        M.append(labels, preds)

    return M.compute()


def train(dataloader, model, start_epoch, end_epoch, optimizer, scheduler):
    model.train()
    for epoch in range(start_epoch, end_epoch):
        total_loss = 0.0
        for input_ids, labels in tqdm(dataloader):
            lengths = torch.tensor([torch.count_nonzero(tokens) for tokens in input_ids])
            loss = model(input_ids=input_ids, lengths=lengths, labels=labels)[0]
            total_loss += loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step(total_loss)
        # total_loss = total_loss / len(input_ids)
        logger.info(f"Epoch: {epoch}, loss: {total_loss:.5f}")


def get_last_checkpoint(path):
    files = os.listdir(path)
    regex = r"^checkpoint-(\d+).pt$"
    matches = [(re.search(regex, file), file) for file in files]
    matches = [(int(epoch.group(1)), file) for epoch, file in matches if epoch]
    if not matches:
        return None
    matches.sort(key=lambda match: match[0], reverse=True)
    return torch.load(os.path.join(path, matches[0][1]))


@dataclass
class Arguments:
    batch_size: int = field(metadata={"help": "Training batch size"})
    embedding_dim: int = field(metadata={"help": ""})
    hidden_dim: int = field(metadata={"help": ""})

    features_file: str = field(
        metadata={"help": "An input features data file."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a JSON file)."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a JSON file)."},
    )
    
    def __post_init__(self):
        if self.train_file is None and self.test_file is None:
            raise ValueError("Need training/testing file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension == "json", "`train_file` should be a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension == "json", "`test_file` should be a json file."


if __name__ == "__main__":
    # Parse arguments
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

    num_train_epochs = int(training_args.num_train_epochs)

    # Create directory
    output_dir = training_args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get last checkpoint
    last_checkpoint = get_last_checkpoint(output_dir)

    # Prepare logger
    log_mode = "a" if last_checkpoint else "w"
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"), mode=log_mode)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log parameters
    logger.info(args)
    logger.info(training_args)


    # ------------------------------------------------------
    # Load dataset from files
    logger.info("* Loading datasets, vocab from files...")
    datasets = custom_load_dataset(
        features_file=args.features_file,
        train_file=args.train_file,
        test_file=args.test_file
    )

    # Get label list
    features = datasets["train"].features if training_args.do_train else datasets["test"].features
    label_list = features["ner_tags"].feature.names
    label2id = {label_list[i]: i for i in range(len(label_list))}
    id2label = id2label = {label2id[label]: label for label in label2id}
    num_labels = len(label_list)

    # Get vocab
    if training_args.do_train:
        token2id = dict()
        token2id["[PAD]"] = 0
        for data in datasets["train"]:
            for token in data["tokens"]:
                if token not in token2id:
                    token2id[token] = len(token2id)
        token2id["[UNK]"] = len(token2id)
        with open(os.path.join(output_dir, "vocab.txt"), encoding="utf-8", mode="w+") as f:
            for token in token2id:
                f.write(token + "\n")
    else:
        with open(os.path.join(output_dir, "vocab.txt"), encoding="utf-8") as f:
            tokens = f.read().splitlines()
            token2id = {token: idx for idx, token in enumerate(tokens)}
    id2token = {token2id[token]: token for token in token2id}

    # Process data
    def preprocess(examples):
        tokens = examples["tokens"]
        token_ids = [token2id[token] if token in token2id else token2id["[UNK]"] for token in tokens]
        examples["token_ids"] = token_ids
        return examples

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"].map(preprocess)
        train_dataloader = custom_get_dataloader(dataset=train_dataset, batch_size=args.batch_size)

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"].map(preprocess)
        test_dataloader = custom_get_dataloader(dataset=test_dataset, batch_size=1)


    # ------------------------------------------------------
    # Load model
    logger.info("* Preparing model...")
    model = BiLSTM(
        vocab_size=len(token2id),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_labels=num_labels,
    )
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info(f"# paramters: {num_params}")
    model.to(device)
    if last_checkpoint:
        logger.info("resume from last checkpoint")
        model.load_state_dict(last_checkpoint["model"])

    # ------------------------------------------------------
    # train
    if training_args.do_train:
        logger.info("* Preparing optimizer & scheduler...")
        optimizer = optim.SGD(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
        start_epoch = 0

        if last_checkpoint:
            logger.info("[resume from last checkpoint]")
            optimizer.load_state_dict(last_checkpoint["optimizer"])
            scheduler.load_state_dict(last_checkpoint["scheduler"])
            start_epoch = last_checkpoint["epoch_num"]

        logger.info("* Starting training...")
        save_interval = 100
        first_interval_end = (start_epoch // save_interval + 1) * save_interval - 1
        epoch_intervals = [[end-save_interval+1, end] for end in range(first_interval_end, num_train_epochs, save_interval)]  # closed interval
        epoch_intervals[0][0] = start_epoch
        if epoch_intervals[-1][1] != num_train_epochs-1:
            epoch_intervals.append([epoch_intervals[-1][1]+1, num_train_epochs-1])

        for start, end in epoch_intervals:
            train(
                dataloader=train_dataloader, 
                model=model, 
                start_epoch=start,
                end_epoch=end+1,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            logger.info(f"Save checkpoint-{end+1}.pt")
            torch.save(
                {
                    "epoch_num": end+1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                },
                os.path.join(output_dir, f"checkpoint-{end+1}.pt")
            )

    # ------------------------------------------------------
    # test
    if training_args.do_predict:
        logger.info("* Starting testing...")
        metrics = test(
            dataloader=test_dataloader,
            model=model,
            id2label=id2label
        )
        with open(os.path.join(output_dir, "all_results.json"), "w") as f:
            f.write(json.dumps(metrics, indent=4))
