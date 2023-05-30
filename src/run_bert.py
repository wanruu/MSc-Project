# customized
from utils import device, custom_load_dataset, custom_get_dataloader, Metrics

# torch
import torch
import torch.nn as nn
# from torch.optim.lr_scheduler import LambdaLR

# transformer
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
# from transformers.optimization import AdamW
from transformers import AdamW, get_linear_schedule_with_warmup

import os
import json
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Arguments:
    model_name: str = field(metadata={"help": "Path to pretrained model from huggingface.co/models"})
    batch_size: int = field(metadata={"help": "Training batch size"})

    features_file: str = field(metadata={"help": "An input features data file."})
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a JSON file)."})
    test_file: Optional[str] = field(default=None, metadata={"help": "An optional input test data file to predict on (a JSON file)."})

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


def train(dataloader, model, epoch_num, loss_func, optimizer, scheduler):
    model.train()
    for epoch in range(int(epoch_num)):
        total_loss = 0.0
        for input_ids, labels in tqdm(dataloader):
            # model output
            logits = model(
                input_ids=input_ids, 
                token_type_ids=None, 
                attention_mask=input_ids.gt(0)
            ).logits  # [32, 38, 13]
            
            # remove [CLS], flatten, -100 if empty
            input_ids = input_ids[:,1:]
            input_ids = torch.reshape(input_ids, (-1,))  # [...]
            logits = logits[:,1:,:]
            logits = torch.reshape(logits, (-1, logits.size(-1)))  # [..., 13]
            logits[input_ids==0] = -100
            labels = torch.reshape(labels, (-1,))
            labels[input_ids==0] = -100
            
            # calculate loss
            loss = loss_func(logits, labels)
            total_loss += loss.item()

            # back-propagate
            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        total_loss = total_loss / len(input_ids)
        logger.info(f"Epoch: {epoch}, loss: {total_loss:.3f}")


def test(dataloader, model, id2label):
    M = Metrics()
    model.eval()
    for input_ids, label_ids in tqdm(dataloader):
        # preds
        logits = model(
            input_ids=input_ids, 
            token_type_ids=None, 
            attention_mask=input_ids.gt(0)
        ).logits  # [1, 32, 13]
        logits = logits[:,1:,:]
        preds = torch.argmax(logits, dim=2)  # [1, 31, 13]

        # to cpu numpy
        labels = [id2label[label] for label in label_ids[0].cpu().numpy()]
        preds = [id2label[pred] for pred in preds[0].cpu().numpy()]
        M.append(labels, preds)

    return M.compute()



if __name__ == "__main__":
    # Parse arguments
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

    # Create directory & logger
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    file_handler = logging.FileHandler(os.path.join(training_args.output_dir, "train.log"))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log parameters
    logger.info(args)
    logger.info(training_args)

    # Load dataset from files
    logger.info("* Loading datasets from files...")
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


    # Load pretrained model and tokenizer
    logger.info("* Loading BERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )
    model.to(device)


    # Prepare data
    logger.info("* Preparing data...")
    def preprocess(examples):
        # add [CLS] before
        tokens = ["[CLS]"] + examples["tokens"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
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


    # train
    if training_args.do_train:
        logger.info("* Preparing loss function & optimizer & scheduler...")
        loss_func = nn.CrossEntropyLoss()
        # optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
        # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)
        optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, eps=training_args.adam_epsilon)
        total_steps = len(train_dataloader) * training_args.num_train_epochs
        warmup_steps = int(total_steps * training_args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        logger.info("* Starting training...")
        train(
            dataloader=train_dataloader, 
            model=model,
            epoch_num=training_args.num_train_epochs,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler
        )
        torch.save(model.state_dict(), os.path.join(training_args.output_dir, "checkpoint.pt"))

    # test
    if training_args.do_predict:
        logger.info("* Starting testing...")
        if not training_args.do_train:
            state_dict = torch.load(os.path.join(training_args.output_dir, "checkpoint.pt"))
            model.load_state_dict(state_dict)

        metrics = test(
            dataloader=test_dataloader, 
            model=model, 
            id2label=id2label
        )
        with open(os.path.join(training_args.output_dir, "results.json"), "w") as f:
            f.write(json.dumps(metrics, indent=4))
        # logger.info(metrics)
