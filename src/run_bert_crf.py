import config
from test import test
from train import train

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# transformer
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, TrainingArguments, HfArgumentParser
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

import os
import json
import numpy as np
from datasets import ClassLabel, load_dataset, Features, Sequence, Value
from torchcrf import CRF
from dataclasses import dataclass, field
from typing import Optional


class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        # remove label [CLS]
        origin_sequence_output = [layer[1:] for layer in sequence_output]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout
        padded_sequence_output = self.dropout(padded_sequence_output)
        # result
        logits = self.classifier(padded_sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs
  
        return outputs

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # config_name: Optional[str] = field(
    #     default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    # )
    # tokenizer_name: Optional[str] = field(
    #     default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    # )
    # cache_dir: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    # )
    # model_revision: str = field(
    #     default="main",
    #     metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    # )
    # use_auth_token: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
    #         "with private models)."
    #     },
    # )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    features_file: str = field(
        metadata={"help": "An input features data file."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    
    def __post_init__(self):
        if self.train_file is None and self.test_file is None:
            raise ValueError("Need training/testing file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


if __name__ == "__main__":
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Create directory
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)


    # Load dataset from files
    print("* Loading datasets from files...")
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    with open(data_args.features_file, 'r') as f:
        tags = json.load(f)["ner_tags"]
    datasets = load_dataset(extension, data_files=data_files, features=Features({
        "raw": Sequence(feature=Value(dtype="string")),
        "tokens": Sequence(feature=Value(dtype="string")),
        "ner_tags": Sequence(feature=ClassLabel(names=tags)),
    }))

    
    # Get label list
    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features
    label_list = features["ner_tags"].feature.names
    label2id = {label_list[i]: i for i in range(len(label_list))}
    id2label = id2label = {label2id[label]: label for label in label2id}
    num_labels = len(label_list)


    # Load pretrained model and tokenizer
    print("* Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    model = BertNER.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)
    model.to(config.device)
    # num_added_toks = tokenizer.add_tokens(config.new_tokens)
    # model.resize_token_embeddings(len(tokenizer))


    # Prepare data
    print("* Preparing data...")
    def preprocess(examples):
        # add [CLS]
        tokens = ["[CLS]"] + examples["tokens"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        examples["token_ids"] = token_ids
        return examples

    def collate_fn(batch):
        # data in batch
        sentences = [x["token_ids"] for x in batch]  # [token_ids]
        labels = [x["ner_tags"] for x in batch]  # [label_ids]

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


    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        train_dataset = train_dataset.map(
            preprocess,
            num_proc=data_args.preprocessing_num_workers
        )
        train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=False, collate_fn=collate_fn)

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        test_dataset = test_dataset.map(
            preprocess,
            num_proc=data_args.preprocessing_num_workers
        )
        test_dataloader = DataLoader(test_dataset, 1, shuffle=False, collate_fn=collate_fn)

    # train
    if training_args.do_train:
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
        train(train_dataloader, model, optimizer, scheduler, epochs=config.epoch_num)
        torch.save(model.state_dict(), os.path.join(training_args.output_dir, "checkpoint.pt"))

    # test
    if training_args.do_predict:
        print("* Starting testing...")
        if not training_args.do_train:
            state_dict = torch.load(os.path.join(training_args.output_dir, "checkpoint.pt"))
            model.load_state_dict(state_dict)

        
        metrics = test(test_dataloader, model, id2label)
        with open(os.path.join(training_args.output_dir, "all_results.json"), "w") as f:
            f.write(json.dumps(metrics, indent=4))
        print(metrics)
