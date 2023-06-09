# customized
from utils import device, custom_load_dataset, custom_get_dataloader, Metrics

# torch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# transformer
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, TrainingArguments, HfArgumentParser
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

import os
import json
import logging
from tqdm import tqdm
from torchcrf import CRF
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BertNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNER, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        # self.init_weights()

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


def test_single(input_ids, model):
    """
    Example:
        data(batch_size=1): tensor([[101,1298,2255,3255,1736,118,21129,128,3406,21128,
            113,1298,2255,3255,1736, 21129,128,3406,122,122,3517,114]], device='cuda:0')
        return: tensor([1, 6, 6, 6, 0, 2, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 9, 0, 0], device='cuda:0')
    """
    mask = input_ids.gt(0)
    output = model(input_ids, token_type_ids=None, attention_mask=mask, labels=None)[0][0]
    labels = torch.argmax(output, dim=1)
    return labels


def test(dataloader, model, id2label):
    model.eval()
    M = Metrics()
    for input_ids, label_ids in tqdm(dataloader):
        labels = [id2label[label] for label in label_ids[0].cpu().numpy()]
        preds = [id2label[pred] for pred in test_single(input_ids, model).cpu().numpy()]
        M.append(labels, preds)

    return M.compute()


def train(dataloader, model, optimizer, scheduler, epoch_num):
    model.train()
    for epoch in range(int(epoch_num)):
        total_loss = 0.0
        for data, labels in tqdm(dataloader):
            loss = model(input_ids=data, token_type_ids=None, attention_mask=data.gt(0), labels=labels)[0]
            total_loss += loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        total_loss = total_loss / len(data)
        logger.info(f"Epoch: {epoch}, loss: {total_loss:.3f}")


@dataclass
class Arguments:
    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    batch_size: int = field(metadata={"help": "Training batch size"})

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


    # Create directory
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
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertNER.from_pretrained(args.model_name, num_labels=num_labels)
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info(f"# paramters: {num_params}")
    model.to(device)


    # Prepare data
    logger.info("* Preparing data...")
    def preprocess(examples):
        # add [CLS]
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
        logger.info("* Preparing optimizer & scheduler...")
        bert_optimizer = list(model.bert.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": training_args.weight_decay},
            {"params": [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0},
            {"params": [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
                "lr": training_args.learning_rate * 5, "weight_decay": training_args.weight_decay},
            {"params": [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
                "lr": training_args.learning_rate * 5, "weight_decay": 0.0},
            {"params": model.crf.parameters(), "lr": training_args.learning_rate * 5}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, correct_bias=False)
        train_steps_per_epoch = len(train_dataset) // args.batch_size
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=(int(training_args.num_train_epochs) // 10) * train_steps_per_epoch,
                                                    num_training_steps=int(training_args.num_train_epochs * train_steps_per_epoch))

        logger.info("* Starting training...")
        train(
            dataloader=train_dataloader, 
            model=model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            epoch_num=training_args.num_train_epochs
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
        with open(os.path.join(training_args.output_dir, "all_results.json"), "w") as f:
            f.write(json.dumps(metrics, indent=4))
        # logger.info(metrics)
