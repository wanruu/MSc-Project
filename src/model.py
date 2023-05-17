import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertPreTrainedModel, BertModel
from torchcrf import CRF

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