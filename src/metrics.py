import torch
import numpy as np

class Metrics:
    def __init__(self):
        self.input_ids = dict()
        self.preds = dict()
        self.labels = dict()
        self.index = 0
    
    def append(self, input_ids, labels, pred):
        self.input_ids[self.index] = input_ids
        self.labels[self.index] = labels
        self.preds[self.index] = pred
        self.index += 1

    # @property
    # def acc(self):
    #     matched = 0
    #     for x, y in zip(self.preds, self.actuals):
    #         match_mat = torch.eq(self.preds[x], self.actuals[y])
    #         matched += int(torch.all(match_mat))
    #     return matched / len(self.preds)

    # @property
    # def accuracy(self):
    #     matched = 0
    #     total = 0
    #     for x, y in zip(self.preds, self.actuals):
    #         match_mat = torch.eq(self.preds[x], self.actuals[y])
    #         matched += torch.count_nonzero(match_mat)
    #         total += match_mat.shape[0]
    #     return matched / total


    def token_level(self, label_id):
        """Token level metrics"""
        tp, fn, fp, tn = 0, 0, 0, 0
        for idx in range(self.index):
            preds = torch.eq(self.preds[idx], label_id)  # [True, False, ...]
            labels = torch.eq(self.labels[idx], label_id)
            tp += torch.count_nonzero(preds & labels)  # pred=True, label=True
            tn += torch.count_nonzero(~preds & ~labels)  # pred=label=False
            fp += torch.count_nonzero(preds & ~labels)  # pred=True, label=False
            fn += torch.count_nonzero(~preds & labels)  # pred=False, label=True
        
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return accuracy.item(), precision.item(), recall.item(), f1.item()


    def entity_level(self, label_id):
        """Entity level accuracy"""

        correct = 0
        for idx in range(self.index):
            # input_ids = self.input_ids[idx][1:]  # remove [CLS]

            preds = self.preds[idx].detach().cpu().numpy()
            labels = self.labels[idx].detach().cpu().numpy()

            pred_idxes = np.where(preds==label_id)[0]
            label_idxes = np.where(labels==label_id)[0]

            # print(input_ids[pred_idxes])
            if pred_idxes.shape==label_idxes.shape and np.all(np.equal(pred_idxes, label_idxes)):
                correct += 1
            
        return correct/self.index

    def sentence_level(self):
        """Sentence level accuracy"""

        correct = 0
        for idx in range(self.index):
            preds = self.preds[idx].detach().cpu().numpy()
            labels = self.labels[idx].detach().cpu().numpy()
            if np.all(np.equal(preds, labels)):
                correct += 1
        return correct/self.index