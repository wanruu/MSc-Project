import torch
import numpy as np

class Metrics:
    def __init__(self):
        self.raws = dict()
        self.preds = dict()
        self.actuals = dict()
        self.index = 0
    
    def append(self, raw, pred, actual):
        self.raws[self.index] = raw
        self.preds[self.index] = pred
        self.actuals[self.index] = actual
        self.index += 1

    # @property
    # def acc(self):
    #     matched = 0
    #     for x, y in zip(self.preds, self.actuals):
    #         match_mat = torch.eq(self.preds[x], self.actuals[y])
    #         matched += int(torch.all(match_mat))
    #     return matched / len(self.preds)

    @property
    def accuracy(self):
        matched = 0
        total = 0
        for x, y in zip(self.preds, self.actuals):
            match_mat = torch.eq(self.preds[x], self.actuals[y])
            matched += torch.count_nonzero(match_mat)
            total += match_mat.shape[0]
        return matched / total


    # def with_target(self, target):
    #     tp, fn, fp, tn = 0, 0, 0, 0
    #     for x, y in zip(self.preds, self.actuals):
    #         pred = torch.eq(self.preds[x], target)
    #         actual = torch.eq(self.actuals[y], target)
    #         tp += torch.count_nonzero(pred & actual)
    #         tn += torch.count_nonzero(~pred & ~actual)
    #         fn += torch.count_nonzero(~pred & actual)
    #         fp += torch.count_nonzero(pred & ~actual)
        
    #     acc = (tp + tn) / (tp + fp + tn + fn)
    #     prec = tp / (tp + fp)
    #     recall = tp / (tp + fn)
    #     f1 = 2 * prec * recall / (prec + recall)
    #     return acc, prec, recall, f1


    def with_target_full_match(self, target):
        bad_cases = set()
        correct, total = 0, 0
        for idx in range(self.index):
            pred = self.preds[idx].detach().cpu().numpy()
            actual = self.actuals[idx].detach().cpu().numpy()

            pred = pred[pred==target].nonzero()[0]
            actual = actual[actual==target].nonzero()[0]
            
            if pred.shape == actual.shape and np.all(pred==actual):
                correct += 1
            else:
                bad_cases.add(idx)
            total += 1

        return correct/total, bad_cases, f"{correct}/{total}"