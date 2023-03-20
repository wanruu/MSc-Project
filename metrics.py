import torch


class Metrics:
    def __init__(self):
        self.preds = dict()
        self.actuals = dict()
        self.index = 0
    
    def append(self, pred, actual):
        self.preds[self.index] = pred
        self.actuals[self.index] = actual
        self.index += 1
    
    @property
    def acc(self):
        matched = 0
        for x, y in zip(self.preds, self.actuals):
            match_mat = torch.eq(self.preds[x], self.actuals[y])
            matched += int(torch.all(match_mat))
        return matched / len(self.preds)
