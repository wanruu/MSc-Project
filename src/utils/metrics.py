import numpy as np

class Metrics:
    def __init__(self, preds=None, labels=None):
        self.preds = []
        self.labels = []
        if preds is not None and labels is not None:
            self.preds = preds
            self.labels = labels


    def append(self, label, pred):
        self.labels.append(label)
        self.preds.append(pred)


    def compute(self):
        assert len(self.preds) == len(self.labels)

        results = dict()

        # remove prefix
        preds = [[item.replace("I-", "").replace("B-", "") for item in pred] for pred in self.preds]
        labels = [[item.replace("I-", "").replace("B-", "") for item in label] for label in self.labels]
        
        # label list
        label_list = set()
        for label in labels:
            label_list = label_list | set(label)
        label_list = list(label_list)
        
        # token level
        for label in label_list:
            tp, fn, fp, tn = 0, 0, 0, 0
            for p, l in zip(preds, labels):
                x, y = np.equal(p, label), np.equal(l, label)
                tp += np.count_nonzero(x & y)  # pred=True, label=True
                tn += np.count_nonzero(~x & ~y)  # pred=label=False
                fp += np.count_nonzero(x & ~y)  # pred=True, label=False
                fn += np.count_nonzero(~x & y)  # pred=False, label=True
            accuracy = (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn != 0 else 0
            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            results[f"token_{label}_accuracy"] = accuracy
            results[f"token_{label}_precision"] = precision
            results[f"token_{label}_recall"] = recall
            results[f"token_{label}_f1"] = f1

        correct = 0
        total = 0
        for p, l in zip(preds, labels):
            correct += np.count_nonzero(np.equal(p, l))
            total += len(p)
        results["token_overall_accuracy"] = correct / total


        # entity level
        for label in label_list:
            correct = 0
            for p, l in zip(preds, labels):
                pred_idxes = np.where(np.array(p)==label)[0]
                label_idxes = np.where(np.array(l)==label)[0]
                if pred_idxes.shape==label_idxes.shape and np.all(np.equal(pred_idxes, label_idxes)):
                    correct += 1
            results[f"entity_{label}_accuracy"] = correct / len(preds)


        # sentence level
        correct = 0
        for p, l in zip(preds, labels):
            if np.all(np.equal(p, l)):
                correct += 1
        results["sentence_number"] = len(preds)
        results["sentence_accuracy"] = correct / len(preds)
        
        return results
