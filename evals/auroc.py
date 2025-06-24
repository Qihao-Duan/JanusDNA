import numpy as np
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

preds, targets = [], []
path = "TO THE LOG FILE PATH"
lines = open(path).readlines()
for idx, line in enumerate(lines):
    items = line.strip().split()
    preds.append(float(items[0]))
    targets.append(float(items[1]))

print(roc_auc_score(targets, preds))
print(average_precision_score(targets, preds))