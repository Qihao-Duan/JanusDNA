import numpy as np
import scipy.stats


# 'HFF': 0, 'H1hESC': 1, 'GM12878': 2, 'IMR90': 3, 'HCT116': 4
lines1 = open("akita/HFF_pred.txt").readlines()
lines2 = open("akita/HFF_tgt.txt").readlines()
corrs = []
for line1, line2 in zip(lines1, lines2):
    preds, targets = [], []
    items1 = line1.strip().split()
    items1 = np.array([float(item) for item in items1]).reshape([200, 200])
    items2 = line2.strip().split()
    items2 = np.array([float(item) for item in items2]).reshape([200, 200])
    for i in range(200):
        for j in range(i+2, 200):

            preds.append(items1[i][j])
            targets.append(items2[i][j])
    cor = scipy.stats.spearmanr(preds, targets)[0]

    corrs.append(cor)
    print(cor)

print(np.average(corrs))

