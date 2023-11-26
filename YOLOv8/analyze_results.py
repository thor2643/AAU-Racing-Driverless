import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

arr = np.random.randn(1000, 3)
print(arr)





precision_list = []
recall_list = []
AP50_list = []

for folder in os.listdir('runs\detect'):

    # Load the result.csv file into a pandas dataframe
    df = pd.read_csv(f'runs\detect\\{folder}\\results.csv')
    print(df)

    #                   row col
    precisions = df.iloc[:, 4].to_list()
    recalls = df.iloc[:, 5].to_list()
    AP50s = df.iloc[:, 6].to_list()

    max_precision_idx = precisions.index(max(precisions))
    best_precision = [df.iloc[max_precision_idx, 0], df.iloc[max_precision_idx, 4], df.iloc[max_precision_idx, 5], df.iloc[max_precision_idx, 6]]

    max_recall_idx = recalls.index(max(recalls))
    best_recall = [df.iloc[max_recall_idx, 0], df.iloc[max_recall_idx, 4], df.iloc[max_recall_idx, 5], df.iloc[max_recall_idx, 6]]

    max_AP50_idx = AP50s.index(max(AP50s))
    best_AP50 = [df.iloc[max_AP50_idx, 0], df.iloc[max_AP50_idx, 4], df.iloc[max_AP50_idx, 5], df.iloc[max_AP50_idx, 6]]

    precision_list.append(best_precision[1])
    recall_list.append(best_precision[2])
    AP50_list.append(best_precision[3])

fig, ax = plt.subplots()

vals = np.zeros([len(precision_list), 3])
for i in range(len(precision_list)):
    val = [precision_list[i], recall_list[i], AP50_list[i]]
    vals[i, 0] = precision_list[i]
    vals[i, 1] = recall_list[i]
    vals[i, 2] = AP50_list[i]

print(vals)
ax.hist(vals, bins = 2, density=True, histtype="bar", color=["red", "tan", "lime"], label=["Precision", "Recall", "AP50"])
ax.legend()
plt.show()

