# Determine how well cell type specific models generalize 
# perform when classifying other cell types

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from time import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
classifier = LinearSVC()
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
sns.set_style("whitegrid")


os.chdir(os.getenv("HOME") + "/Desktop/Dropbox/CS229/Project")
d = pd.read_table("./data/svz_data.txt", sep=",")
d = d.drop("Unnamed: 0", axis=1)

# LogReg on cell types
celltypes = np.unique(d["Celltype"])
d_celltypes = dict()

for c in celltypes:
    d_celltypes[c] = d.loc[d['Celltype'] == c]
    print("{} cell count: {}".format(c, d_celltypes[c].shape[0]))
# Add All cells as a cell type
celltypes = np.append(celltypes, "All")
d_celltypes["All"] = d



cell_model_data = dict()
celltype_clfs = dict()
celltype_testdata = dict()
y = pd.factorize(d.Age)[0]
X = d.drop(["Age"], axis=1)
pre_X_train, pre_X_test, pre_y_train, pre_y_test = train_test_split(X, y, test_size=0.3, random_state=2, shuffle=True)

celltypes = ["All"]
for c in celltypes:
    print(c)
    # Celltype LogReg
    if c == "All":
        X_train = pre_X_train
        y_train = pre_y_train
        X_train = X_train.iloc[:, 7:]
    else:
        X_train = pre_X_train.loc[pre_X_train['Celltype'] == c]
        y_train = pre_y_train[pre_X_train['Celltype'] == c]
        X_train = X_train.iloc[:, 7:]
    # Train and Save
    classifier = LogisticRegression(C=0.01, penalty="l2", n_jobs=-1)
    fitted = classifier.fit(X_train, y_train)
    celltype_clfs[c] = fitted

types = len(celltypes)

accuracy_matrix = np.zeros((types, types))
auroc_matrix = np.zeros((types, types))
col = 0; row = 0

for clf in celltypes:
    print(); print("-----"); print(clf)
    row = 0
    for c in celltypes:
        print(c)
        if c == "All":
            X_test = pre_X_test
            y_test = pre_y_test
            X_test = X_test.iloc[:, 7:]
        else:
            y_test = pre_y_test[pre_X_test['Celltype'] == c]
            X_test = pre_X_test.loc[pre_X_test['Celltype'] == c]
            X_test = X_test.iloc[:, 7:]
        pred = celltype_clfs[clf].predict(X_test)
        probs = celltype_clfs[clf].predict_proba(X_test)[:, 1]
        acc_test = np.mean(pred == y_test)
        print(acc_test)
        accuracy_matrix[row, col] = acc_test
        fpr, tpr, thresh = roc_curve(y_test, probs)
        auroc_matrix[row, col] = auc(fpr, tpr)
        row += 1
    col += 1

drop = ["10", "9", "T_Cells", "Pericytes", "OPC", "Neuroblasts", "aNSCs_NPCs"]
acc_mat = pd.DataFrame(data = accuracy_matrix, index = celltypes, columns = celltypes)
acc_mat = acc_mat.drop(drop)
acc_mat = acc_mat.drop(drop, axis=1)
auc_mat = pd.DataFrame(data = auroc_matrix, index = celltypes, columns = celltypes)
auc_mat = auc_mat.drop(drop)
auc_mat = auc_mat.drop(drop, axis=1)

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
# PLOT Accuracy Heatmap
sns.set(font_scale=1.6)
plt.figure(figsize=(12,10))
sns.heatmap(acc_mat, annot=True, fmt='.3f', cmap = "viridis", linewidths=.75, annot_kws={"size": 20})
plt.xticks(rotation=20)
plt.yticks(rotation=0)
plt.savefig("plots/Generalization/trainTestHeatmap_acc.pdf")

# Plot AUROC heatmap
sns.set(font_scale=1.6)
plt.figure(figsize=(12,10))
sns.heatmap(auc_mat, annot=True, fmt='.3f', cmap = "viridis", linewidths=.75, annot_kws={"size": 20})
plt.xticks(rotation=20)
plt.yticks(rotation=0)
plt.savefig("plots/Generalization/trainTestHeatmap_AUC.pdf")




cm = confusion_matrix(y_test, pred)
df_cm = pd.DataFrame(cm, columns = ["Pred Young", "Pred Old"], index=["Young","Old"])
plt.figure(figsize=(12,10))
sns.heatmap(df_cm, annot=True, cmap = "viridis", linewidths=.75, annot_kws={"size": 20})
plt.savefig("plots/Generalization/trainTestHeatmap_All2.pdf")

df_cm = pd.DataFrame(cm/np.sum(cm), columns = ["Pred Young", "Pred Old"], index=["Young","Old"])
plt.figure(figsize=(12,10))
sns.heatmap(df_cm, annot=True, cmap = "viridis", linewidths=.75, annot_kws={"size": 20})
plt.savefig("plots/Generalization/trainTestHeatmap_All3.pdf")




plot_cm(df_cm)


# Confusion Matrix
def plot_cm(df_cm):
    plt.figure(figsize=(10,12))
    sns.heatmap(df_cm, annot=True, fmt='.3f')
    plt.show()
plot_cm(acc_mat)



