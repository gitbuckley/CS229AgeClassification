import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sn
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle

#===================================================================
# Load Data

os.chdir(os.getenv("HOME") + "/Desktop/Dropbox/CS229/Project")

d = pd.read_table("./data/svz_data.txt", sep=",")
d = d.drop("Unnamed: 0", axis=1)

y = pd.factorize(d.Age)[0]
X = d.iloc[:, 7:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

#===================================================================
# Basic naive bayes on full dataset

classifier = GaussianNB()
classifier.fit(X_train, y_train)
acc_train = classifier.score(X_train, y_train)# 0.834
acc_test = classifier.score(X_test, y_test) # 0.704
pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, pred)

def plot_cm(df_cm):
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()

df_cm = pd.DataFrame(cm/np.sum(cm), columns = ["Pred Young", "Pred Old"], index=["Young","Old"])
plot_cm(df_cm)


#===================================================================
# Basic LogReg on cell types
celltypes = np.unique(d["Celltype"])
d_celltypes = dict()

for c in celltypes:
    d_celltypes[c] = d.loc[d['Celltype'] == c]
    print("{} cell count: {}".format(c, d_celltypes[c].shape[0]))

# 10 cell count: 37
# 9 cell count: 77
# Astrocytes_qNSCs cell count: 1007
# Endothelial cell count: 2140
# Microglia cell count: 1736
# Neuroblasts cell count: 802
# OPC cell count: 167
# Oligodendrocytes cell count: 1657
# Pericytes cell count: 457
# T_Cells cell count: 202
# aNSCs_NPCs cell count: 682


cell_model_data = dict()
classifier = GaussianNB()

for c in celltypes:
#for c in ["Endothelial"]:

    y = pd.factorize(d_celltypes[c].Age)[0]
    X = d_celltypes[c].iloc[:, 7:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2, shuffle=True)
    
    # Celltype LogReg
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    probs = classifier.predict_proba(X_test)[:, 1]
    acc_test = np.mean(pred == y_test)

    with open('models/logreg_{}.pickle'.format(c), "wb") as file:
        pickle.dump(classifier, file)

    cell_model_data[c] = [probs, pred, X_train, X_test, y_train, y_test]
    print("{} Test accuracy: {}    Class imbalance: {}".format(c, acc_test, (1-sum(y_test)/len(y_test))))

with open('models/logreg_data.pickle'.format(c), "wb") as file:
    pickle.dump(cell_model_data, file)

# Astrocytes_qNSCs LogReg test accuracy: 0.6468253968253969
# Endothelial LogReg test accuracy: 0.6485981308411215
# Microglia LogReg test accuracy: 0.7211981566820277
# Neuroblasts LogReg test accuracy: 0.8656716417910447
# OPC LogReg test accuracy: 0.6428571428571429
# Oligodendrocytes LogReg test accuracy: 0.6096385542168675
# Pericytes LogReg test accuracy: 0.6521739130434783
# T_Cells LogReg test accuracy: 0.9215686274509803
# aNSCs_NPCs LogReg test accuracy: 0.9122807017543859

multi_roc = dict()
for c in celltypes:
    # Load cell type specific classifer and relevant data
    probs, pred, X_train, X_test, y_train, y_test = cell_model_data[c]
    # Metrics and Plots
    ## Confusion matrix
    cm = confusion_matrix(y_test, pred)
    df_cm = pd.DataFrame(cm/np.sum(cm), columns=["Pred Young", "Pred Old"], index=["Young","Old"])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig("plots/Naive/confusion_" + c + ".pdf")
    plt.close()
    ## AUC
    fpr, tpr, thresh = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    multi_roc[c] = [fpr, tpr, roc_auc]
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Age ROC: {}'.format(c))
    plt.legend(loc="lower right")
    plt.savefig("plots/Naive/roc_" + c + ".pdf")
    plt.close()


# Plot all ROC curves
plt.figure()
lw = 1.6
for c in ['Astrocytes_qNSCs', 'Endothelial', 'Microglia','Neuroblasts','Oligodendrocytes', 'Pericytes','aNSCs_NPCs']:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.2f}'.format(c, multi_roc[c][2]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, .30])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Young/Old Naive Bayes Classification ROC')
plt.legend(loc="lower right")
plt.savefig("plots/Naive/roc_combo_0.3.pdf")
plt.close()

plt.figure()
lw = 1.6
for c in ['Astrocytes_qNSCs', 'Endothelial', 'Microglia','Neuroblasts','Oligodendrocytes', 'Pericytes','aNSCs_NPCs']:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.2f}'.format(c, multi_roc[c][2]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Young/Old Naive Bayes Classification ROC')
plt.legend(loc="lower right")
plt.savefig("plots/Naive/roc_combo.pdf")
plt.close()


