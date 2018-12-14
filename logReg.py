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
from time import time
#===================================================================
# Load Data

os.chdir(os.getenv("HOME") + "/Desktop/Dropbox/CS229/Project")

d = pd.read_table("./data/svz_data.txt", sep=",")
d = d.drop("Unnamed: 0", axis=1)

y = pd.factorize(d.Age)[0]
X = d.iloc[:, 7:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

#===================================================================
# Basic log regression on full dataset

classifier = LogisticRegression(penalty = "l2", n_jobs = 3)
classifier.fit(X_train, y_train)
acc_train = classifier.score(X_train, y_train)# 1.00
acc_test = classifier.score(X_test, y_test) # 0.93
pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, pred)


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# specify parameters and distributions to sample from
param_grid = {'C': [0.001, 0.01, 0.1, .5, 1, 2, 10, 100]}

# run grid search
grid_search = GridSearchCV(classifier, param_grid=param_grid, cv=10)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))

report(grid_search.cv_results_, n_top = 8)
# Model with rank: 1
# Mean validation score: 0.934 (std: 0.020)
# Parameters: {'C': 0.01}

# Model with rank: 2
# Mean validation score: 0.934 (std: 0.021)
# Parameters: {'C': 0.1}

# Model with rank: 3
# Mean validation score: 0.933 (std: 0.020)
# Parameters: {'C': 0.5}

# Model with rank: 4
# Mean validation score: 0.933 (std: 0.021)
# Parameters: {'C': 100}

# Model with rank: 5
# Mean validation score: 0.933 (std: 0.021)
# Parameters: {'C': 2}

# Model with rank: 6
# Mean validation score: 0.933 (std: 0.021)
# Parameters: {'C': 1}

# Model with rank: 6
# Mean validation score: 0.933 (std: 0.021)
# Parameters: {'C': 10}

# Model with rank: 8
# Mean validation score: 0.926 (std: 0.026)
# Parameters: {'C': 0.001}

best = grid_search.best_estimator_
feature_coef = pd.DataFrame(best.coef_.ravel(), index = X.columns, columns=['coef']).sort_values('coef', ascending=False)
print(feature_coef)
print(sum(feature_coef["coef"] != 0.0)) # 1207
feature_coef.to_csv("feature_Imp/logreg_all_c0.01.txt", sep = "\t", header = False)


# Confusion Matrix
def plot_cm(df_cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
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
classifier = LogisticRegression()

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
    print("{} LogReg test accuracy: {}    Class imbalance: {}".format(c, acc_test, (1-sum(y_test)/len(y_test))))

with open('models/logreg_data.pickle'.format(c), "wb") as file:
    pickle.dump(cell_model_data, file)

# Astrocytes_qNSCs LogReg test accuracy: 0.916    Class imbalance: 0.591
# Endothelial LogReg test accuracy: 0.98    Class imbalance: 0.452
# Microglia LogReg test accuracy: 0.9377880184331797    Class imbalance: 0.412
# Neuroblasts LogReg test accuracy: 0.865    Class imbalance: 0.865
# OPC LogReg test accuracy: 0.928    Class imbalance: 0.642
# Oligodendrocytes LogReg test accuracy: 0.949    Class imbalance: 0.604
# Pericytes LogReg test accuracy: 0.852    Class imbalance: 0.539
# T_Cells LogReg test accuracy: 0.921    Class imbalance: 0.0784
# aNSCs_NPCs LogReg test accuracy: 0.923    Class imbalance: 0.912

multi_roc = dict()
for c in celltypes:
    # Load cell type specific classifer and relevant data
    probs, pred, X_train, X_test, y_train, y_test = cell_model_data[c]
    # Metrics and Plots
    ## Confusion matrix
    cm = confusion_matrix(y_test, pred)
    df_cm = pd.DataFrame(cm/np.sum(cm), columns=["Pred Young", "Pred Old"], index=["Young","Old"])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig("plots/LogReg/confusion_" + c + ".pdf")
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
    plt.savefig("plots/LogReg/roc_" + c + ".pdf")
    plt.close()


# Plot all ROC curves
plt.figure()
lw = 1.6
for c in ['Astrocytes_qNSCs', 'Endothelial', 'Microglia','Neuroblasts','Oligodendrocytes', 'Pericytes','aNSCs_NPCs']:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.3f}'.format(c, multi_roc[c][2]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, .30])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Young/Old Logistic Regression Classification ROC')
plt.legend(loc="lower right")
plt.savefig("plots/LogReg/roc_combo_0.3.pdf")
plt.close()

plt.figure()
lw = 1.6
for c in ['Astrocytes_qNSCs', 'Endothelial', 'Microglia','Neuroblasts','Oligodendrocytes', 'Pericytes','aNSCs_NPCs']:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.3f}'.format(c, multi_roc[c][2]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Young/Old Logistic Regression Classification ROC')
plt.legend(loc="lower right")
plt.savefig("plots/LogReg/roc_combo.pdf")
plt.close()


