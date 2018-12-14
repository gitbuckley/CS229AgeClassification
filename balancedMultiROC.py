# Create ROC plots for a variety of methods.
# Input data: balanced endothelial cells. 990 young, 990 old.

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
from sklearn.svm import LinearSVC
from sklearn import svm
from time import time
from sklearn.preprocessing import PolynomialFeatures
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
#  ===================================================================
# Load Data

os.chdir(os.getenv("HOME") + "/Desktop/Dropbox/CS229/Project")
d = pd.read_table("./data/svz_data.txt", sep=",")
d = d.drop("Unnamed: 0", axis=1)

#  ===================================================================
d.endo = d[d.Celltype == "Endothelial"]
d.endo.o = d.endo[d.endo.Age == "o"]  # 1149, mean genes 1155.78
d.endo.y = d.endo[d.endo.Age == "y"]  # 991, 1179.85 (-2% delta with age)

SAMPLE = 990
RANSTATE = 7

endo_o = d.endo.o.sample(n=SAMPLE, random_state=RANSTATE)
endo_y = d.endo.y.sample(n=SAMPLE, random_state=RANSTATE)
endo_bal = pd.concat((endo_y, endo_o))

# All Genes as features
y = pd.factorize(endo_bal.Age)[0]
X = endo_bal.iloc[:, 7:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

# Classifier Dictionary
clfs = dict()
clfs["LogRegr_L2"] = LogisticRegression(C=0.01, penalty="l2", n_jobs=-1)
clfs["LogRegr_L1"] = LogisticRegression(penalty="l1", n_jobs=-1)
clfs["RandForest"] = RandomForestClassifier(n_estimators=300, n_jobs=-1)
clfs["SVM_Linear"] = svm.SVC(kernel='linear', probability=True)
clfs["SVM_rbf"] = svm.SVC(kernel='rbf', probability=True)
clfs["GBM"] = GradientBoostingClassifier(n_estimators=300)
# No Naive Bayes, KNN

cell_model_data = dict()
# Train each classifier
for classifier in clfs:
    print(classifier)
    clfs[classifier].fit(X_train, y_train)
    pred = clfs[classifier].predict(X_test)
    probs = clfs[classifier].predict_proba(X_test)[:, 1]
    acc_test = np.mean(pred == y_test)
    cell_model_data[classifier] = [probs, pred, y_test]

# AUC
multi_roc = dict()
for c in cell_model_data:
    print(c)
    probs, pred, y_test = cell_model_data[c]
    fpr, tpr, thresh = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    multi_roc[c] = [fpr, tpr, roc_auc]





# Plot all ROC curves in one figure
sns.set_style("white")
plt.figure()
lw = 2
for c in clfs:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.4f}'.format(c, multi_roc[c][2]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # DIAGONAL LINE
plt.xlim([0.0, 1.05])
plt.ylim([0., 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Young/Old Endothelial Cell Classifier ROC')
plt.legend(loc="lower right")
plt.savefig("plots/bal_endo_geneFeats_roc_combo.pdf")
plt.close()

# Plot all ROC curves in one figure
sns.set()
plt.figure()
lw = 2
for c in clfs:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.4f}'.format(c, multi_roc[c][2]))
plt.plot([0, 1], [1, 1], 'k--', lw=lw) # HORIZONTAL
plt.xlim([0.0, .1])
plt.ylim([0.6, 1.001])
plt.savefig("plots/bal_endo_geneFeats_roc_combo_zoom_nokey.pdf")
plt.close()


#  ===================================================================
#  ===================================================================
#  ===================================================================
# With but just lasso selected genes

# Use lasso to get features. Use only text set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=1)
classifier = LogisticRegression(penalty="l1", C=100, n_jobs=-1)
classifier.fit(X_train, y_train)
nonZero = sum(classifier.coef_.ravel() > 0); print(nonZero) # 234
lasso_g = pd.DataFrame(classifier.coef_.ravel(), index=X.columns, columns=['coef']).sort_values('coef',ascending=False)
lasso_g["abs"] = abs(lasso_g["coef"])
lasso_g = lasso_g.sort_values(by = ["abs"], ascending=False)
lasso_genes = lasso_g.index[0:nonZero].tolist()

# Subset features
X_train, X_test, y_train, y_test = train_test_split(X.ix[:,lasso_genes], y, test_size=0.20, shuffle=True, random_state=1)

# Classifier Dictionary
clfs = dict()
clfs["LogRegr_L2"] = LogisticRegression(C=0.01, penalty="l2", n_jobs=-1)
clfs["LogRegr_L1"] = LogisticRegression(penalty="l1", n_jobs=-1)
clfs["RandForest"] = RandomForestClassifier(n_estimators=500, n_jobs=-1)
clfs["SVM_Linear"] = svm.SVC(kernel='linear', probability=True)
clfs["SVM_rbf"] = svm.SVC(kernel='rbf', probability=True)
clfs["GBM"] = GradientBoostingClassifier(n_estimators=500)
# No Naive Bayes, KNN

cell_model_data = dict()
# Train each classifier
for classifier in clfs:
    print(classifier)
    clfs[classifier].fit(X_train, y_train)
    pred = clfs[classifier].predict(X_test)
    probs = clfs[classifier].predict_proba(X_test)[:, 1]
    acc_test = np.mean(pred == y_test)
    cell_model_data[classifier] = [probs, pred, y_test]

# AUC
multi_roc = dict()
for c in cell_model_data:
    print(c)
    probs, pred, y_test = cell_model_data[c]
    fpr, tpr, thresh = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    multi_roc[c] = [fpr, tpr, roc_auc]

# Plot all ROC curves in one figure
sns.set_style("white")
plt.figure()
lw = 2
for c in clfs:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.4f}'.format(c, multi_roc[c][2]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # DIAGONAL LINE
plt.xlim([0.0, 1.05])
plt.ylim([0., 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Young/Old Endothelial Cell Classifier ROC')
plt.legend(loc="lower right")
plt.savefig("plots/bal_endo_lassoFeats_roc_combo.pdf")
plt.close()

# Plot all ROC curves in one figure
sns.set()
plt.figure()
lw = 2
for c in clfs:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.4f}'.format(c, multi_roc[c][2]))
plt.plot([0, 1], [1, 1], 'k--', lw=lw) # HORIZONTAL
plt.xlim([0.0, .1])
plt.ylim([0.6, 1.001])
plt.savefig("plots/bal_endo_lassoFeats_roc_combo_zoom_nokey.pdf")
plt.close()

#  ===================================================================
#  ===================================================================
#  ===================================================================
# Polynomial transformation of lasso genes

# Use lasso to get features. Use only text set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=1)
classifier = LogisticRegression(penalty="l1", C=200, n_jobs=-1)
classifier.fit(X_train, y_train)
nonZero = sum(classifier.coef_.ravel() > 0); print(nonZero) # 240, 956
lasso_g = pd.DataFrame(classifier.coef_.ravel(), index=X.columns, columns=['coef']).sort_values('coef',ascending=False)
lasso_g["abs"] = abs(lasso_g["coef"])
lasso_g = lasso_g.sort_values(by = ["abs"], ascending=False)
lasso_genes = lasso_g.index[0:nonZero].tolist()

# Transform features
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train, X_test, y_train, y_test = train_test_split(X.ix[:,lasso_genes], y, test_size=0.20, shuffle=True, random_state=1)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)


# Classifier Dictionary
clfs = dict()
clfs["LogRegr_L2"] = LogisticRegression(C=0.01, penalty="l2", n_jobs=-1)
clfs["LogRegr_L1"] = LogisticRegression(penalty="l1", n_jobs=-1)
clfs["RandForest"] = RandomForestClassifier(n_estimators=500, n_jobs=-1)
clfs["SVM_Linear"] = svm.SVC(kernel='linear', probability=True)
#clfs["SVM_rbf"] = svm.SVC(kernel='rbf', probability=True)
#clfs["GBM"] = GradientBoostingClassifier(n_estimators=500)
# No Naive Bayes, KNN

cell_model_data = dict()
# Train each classifier
for classifier in clfs:
    print(classifier)
    clfs[classifier].fit(X_train, y_train)
    pred = clfs[classifier].predict(X_test)
    probs = clfs[classifier].predict_proba(X_test)[:, 1]
    acc_test = np.mean(pred == y_test)
    cell_model_data[classifier] = [probs, pred, y_test]

# AUC
multi_roc = dict()
for c in cell_model_data:
    if c == "GBM": continue
    print(c)
    probs, pred, y_test = cell_model_data[c]
    fpr, tpr, thresh = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    multi_roc[c] = [fpr, tpr, roc_auc]

# Plot all ROC curves in one figure
sns.set_style("white")
plt.figure()
lw = 2
for c in clfs:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.4f}'.format(c, multi_roc[c][2]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # DIAGONAL LINE
plt.xlim([0.0, 1.05])
plt.ylim([0., 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Young/Old Endothelial Cell Classifier ROC')
plt.legend(loc="lower right")
plt.savefig("plots/bal_endo_polyFeats_roc_combo.pdf")
plt.close()

# Plot all ROC curves in one figure
sns.set()
plt.figure()
lw = 2
for c in clfs:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.4f}'.format(c, multi_roc[c][2]))
plt.plot([0, 1], [1, 1], 'k--', lw=lw) # HORIZONTAL
plt.xlim([0.0, .1])
plt.ylim([0.6, 1.001])
plt.savefig("plots/bal_endo_polyFeats_roc_combo_zoom_nokey.pdf")
plt.close()


















