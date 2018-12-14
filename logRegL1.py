import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sn
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model
from sklearn import datasets
from itertools import cycle
from sklearn.linear_model import lasso_path, enet_path
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from time import time
#===================================================================
# Load Data

os.chdir(os.getenv("HOME") + "/Desktop/Dropbox/CS229/Project")

d = pd.read_table("./data/svz_data.txt", sep=",")
d = d.drop("Unnamed: 0", axis=1)

y = pd.factorize(d.Age)[0]
X = d.iloc[:, 7:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

#===================================================================
# Basic log regression on full dataset

classifier = LogisticRegression(penalty = "l1", n_jobs = 3)
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
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# run grid search
grid_search = GridSearchCV(classifier, param_grid=param_grid, cv=5)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))

report(grid_search.cv_results_, n_top = 6)
# Model with rank: 1
# Mean validation score: 0.919 (std: 0.026)
# Parameters: {'C': 0.1}

# Model with rank: 2
# Mean validation score: 0.915 (std: 0.023)
# Parameters: {'C': 1}

# Model with rank: 3
# Mean validation score: 0.913 (std: 0.026)
# Parameters: {'C': 10}

# Model with rank: 4
# Mean validation score: 0.908 (std: 0.026)
# Parameters: {'C': 100}

# Model with rank: 5
# Mean validation score: 0.883 (std: 0.029)
# Parameters: {'C': 0.01}

# Model with rank: 6
# Mean validation score: 0.731 (std: 0.038)
# Parameters: {'C': 0.001}

best = grid_search.best_estimator_
feature_coef = pd.DataFrame(best.coef_.ravel(), index = X.columns, columns=['coef']).sort_values('coef', ascending=False)
print(feature_coef)
print(sum(feature_coef["coef"] != 0.0)) # 1207
feature_coef.to_csv("feature_Imp/lasso_all_c0.1.txt", sep = "\t", header = False)



# Confusion Matrix
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
classifier = LogisticRegression(penalty = "l1", C = .1)

#for c in celltypes:
for c in ["Endothelial", "Microglia", "Oligodendrocytes"]:

    y = pd.factorize(d_celltypes[c].Age)[0]
    X = d_celltypes[c].iloc[:, 7:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2, shuffle=True)
    
    # Celltype LogReg
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    probs = classifier.predict_proba(X_test)[:, 1]
    acc_test = np.mean(pred == y_test) # 0.924

    # Feature importance
    feature_im = pd.DataFrame(classifier.coef_.ravel(), index = X.columns, columns=['coef']).sort_values('coef',ascending=False)
    print(feature_im[0:25])
    feature_im.to_csv("feature_Imp/lasso_{}_c0.1.txt".format(c), sep = "\t", header = False)

    cell_model_data[c] = [probs, pred, X_train, X_test, y_train, y_test]
    print("{} LogReg test accuracy: {}    Class imbalance: {}".format(c, acc_test, (1-sum(y_test)/len(y_test))))

# Astrocytes_qNSCs LogReg test accuracy: 0.884    Class imbalance: 0.591
# Endothelial LogReg test accuracy: 0.968    Class imbalance: 0.452
# Microglia LogReg test accuracy: 0.95    Class imbalance: 0.412
# Neuroblasts LogReg test accuracy: 0.895    Class imbalance: 0.865
# OPC LogReg test accuracy: 0.785    Class imbalance: 0.642
# Oligodendrocytes LogReg test accuracy: 0.951    Class imbalance: 0.604
# Pericytes LogReg test accuracy: 0.817    Class imbalance: 0.539
# T_Cells LogReg test accuracy: 0.9215   Class imbalance: 0.0784
# aNSCs_NPCs LogReg test accuracy: 0.953    Class imbalance: 0.912

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
    plt.savefig("plots/LogRegL1/confusion_" + c + ".pdf")
    plt.close()
    ## AUC
    fpr, tpr, thresh = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    multi_roc[c] = [fpr, tpr, roc_auc]
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Age ROC: {}'.format(c))
    plt.legend(loc="lower right")
    plt.savefig("plots/LogRegL1/roc_" + c + ".pdf")
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
plt.title('Young/Old Lasso Logistic Regression Classification ROC')
plt.legend(loc="lower right")
plt.savefig("plots/LogRegL1/roc_combo_0.3.pdf")
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
plt.title('Young/Old Lasso Logistic Regression Classification ROC')
plt.legend(loc="lower right")
plt.savefig("plots/LogRegL1/roc_combo.pdf")
plt.close()

#=======================================================================
# Lasso Path


diabetes = datasets.load_diabetes()
a = diabetes.data
b = diabetes.target

print("Computing regularization path using the LARS ...")
_, _, coefs = linear_model.lars_path(np.array(X_train), y_train, method='lasso', verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.savefig("plots/LogRegL1/lars_path.pdf")
plt.close()

#=======================================================================


X = np.array(X_train)
y = y_train


X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

# Compute paths

eps = 5e-3  # the smaller it is the longer is the path

print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(np.array(X_train), y_train, eps, fit_intercept=False)

print("Computing regularization path using the positive lasso...")
alphas_positive_lasso, coefs_positive_lasso, _ = lasso_path(
    X, y, eps, positive=True, fit_intercept=False)

print("Computing regularization path using the elastic net...")
alphas_enet, coefs_enet, _ = enet_path(
    X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)

print("Computing regularization path using the positive elastic net...")
alphas_positive_enet, coefs_positive_enet, _ = enet_path(
    X, y, eps=eps, l1_ratio=0.8, positive=True, fit_intercept=False)


colors = cycle(['b', 'r', 'g', 'c', 'k'])
plt.figure()
neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_positive_lasso = -np.log10(alphas_positive_lasso)

for coef_l, coef_pl, c in zip(coefs_lasso, coefs_positive_lasso, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    #l2 = plt.plot(neg_log_alphas_positive_lasso, coef_pl, linestyle='--', c=c)

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso Coefficients')
plt.legend((l1[-1]), ('Lasso'), loc='lower left')
plt.axis('tight')
plt.savefig("plots/LogRegL1/lasso_path.pdf")
plt.close()



#=======================================================================
# Inspect number of non-zero parameters
model = SelectFromModel(classifier, prefit=True)
newX = model.transform(X_train)
newX.shape # 1532

# Find top 100
# Set a minimum threshold of 0.25
sfm = SelectFromModel(classifier, threshold=0.001)
sfm.fit(X_train, y_train)
n_features = sfm.transform(X_train).shape[1]

while n_features > 100:
    sfm.threshold += 0.05
    X_transform = sfm.transform(X_train)
    n_features = X_transform.shape[1]
print(sfm.transform)



feature_idx = sfm.get_support()
feature_name = X_train.columns[feature_idx]
print(feature_name)
# Check performance
classifier.fit(X_transform, y_train)
pred = classifier.predict(X_test[feature_name])
acc_test = np.mean(pred == y_test) # 0.924 (All features) ---> 0.896 (97 Genes)

feature_coef = pd.DataFrame(classifier.coef_.ravel(), index = feature_name, columns=['coef']).sort_values('coef', ascending=False)
print(feature_coef)
print(sum(feature_coef["coef"] != 0.0)) # 97
feature_coef.to_csv("feature_Imp/lasso_all_c0.301.txt", sep = "\t", header = False)

# Compare to DEGs? 
