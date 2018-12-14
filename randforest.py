
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import GridSearchCV
from time import time
from scipy.stats import randint as sp_randint

#===================================================================
# Load Data
os.chdir(os.getenv("HOME") + "/Desktop/Dropbox/CS229/Project")

d = pd.read_table("./data/svz_data.txt", sep=",")
d = d.drop("Unnamed: 0", axis=1)

y = pd.factorize(d.Age)[0]
X = d.iloc[:, 7:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2, shuffle=True)

#===================================================================
# Basic rf on full dataset
classifier = RandomForestClassifier(n_estimators=200, n_jobs=3, random_state=0)
classifier.fit(X_train, y_train)
acc_train = classifier.score(X_train, y_train) # 0.72
acc_test = classifier.score(X_test, y_test) # 0.71
pred = classifier.predict(X_test)



#===================================================================
# Parameter search with cross val
# Utility function to report best scores
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html

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
param_grid = {"max_depth": [5, 10, 20, None],
              "min_samples_split": [2, 10],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(classifier, param_grid=param_grid, cv=5)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))

report(grid_search.cv_results_, n_top = 5)
# Essentially invariant:

# Model with rank: 1
# Mean validation score: 0.847 (std: 0.041)
# Parameters: {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2}

# Model with rank: 2
# Mean validation score: 0.846 (std: 0.045)
# Parameters: {'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 2}

# Model with rank: 3
# Mean validation score: 0.846 (std: 0.037)
# Parameters: {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 10}

# Model with rank: 4
# Mean validation score: 0.844 (std: 0.047)
# Parameters: {'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 10}

# Model with rank: 5
# Mean validation score: 0.837 (std: 0.046)
# Parameters: {'criterion': 'entropy', 'max_depth': 20, 'min_samples_split': 2}

best = grid_search.best_estimator_
feature_importances = pd.DataFrame(best.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances[0:100])
feature_importances.to_csv("feature_Imp/rf_all.txt", sep = "\t", header = False)



#===================================================================
# Basic RF on cell types
celltypes = np.unique(d["Celltype"])
d_celltypes = dict()

for c in celltypes:
    d_celltypes[c] = d.loc[d['Celltype'] == c]
    print("{} cell count: {}".format(c, d_celltypes[c].shape[0]))



cell_model_data = dict()
classifier = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=0, n_jobs=3)

for c in ["Microglia", "Oligodendrocytes", "Endothelial"]:
#for c in ["Endothelial"]:

    y = pd.factorize(d_celltypes[c].Age)[0]
    X = d_celltypes[c].iloc[:, 7:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, shuffle=True)
    
    # Fit and predict
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    probs = classifier.predict_proba(X_test)[:, 1]
    acc_test = np.mean(pred == y_test)

    # Feature importance
    feature_im = pd.DataFrame(classifier.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance',ascending=False)
    print(feature_im[0:25])
    feature_im.to_csv("feature_Imp/rf_{}.txt".format(c), sep = "\t", header = False)

    cell_model_data[c] = [probs, pred, X_train, X_test, y_train, y_test]
    print("{} Test accuracy: {}    Class imbalance: {}".format(c, acc_test, (1-sum(y_test)/len(y_test))))


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
    plt.savefig("plots/rf/confusion_" + c + ".pdf")
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
    plt.savefig("plots/rf/roc_" + c + ".pdf")
    plt.close()

# Plot all ROC curves
plt.figure()
lw = 1.6
for c in ['Astrocytes_qNSCs', 'Endothelial', 'Microglia','Neuroblasts','Oligodendrocytes', 'Pericytes','aNSCs_NPCs']:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.3f}'.format(c, multi_roc[c][2]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Young/Old Random Forest ROC')
plt.legend(loc="lower right")
plt.savefig("plots/rf/roc_combo.pdf")
plt.close()

plt.figure()
lw = 1.6
for c in ['Astrocytes_qNSCs', 'Endothelial', 'Microglia','Neuroblasts','Oligodendrocytes', 'Pericytes','aNSCs_NPCs']:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.3f}'.format(c, multi_roc[c][2]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, .305])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Young/Old Random Forest ROC')
plt.legend(loc="lower right")
plt.savefig("plots/rf/roc_combo_0.3.pdf")
plt.close()




