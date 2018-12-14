# Create young/old cell balanced datasets for main tissues, 
# and best models of different types.
# Include polynomial feature logistic regression model.

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
from sklearn.ensemble import RandomForestClassifier
from time import time
sns.set_style("whitegrid")
#===================================================================
# Load Data

os.chdir(os.getenv("HOME") + "/Desktop/Dropbox/CS229/Project")
d = pd.read_table("./data/svz_data.txt", sep=",")
d = d.drop("Unnamed: 0", axis=1)

#===================================================================
d.endo = d[d.Celltype == "Endothelial"]
d.endo.o = d.endo[d.endo.Age == "o"] # 1149, mean genes 1155.78
d.endo.y = d.endo[d.endo.Age == "y"] # 991, 1179.85 (-2% delta with age)

d.oligo = d[d.Celltype == "Oligodendrocytes"]
d.oligo.o = d.oligo[d.oligo.Age == "o"] # 691, mean genes 1490.86
d.oligo.y = d.oligo[d.oligo.Age == "y"] # 966, 1492.18 (0% delta)

d.micro = d[d.Celltype == "Microglia"]
d.micro.o = d.micro[d.micro.Age == "o"] # 989, mean genes 1386.62
d.micro.y = d.micro[d.micro.Age == "y"] # 747, 1217.98 (+12% delta with age)

d.astroq = d[d.Celltype == "Astrocytes_qNSCs"]
d.astroq.o = d.astroq[d.astroq.Age == "o"] # 600, mean genes 1128.00
d.astroq.y = d.astroq[d.astroq.Age == "y"] # 407, 1163.11 (-3% delta with age)

# Lowest number is 407. Thus randomly sample 407 rows
#	from each age and celltype combination, then combine
#	into age balanced cell type specific matrices.
s = 407
rs = 7

endo_o = d.endo.o.sample(n=s, random_state = rs)
endo_y = d.endo.y.sample(n=s, random_state = rs)
endo_bal = pd.concat((endo_y, endo_o))

oligo_o = d.oligo.o.sample(n=s, random_state = rs)
oligo_y = d.oligo.y.sample(n=s, random_state = rs)
oligo_bal = pd.concat((oligo_y, oligo_o))

micro_o = d.micro.o.sample(n=s, random_state = rs)
micro_y = d.micro.y.sample(n=s, random_state = rs)
micro_bal = pd.concat((micro_y, micro_o))

astroq_o = d.astroq.o.sample(n=s, random_state = rs)
astroq_y = d.astroq.y.sample(n=s, random_state = rs)
astroq_bal = pd.concat((astroq_y, astroq_o))

d_celltypes = dict()
d_celltypes["Endothelial"] = endo_bal
d_celltypes["Oligodendrocytes"] = oligo_bal
d_celltypes["Microglia"] = micro_bal
d_celltypes["Astrocytes_qNSCs"] = astroq_bal
#===================================================================
# LOGISTIC REGRESSION L2
classifier = LogisticRegression(C = .01, penalty = "l2", n_jobs = -1)
cell_model_data = dict()

for c in d_celltypes.keys():
#for c in ["Endothelial"]:

	y = pd.factorize(d_celltypes[c].Age)[0]
	X = d_celltypes[c].iloc[:, 7:]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4, shuffle=True)

	# Celltype LogReg
	classifier.fit(X_train, y_train)
	pred = classifier.predict(X_test)
	probs = classifier.predict_proba(X_test)[:, 1]
	acc_test = np.mean(pred == y_test)
	print(c)
	print(acc_test)

	cell_model_data[c] = [probs, pred, X_train, X_test, y_train, y_test, acc_test]

multi_roc = dict()
for c in d_celltypes.keys():
    # Load cell type specific classifer and relevant data
    probs, pred, X_train, X_test, y_train, y_test, acc_test = cell_model_data[c]
    # Metrics and Plots
    ## AUC
    fpr, tpr, thresh = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    multi_roc[c] = [fpr, tpr, roc_auc]

# Plot all ROC curves
plt.figure()
lw = 2
for c in ['Astrocytes_qNSCs', 'Endothelial', 'Microglia', "Oligodendrocytes"]:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.3f}, Acc: {2:0.3f}'.format(c, multi_roc[c][2], cell_model_data[c][6]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Young/Old Balanced Logistic Regression Classification ROC')
plt.legend(loc="lower right")
plt.savefig("plots/Balanced/LogReg_roc_combo_Dec7.3.pdf")
plt.close()

#===================================================================
# LOGISTIC REGRESSION L1
classifier = LogisticRegression(C = .1, penalty = "l1", n_jobs = 3)
cell_model_data = dict()

for c in d_celltypes.keys():
#for c in ["Endothelial"]:

	y = pd.factorize(d_celltypes[c].Age)[0]
	X = d_celltypes[c].iloc[:, 7:]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2, shuffle=True)

	# Celltype LogReg
	classifier.fit(X_train, y_train)
	pred = classifier.predict(X_test)
	probs = classifier.predict_proba(X_test)[:, 1]
	acc_test = np.mean(pred == y_test)
	print(c)
	print(acc_test)

	cell_model_data[c] = [probs, pred, X_train, X_test, y_train, y_test, acc_test]

multi_roc = dict()
for c in d_celltypes.keys():
    # Load cell type specific classifer and relevant data
    probs, pred, X_train, X_test, y_train, y_test, acc_test = cell_model_data[c]
    # Metrics and Plots
    ## AUC
    fpr, tpr, thresh = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    multi_roc[c] = [fpr, tpr, roc_auc]

# Plot all ROC curves
plt.figure()
lw = 1.6
for c in ['Astrocytes_qNSCs', 'Endothelial', 'Microglia', "Oligodendrocytes"]:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.3f}, Acc: {2:0.3f}'.format(c, multi_roc[c][2], cell_model_data[c][6]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, .4])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Young/Old Balanced Logistic Regression Classification ROC')
plt.legend(loc="lower right")
plt.savefig("plots/Balanced/Lasso_roc_combo_.4.pdf")
plt.close()


#===================================================================
# LOGISTIC REGRESSION L2 with polynomial interaction terms
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only = True)
lasso_all = pd.read_table("./feature_Imp/lasso_all_c0.1.txt", header = None)
lasso_all["abs"] = abs(lasso_all[1])
lasso_all = lasso_all.sort_values(by = ["abs"], ascending=False)
lasso_genes = lasso_all.iloc[0:1000, 0].tolist()

classifier = LogisticRegression(C = .01, penalty = "l2", n_jobs = 3)
cell_model_data = dict()

for c in d_celltypes.keys():
#for c in ["Endothelial"]:

	y = pd.factorize(d_celltypes[c].Age)[0]
	X = d_celltypes[c].ix[:,lasso_genes]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2, shuffle=True)
	X_train_poly = poly.fit_transform(X_train)
	X_test_poly = poly.fit_transform(X_test)

	# Celltype LogReg
	classifier.fit(X_train_poly, y_train)
	pred = classifier.predict(X_test_poly)
	probs = classifier.predict_proba(X_test_poly)[:, 1]
	acc_test = np.mean(pred == y_test)
	print(c)
	print(acc_test)

	cell_model_data[c] = [probs, pred, X_train, X_test, y_train, y_test, acc_test]

multi_roc = dict()
for c in d_celltypes.keys():
    # Load cell type specific classifer and relevant data
    probs, pred, X_train, X_test, y_train, y_test, acc_test = cell_model_data[c]
    # Metrics and Plots
    ## AUC
    fpr, tpr, thresh = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    multi_roc[c] = [fpr, tpr, roc_auc]

# Plot all ROC curves
plt.figure()
lw = 1.6
for c in ['Astrocytes_qNSCs', 'Endothelial', 'Microglia', "Oligodendrocytes"]:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.3f}, Acc: {2:0.3f}'.format(c, multi_roc[c][2], cell_model_data[c][6]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, .2])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Young/Old Balanced Polynomial LogReg L2')
plt.legend(loc="lower right")
plt.savefig("plots/Balanced/LogregPoly_roc_combo_.2.pdf")
plt.close()

#===================================================================
# RANDOM FOREST

classifier = RandomForestClassifier(n_estimators=1000, max_depth = None, n_jobs=3, random_state=0)
cell_model_data = dict()

for c in d_celltypes.keys():
#for c in ["Endothelial"]:

	y = pd.factorize(d_celltypes[c].Age)[0]
	X = d_celltypes[c].iloc[:, 7:]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2, shuffle=True)

	# Celltype LogReg
	classifier.fit(X_train, y_train)
	pred = classifier.predict(X_test)
	probs = classifier.predict_proba(X_test)[:, 1]
	acc_test = np.mean(pred == y_test)
	print(c)
	print(acc_test)

	cell_model_data[c] = [probs, pred, X_train, X_test, y_train, y_test, acc_test]

multi_roc = dict()
for c in d_celltypes.keys():
    # Load cell type specific classifer and relevant data
    probs, pred, X_train, X_test, y_train, y_test, acc_test = cell_model_data[c]
    # Metrics and Plots
    ## AUC
    fpr, tpr, thresh = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    multi_roc[c] = [fpr, tpr, roc_auc]

# Plot all ROC curves
plt.figure()
lw = 1.6
for c in ['Astrocytes_qNSCs', 'Endothelial', 'Microglia', "Oligodendrocytes"]:
    plt.plot(multi_roc[c][0], multi_roc[c][1], lw=lw, label='{0} AUC: {1:0.3f}, Acc: {2:0.3f}'.format(c, multi_roc[c][2], cell_model_data[c][6]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, .4])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Young/Old Balanced Logistic Regression Classification ROC')
plt.legend(loc="lower right")
plt.savefig("plots/Balanced/RF_roc_combo_.4.pdf")
plt.close()













