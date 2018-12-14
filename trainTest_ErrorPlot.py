# Plot test/train error with increasing amounts of training data.
# Do this on age balanced endothelial cell data. 

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
sns.set_style("whitegrid")
#===================================================================
# Load Data

os.chdir(os.getenv("HOME") + "/Desktop/Dropbox/CS229/Project")
d = pd.read_table("./data/svz_data.txt", sep=",")
d = d.drop("Unnamed: 0", axis=1)

#===================================================================
d.endo = d[d.Celltype == "Endothelial"] # (2140, 28003)
d.endo.o = d.endo[d.endo.Age == "o"] # 1149, mean genes 1155.78
d.endo.y = d.endo[d.endo.Age == "y"] # 991, 1179.85 (-2% delta with age)

# Lowest number is 407. Thus randomly sample 407 rows
SAMPLES = 991
STATE = 7

endo_o = d.endo.o.sample(n=SAMPLES, random_state=STATE)
endo_y = d.endo.y.sample(n=SAMPLES, random_state=STATE)
endo_bal = pd.concat((endo_y, endo_o))

y = pd.factorize(endo_bal.Age)[0]
X = endo_bal.iloc[:, 7:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=STATE, shuffle=True)
#===================================================================
# CLASSIFIER TYPE
classifier = LogisticRegression(C = .01, penalty = "l2", n_jobs = 3)

train_error = []
test_error = []
train_data_size = []
test_auc = []
for i in range(10, X_train.shape[0], 10):
	train_data_size.append(i)
	classifier.fit(X_train[0:i], y_train[0:i])
	train_error.append(1 - np.mean(classifier.predict(X_train) == y_train))
	test_error.append(1 - np.mean(classifier.predict(X_test) == y_test))
	probs = classifier.predict_proba(X_test)[:, 1]
	fpr, tpr, thresh = roc_curve(y_test, probs)
	test_auc.append(auc(fpr, tpr))

OneMinusAUC = [1-x for x in test_auc]
df = {"TrainingDataSize" : train_data_size, "Training_Error": train_error, "Testing_Error" : test_error, "Test_AUC" : test_auc, "OneMinusAUC" : OneMinusAUC}
plotting_df = pd.DataFrame(df)

#===================================================================
#PLOT
# Plot
lw = 2
plt.clf()
ax = sns.lineplot(x="TrainingDataSize", y="Training_Error", data=plotting_df, label = "Training Error Rate,", lw=lw)
ax = sns.lineplot(x="TrainingDataSize", y="Testing_Error", data=plotting_df, label = "Test Set Error Rate,", lw=lw)
ax = sns.lineplot(x="TrainingDataSize", y ="OneMinusAUC", linestyle='dotted', data = plotting_df, label = "1 - AUROC",  lw=lw)
plt.plot([0, X_train.shape[0]], [0.03, 0.03], linewidth=1, dashes=[4, 4], color = "black", label = "3% Error", lw=lw)
plt.title('Logistic Regression on Endothelial')
plt.xlabel('Training Data Size')
plt.ylabel('')
ax = plt.legend(loc="top right")
plt.show()

plt.savefig("plots/Diagnostics/trainTestError_logreg.pdf")
plt.close()






# ===================================================================
# ===================================================================
# Polynomial featueres using 1000 Genes
poly = PolynomialFeatures(degree=2, interaction_only = True)
lasso_all = pd.read_table("./feature_Imp/lasso_all_c0.1.txt", header = None)
lasso_all["abs"] = abs(lasso_all[1])
lasso_all = lasso_all.sort_values(by = ["abs"], ascending=False)
lasso_genes = lasso_all.iloc[0:1000, 0].tolist()
X_train, X_test, y_train, y_test = train_test_split(X.ix[:,lasso_genes], y, test_size=0.20, random_state=STATE, shuffle=True)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)

# CLASSIFIER TYPE
classifier = LogisticRegression(C = .01, penalty = "l2", n_jobs = 3)
classifier = RandomForestClassifier(n_estimators=500, n_jobs=-1)
classifier = GradientBoostingClassifier(n_estimators=500, loss = "exponential")
classifier = KNeighborsClassifier(n_neighbors=9, n_jobs=-1)

train_error = []
test_error = []
train_data_size = []
test_auc = []
for i in range(10, X_train.shape[0], 100):
	train_data_size.append(i)
	classifier.fit(X_train[0:i], y_train[0:i])
	train_error.append(1 - np.mean(classifier.predict(X_train) == y_train))
	test_error.append(1 - np.mean(classifier.predict(X_test) == y_test))
	probs = classifier.predict_proba(X_test)[:, 1]
	fpr, tpr, thresh = roc_curve(y_test, probs)
	test_auc.append(auc(fpr, tpr))

df = {"TrainingDataSize" : train_data_size[:-1], "Training_Error": train_error, "Testing_Error" : test_error, "Test_AUC" : test_auc}
plotting_df = pd.DataFrame(df)

#===================================================================
#PLOT
sns.set_style("whitegrid")
# Plot
plt.clf()
ax = sns.lineplot(x="TrainingDataSize", y="Training_Error", data=plotting_df, label = "Training Error Rate")
ax = sns.lineplot(x="TrainingDataSize", y="Testing_Error", data=plotting_df, label = "Test Set Error Rate")
#ax = sns.lineplot(x="TrainingDataSize", y ="Test_AUC", data = plotting_df, label = "Test Set Area Under ROC" )
plt.plot([0, X_train.shape[0]], [0.03, 0.03], linewidth=1, dashes=[4, 2], color = "black", label = "3% Error")
plt.title('NN, Endothelial, Feature Selection and interaction terms')
plt.xlabel('Training Data Size')
plt.ylabel('')
ax = plt.legend(loc="center right")
plt.show()

plt.savefig("plots/Diagnostics/trainTestError_NN_9_poly.pdf")
plt.close()

#===================================================================
#===================================================================
# Train on only batch 1

d.endo = d[d.Celltype == "Endothelial"] # (2140, 28003)

d.endo_b1 = d.endo[d.endo.Replicate == 1] # 760
d.endo_b2 = d.endo[d.endo.Replicate == 2] # 1380

d.endo_b1.o = d.endo_b1[d.endo_b1.Age == "o"] # 360
d.endo_b1.y = d.endo_b1[d.endo_b1.Age == "y"] # 400

d.endo_b2.o = d.endo_b2[d.endo_b2.Age == "o"] # 789
d.endo_b2.y = d.endo_b2[d.endo_b2.Age == "y"] # 591

# Lowest cell number is 591 in batch 2.
SAMPLES = 591
STATE = 7
endo_o_b2 = d.endo_b2.o.sample(n=SAMPLES, random_state=STATE)
endo_y_b2 = d.endo_b2.y.sample(n=SAMPLES, random_state=STATE)
endo_bal_b2 = pd.concat((endo_y_b2, endo_o_b2))
SAMPLES = 360
endo_o_b1 = d.endo_b1.o.sample(n=SAMPLES, random_state=STATE)
endo_y_b1 = d.endo_b1.y.sample(n=SAMPLES, random_state=STATE)
endo_bal_b1 = pd.concat((endo_y_b1, endo_o_b1))


y_b1 = pd.factorize(endo_bal_b1.Age)[0]
X_b1 = endo_bal_b1.iloc[:, 7:]

y_b2 = pd.factorize(endo_bal_b2.Age)[0]
X_b2 = endo_bal_b2.iloc[:, 7:]

# Selected features
poly = PolynomialFeatures(degree=2, interaction_only = True)
lasso_all = pd.read_table("./feature_Imp/lasso_all_c0.1.txt", header = None)
lasso_all["abs"] = abs(lasso_all[1])
lasso_all = lasso_all.sort_values(by = ["abs"], ascending=False)
lasso_genes = lasso_all.iloc[0:1000, 0].tolist()

#===================================================================
# TRAINING ON BATCH 1 ONLY
X_train, X_test, y_train, y_test = train_test_split(X_b1.ix[:,lasso_genes], y_b1, test_size=0.20, random_state=STATE, shuffle=True)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)
X_test_otherbatch = poly.fit_transform(X_b2.ix[:,lasso_genes])

# CLASSIFIER TYPE
classifier = LogisticRegression(C = .01, penalty = "l2", n_jobs = 3)

train_error = []
test_error = []
test_error_otherbatch = []
train_data_size = []
test_auc = []
for i in range(10, X_train.shape[0], 10):
	train_data_size.append(i)
	classifier.fit(X_train[0:i], y_train[0:i])
	# Error Rates
	train_error.append(1 - np.mean(classifier.predict(X_train) == y_train))
	test_error.append(1 - np.mean(classifier.predict(X_test) == y_test))
	test_error_otherbatch.append(1 - np.mean(classifier.predict(X_test_otherbatch) == y_b2))
	# AUROC 
	probs = classifier.predict_proba(X_test_otherbatch)[:, 1]
	fpr, tpr, thresh = roc_curve(y_b2, probs)
	test_auc.append(auc(fpr, tpr))

df = {"TrainingDataSize" : train_data_size, "Training_Error": train_error, "Testing_Error" : test_error, "Test_AUC" : test_auc, "Test_Other_Batch":test_error_otherbatch }
plotting_df = pd.DataFrame(df)
plotting_df.to_csv("data/pd_trainingOnBatch1only_polylogreg.csv", sep = "\t")


#===================================================================
#PLOT
# Plot
plt.clf()
ax = sns.lineplot(x="TrainingDataSize", y="Training_Error", data=plotting_df, label = "Batch 1 Training Error Rate")
ax = sns.lineplot(x="TrainingDataSize", y="Testing_Error", data=plotting_df, label = "Batch 1 Test Set Error Rate")
ax = sns.lineplot(x="TrainingDataSize", y="Test_Other_Batch", data=plotting_df, label = "Batch 2 Test Error Rate")
ax = sns.lineplot(x="TrainingDataSize", y ="Test_AUC", data = plotting_df, label = "Batch 2 Test Set Area Under ROC" )
plt.plot([0, X_train.shape[0]], [0.05, 0.05], linewidth=1, dashes=[4, 2], color = "black", label = "5% Error")
plt.title('Logistic Regression, Endo, Poly, ContraBatch')
plt.xlabel('Training Data Size')
plt.ylabel('')
ax = plt.legend(loc="center right")
plt.show()

plt.savefig("plots/Diagnostics/trainTestError_polylogreg_contrabatch1.pdf")
plt.close()


#===================================================================
# TRAINING ON BATCH 2 ONLY
X_train, X_test, y_train, y_test = train_test_split(X_b2.ix[:,lasso_genes], y_b2, test_size=0.20, random_state=STATE, shuffle=True)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)
X_test_otherbatch = poly.fit_transform(X_b1.ix[:,lasso_genes])

# CLASSIFIER TYPE
classifier = LogisticRegression(C = .01, penalty = "l2", n_jobs = 3)

train_error = []
test_error = []
test_error_otherbatch = []
train_data_size = []
test_auc = []
for i in range(10, X_train.shape[0], 10):
	train_data_size.append(i)
	classifier.fit(X_train[0:i], y_train[0:i])
	# Error Rates
	train_error.append(1 - np.mean(classifier.predict(X_train) == y_train))
	test_error.append(1 - np.mean(classifier.predict(X_test) == y_test))
	test_error_otherbatch.append(1 - np.mean(classifier.predict(X_test_otherbatch) == y_b1))
	# AUROC 
	probs = classifier.predict_proba(X_test_otherbatch)[:, 1]
	fpr, tpr, thresh = roc_curve(y_b1, probs)
	test_auc.append(auc(fpr, tpr))

df = {"TrainingDataSize" : train_data_size, "Training_Error": train_error, "Testing_Error" : test_error, "Test_AUC" : test_auc, "Test_Other_Batch":test_error_otherbatch }
plotting_df = pd.DataFrame(df)
plotting_df.to_csv("data/pd_trainingOnBatch2only_polylogreg.csv", sep = "\t")



#===================================================================
#PLOT
# Plot
plt.clf()
ax = sns.lineplot(x="TrainingDataSize", y="Training_Error", data=plotting_df, label = "Batch 2 Training Error Rate")
ax = sns.lineplot(x="TrainingDataSize", y="Testing_Error", data=plotting_df, label = "Batch 2 Test Set Error Rate")
ax = sns.lineplot(x="TrainingDataSize", y="Test_Other_Batch", data=plotting_df, label = "Batch 1 Test Error Rate")
ax = sns.lineplot(x="TrainingDataSize", y ="Test_AUC", data = plotting_df, label = "Batch 1 Test Set Area Under ROC" )
plt.plot([0, X_train.shape[0]], [0.03, 0.03], linewidth=1, dashes=[4, 2], color = "black", label = "3% Error")
plt.title('Logistic Regression, Endo, Poly, ContraBatch2')
plt.xlabel('Training Data Size')
plt.ylabel('')
ax = plt.legend(loc="center right")
plt.show()

plt.savefig("plots/Diagnostics/trainTestError_polylogreg_contraTrainOnBatch2.pdf")
plt.close()










endo_bal = pd.concat((endo_y, endo_o))



y = pd.factorize(endo_bal.Age)[0]
X = endo_bal.iloc[:, 7:]



