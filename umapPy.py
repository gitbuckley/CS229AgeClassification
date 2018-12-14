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
from sklearn.preprocessing import LabelEncoder
import umap
sns.set()
#===================================================================
# LOAD DATA

os.chdir(os.getenv("HOME") + "/Desktop/Dropbox/CS229/Project")

d = pd.read_table("./data/svz_data.txt", sep=",")
d = d.drop("Unnamed: 0", axis=1)

y = pd.factorize(d.Age)[0]
X = d.iloc[:, 7:]
X_full = d

X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.80, random_state=2)

#==================================================================
# UMAP

umapper = umap.UMAP(n_neighbors=45, min_dist=0.075, n_components=2, metric='euclidean')
embedding = umapper.fit_transform(X_train.iloc[:, 7:])


#==================================================================
# PLOT
sns.set(style="ticks", palette="deep")
sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=X_train["Celltype"], edgecolor='none', sizes = .2)
plt.show()


#===============================================================
# Joint Distribution plot test
sns.set(style="white")

# Show the joint distribution using kernel density estimation
X_train_endo = X_train[X_train['Celltype'].str.match('Endothelial')]
g = sns.jointplot(X_train_endo["Bst2"], X_train_endo["B2m"], kind="", height=7, space=0)
plt.show()