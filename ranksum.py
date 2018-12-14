import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle
from scipy.stats import ranksums
#===================================================================
# Load Data

os.chdir(os.getenv("HOME") + "/Desktop/Dropbox/CS229/Project")

d = pd.read_table("./data/svz_data.txt", sep=",")
d = d.drop("Unnamed: 0", axis=1)

y = pd.factorize(d.Age)[0]
X = d.iloc[:, 7:]



def rank_test(covariates, groups):
    """ 
    Wilcoxon rank sum test for the distribution of treatment and control covariates.
    https://github.com/kellieotto/pscore_match/blob/master/pscore_match/match.py
    
    Parameters
    ----------
    covariates : DataFrame 
        Dataframe with one covariate per column.
        If matches are with replacement, then duplicates should be 
        included as additional rows.
    groups : array-like
        treatment assignments, must be 2 groups
    
    Returns
    -------
    A list of p-values, one for each column in covariates
    """    
    colnames = list(covariates.columns)
    J = len(colnames)
    pvalues = np.zeros(J)
    for j in range(J):
        var = covariates[colnames[j]]
        res = ranksums(var[groups == 1], var[groups == 0])
        pvalues[j] = res.pvalue
    return pvalues

tr = rank_test(X, y)

# Feature importance
feature_im = pd.DataFrame(tr, index = X.columns, columns=['coef']).sort_values('coef',ascending=True)
print(feature_im[0:25])
feature_im.to_csv("feature_Imp/ranksum_all.txt".format(c), sep = "\t", header = False)


#===================================================================
celltypes = np.unique(d["Celltype"])
d_celltypes = dict()

for c in ["Endothelial", "Microglia", "Oligodendrocytes"]:
    d_celltypes[c] = d.loc[d['Celltype'] == c]
    print("{} cell count: {}".format(c, d_celltypes[c].shape[0]))

    y = pd.factorize(d_celltypes[c].Age)[0]
    X = d_celltypes[c].iloc[:, 7:]

    ranksum_pvals = rank_test(X, y)
    feature_im = pd.DataFrame(ranksum_pvals, index = X.columns, columns=['coef']).sort_values('coef',ascending=True)
    print(feature_im[0:25])
    feature_im.to_csv("feature_Imp/ranksum_{}.txt".format(c), sep = "\t", header = False)


