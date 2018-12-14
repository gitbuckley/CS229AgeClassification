# Venn Diagram overlaps of important features

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
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42

os.chdir(os.getenv("HOME") + "/Desktop/Dropbox/CS229/Project")
rf_all = pd.read_table("./feature_Imp/rf_all.txt", header = None)
rf_Microglia = pd.read_table("./feature_Imp/rf_Microglia.txt", header = None)
rf_Endothelial = pd.read_table("./feature_Imp/rf_Endothelial.txt", header = None)
rf_Oligodendrocytes = pd.read_table("./feature_Imp/rf_Oligodendrocytes.txt", header = None)


ranksum_all = pd.read_table("./feature_Imp/ranksum_all.txt", header = None)
ranksum_Microglia = pd.read_table("./feature_Imp/ranksum_Microglia.txt", header = None)
ranksum_Endothelial = pd.read_table("./feature_Imp/ranksum_Endothelial.txt", header = None)
ranksum_Oligodendrocytes = pd.read_table("./feature_Imp/ranksum_Oligodendrocytes.txt", header = None)

lasso_all = pd.read_table("./feature_Imp/lasso_all_c0.1.txt", header = None)
lasso_Microglia = pd.read_table("./feature_Imp/lasso_Microglia_c0.1.txt", header = None)
lasso_Endothelial = pd.read_table("./feature_Imp/lasso_Endothelial_c0.1.txt", header = None)
lasso_Oligodendrocytes = pd.read_table("./feature_Imp/lasso_Oligodendrocytes_c0.1.txt", header = None)

lasso_all["abs"] = abs(lasso_all[1])
lasso_all = lasso_all.sort_values(by = ["abs"], ascending=False)
lasso_Microglia["abs"] = abs(lasso_Microglia[1])
lasso_Microglia = lasso_Microglia.sort_values(by = ["abs"], ascending=False)
lasso_Endothelial["abs"] = abs(lasso_Endothelial[1])
lasso_Endothelial = lasso_Endothelial.sort_values(by = ["abs"], ascending=False)
lasso_Oligodendrocytes["abs"] = abs(lasso_Oligodendrocytes[1])
lasso_Oligodendrocytes = lasso_Oligodendrocytes.sort_values(by = ["abs"], ascending=False)

logreg_all = pd.read_table("./feature_Imp/logreg_all_c0.01.txt", header = None)
logreg_all["abs"] = abs(logreg_all[1])
logreg_all = logreg_all.sort_values(by = ["abs"], ascending=False)

top = 100


#=========================================================================
# Plotting
# Question: How many top gene must be included before ranksum DEG and RF features importance
#	ranked lists reflect each other with 90% accuracy?

t = 1000
b = 10
overlap = np.zeros(t-b)
ol_logreg = np.zeros(t-b)
ol_lasso = np.zeros(t-b)
for top in range(b, t):
	# Ranks vs RF
	s1 = pd.merge(ranksum_all[0:top], rf_all[0:top], how='inner', on=[0])
	overlap[top - b] = float(s1.shape[0]) / top
	# Ranks vs LogReg C = 0.01
	s2 = pd.merge(ranksum_all[0:top], logreg_all[0:top], how='inner', on=[0])
	ol_logreg[top - b] = float(s2.shape[0]) / top
	# Ranks vs Lasso
	s3 = pd.merge(ranksum_all[0:top], lasso_all[0:top], how='inner', on=[0])
	ol_lasso[top - b] = float(s3.shape[0]) / top

d = dict()
d["rf"] = overlap.tolist()
d["logreg"] = ol_logreg.tolist()
d["lasso"] = ol_lasso.tolist()

data = pd.DataFrame(d)
ax = sns.lineplot(data = data, palette="tab10", linewidth=1.5)
ax.set_title('Top Gene Overlap with Rank Sum Test')
ax.set_ylabel('% Overlap')
ax.set_xlabel('Number of Genes')
plt.show()
plt.savefig('plots/overlapRankSum.pdf')



#========================================================================
# RANK SUM
s1 = pd.merge(ranksum_all[0:top], ranksum_Microglia[0:top], how='inner', on=[0])
print(s1.shape) # 27/100 8/25
s1 = pd.merge(ranksum_all[0:top], ranksum_Endothelial[0:top], how='inner', on=[0])
print(s1.shape) # 34/100 9/25
s1 = pd.merge(ranksum_all[0:top], ranksum_Oligodendrocytes[0:top], how='inner', on=[0])
print(s1.shape) # 14/100 4/25

# Compare cell types to each other
s1 = pd.merge(ranksum_Endothelial[0:top], ranksum_Microglia[0:top], how='inner', on=[0])
print(s1.shape) # 29/100 5/25
s1 = pd.merge(ranksum_Endothelial[0:top], ranksum_Oligodendrocytes[0:top], how='inner', on=[0])
print(s1.shape) # 12/100 3/25
s1 = pd.merge(ranksum_Microglia[0:top], ranksum_Oligodendrocytes[0:top], how='inner', on=[0])
print(s1.shape) # 12/100 3/25

# Intersection of all 
s1 = pd.merge(ranksum_Microglia[0:top], ranksum_Oligodendrocytes[0:top], how='inner', on=[0])
s2 = pd.merge(s1, ranksum_Endothelial[0:top], how = "inner", on = [0])
print(s2.shape) # 6/100
s3 = pd.merge(s2, ranksum_all[0:top], how = "inner", on = [0])
print(s3.shape) # 5/100

#========================================================================
# RANDOM FOREST
top = 25
# Cell type overlap with SVZ overall
s1 = pd.merge(rf_all[0:top], rf_Microglia[0:top], how='inner', on=[0])
print(s1.shape) # 44/100 11/25
s1 = pd.merge(rf_all[0:top], rf_Endothelial[0:top], how='inner', on=[0])
print(s1.shape) # 45/100 13/25
s1 = pd.merge(rf_all[0:top], rf_Oligodendrocytes[0:top], how='inner', on=[0])
print(s1.shape) # 21/100 11/25

# Compare cell types to each other
s1 = pd.merge(rf_Endothelial[0:top], rf_Microglia[0:top], how='inner', on=[0])
print(s1.shape) # 26/100 7/25
s1 = pd.merge(rf_Endothelial[0:top], rf_Oligodendrocytes[0:top], how='inner', on=[0])
print(s1.shape) # 15/100 6/25
s1 = pd.merge(rf_Microglia[0:top], rf_Oligodendrocytes[0:top], how='inner', on=[0])
print(s1.shape) # 15/100 6/25

# Intersection of all 
s1 = pd.merge(rf_Microglia[0:top], rf_Oligodendrocytes[0:top], how='inner', on=[0])
s2 = pd.merge(s1, rf_Endothelial[0:top], how = "inner", on = [0])
print(s2.shape) # 9/100
s3 = pd.merge(s2, rf_all[0:top], how = "inner", on = [0])
print(s3.shape) # 9/100

#========================================================================
# RANDOM FOREST & RANKSUM
# Compare cell types to each other
s1 = pd.merge(ranksum_all[0:top], rf_all[0:top], how='inner', on=[0])
print(s1.shape) # 56/100
s1 = pd.merge(ranksum_Microglia[0:top], rf_Microglia[0:top], how='inner', on=[0])
print(s1.shape) # 86/100
s1 = pd.merge(ranksum_Endothelial[0:top], rf_Endothelial[0:top], how='inner', on=[0])
print(s1.shape) # 79/100
s1 = pd.merge(ranksum_Oligodendrocytes[0:top], rf_Oligodendrocytes[0:top], how='inner', on=[0])
print(s1.shape) # 80/100

#========================================================================
# RANDOM FOREST & LASSO
# Compare cell types to each other
s1 = pd.merge(lasso_all[0:top], rf_all[0:top], how='inner', on=[0])
print(s1.shape) # 36/100
s1 = pd.merge(lasso_Microglia[0:top], rf_Microglia[0:top], how='inner', on=[0])
print(s1.shape) # 28/100
s1 = pd.merge(lasso_Endothelial[0:top], rf_Endothelial[0:top], how='inner', on=[0])
print(s1.shape) # 54/100
s1 = pd.merge(lasso_Oligodendrocytes[0:top], rf_Oligodendrocytes[0:top], how='inner', on=[0])
print(s1.shape) # 40/100


#=========================================================================
# Plotting
# Question: How many top gene must be included before RF high features importance
#	genes and logistic regression gene lists reflect each other with 90% accuracy?

t = 10000
b = 10
overlap = np.zeros(t-b)
ol_logreg = np.zeros(t-b)
ol_lasso = np.zeros(t-b)
for top in range(b, t):
	# Ranks vs RF
	s1 = pd.merge(rf_all[0:top], ranksum_all[0:top], how='inner', on=[0])
	overlap[top - b] = float(s1.shape[0]) / top
	# Ranks vs LogReg C = 0.01
	s2 = pd.merge(rf_all[0:top], logreg_all[0:top], how='inner', on=[0])
	ol_logreg[top - b] = float(s2.shape[0]) / top
	# Ranks vs Lasso
	s3 = pd.merge(rf_all[0:top], lasso_all[0:top], how='inner', on=[0])
	ol_lasso[top - b] = float(s3.shape[0]) / top

d = dict()
d["ranksum"] = overlap.tolist()
d["logreg"] = ol_logreg.tolist()
d["lasso"] = ol_lasso.tolist()


data = pd.DataFrame(d)
plt.clf()
ax = sns.lineplot(data = data, palette="tab10", linewidth=1.5)
ax.set_title('Top Gene Overlap with RF Features')
ax.set_ylabel('% Overlap')
ax.set_xlabel('Number of Genes')
#plt.show()
plt.savefig('plots/overlapRF_10000.pdf')


#============================================================================
# PLOTTING
# VENN
from matplotlib_venn import venn3, venn3_circles
import venn

# Ex 1
venn3(subsets = (10, 8, 22, 6,9,4,2)) # (Abc, aBc, ABc, abC, AbC, aBC, ABC)
plt.show()

#Ex 2
from matplotlib_venn import venn3, venn3_circles
plt.figure(figsize=(4,4))
v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels = ('A', 'B', 'C'))
v.get_patch_by_id('100').set_alpha(1.0)
v.get_patch_by_id('100').set_color('white')
v.get_label_by_id('100').set_text('Unknown')
v.get_label_by_id('A').set_text('Set "A"')
c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='dashed')
c[0].set_lw(1.0)
c[0].set_ls('dotted')
plt.title("Sample Venn diagram")
plt.annotate('Unknown set', xy=v.get_label_by_id('100').get_position() - np.array([0, 0.05]), xytext=(-70,-70),
             ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))
plt.show()

# Top 25 RF
plt.clf()
venn3(subsets = (17, 18, 1, 18,2,1,5), set_labels = ('Endothelial', 'Oligodendrocytes', 'Microglia'))
plt.title("Random Forest Aging Celltype Top25 Feature Overlaps")
#plt.show()
plt.savefig("plots/venn/rf_top25.pdf")

# Top 25 Ranksum
plt.clf()
venn3(subsets = (20,22,0,20,2,0,3), set_labels = ('Endothelial', 'Oligodendrocytes', 'Microglia'))
plt.title("Rank Sum Top25 Aging Celltype DEG Overlap")
#plt.show()
plt.savefig("plots/venn/ranksum_top25.pdf")

#=====================================================================
# Plotting FEATURE IMPORTANCE BAR
rf_all.columns = ["Gene", "Importance"]

plt.figure(figsize = (5,10))
sns.barplot(y="Gene", x="Importance", data=rf_all.iloc[0:50,:], orient="h", palette=("Blues_d"))
plt.savefig("plots/venn/Rf_feature_barplot.pdf")
plt.show()

