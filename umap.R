# First UMAP plot

library(ggplot2)
library(tidyverse)
library(umap)


# Load data and make basic custome tSNE plot
setwd("~/Desktop/Dropbox/CS229/Project/")

d <- read.csv(file = "data/svz_data.txt", sep = ",")
d["X"] <- NULL

# Select gene expression features only
X <- d[,8:dim(d)[2]]

# Run UMAP
embed <- umap(X)

# Add UMAP coords to main dataset
d$UMAP1 <- embed$layout[, 1]
d$UMAP2 <- embed$layout[, 2]

write.csv(d, file = "data/data_umap.txt", quote = F)

d2 <- filter(d, Celltype %in% c("Astrocytes_qNSCs", "aNSCs_NPCs", "Oligodendrocytes", "Microglia", "Endothelial", "Neuroblasts"))



pdf("plots/UMAP_cell2.pdf", height = 10, width = 10)
ggplot(d, aes(UMAP1, UMAP2, color = Celltype)) + geom_point(shape = '.')
dev.off()

rand.ord <- sample(nrow(d))
d <- d[rand.ord, ] # Randomize rows to avoid covering young samples
pdf("plots/UMAP_age.pdf", height = 10, width = 10)
ggplot(d, aes(UMAP1, UMAP2, color = Age)) + geom_point(alpha = 1, shape = ".")
dev.off()

pdf("plots/UMAP_sample.pdf", height = 10, width = 10)
ggplot(d, aes(UMAP1, UMAP2, color = orig.ident)) + geom_point(alpha = 1, shape = ".")
dev.off()



