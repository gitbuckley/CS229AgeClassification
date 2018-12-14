# Read in 4 gene matrices from 10X output using Seurat package.
# Combine, assign clusters, save.
# Matthew Buckley
# Started: Nov 10, 2018. Updated: 

library(Seurat)
library(dplyr)

setwd("~/Desktop/Dropbox/BenNSCProject/10X/0.10X_Data/")

# Read in 10X samples
y1.data <- Read10X(data.dir = "./Y1_Ben/filtered_gene_bc_matrices/mm10/")
o1.data <- Read10X(data.dir = "./O1_Ben/filtered_gene_bc_matrices/mm10/")
y2.data <- Read10X(data.dir = "./Y2_Ben/filtered_gene_bc_matrices/mm10/")
o2.data <- Read10X(data.dir = "./O2_Ben/filtered_gene_bc_matrices/mm10/")

setwd("~/Desktop/Dropbox/CS229/Project")


# Create one data object
y1.obj <- CreateSeuratObject(raw.data = y1.data, project = "y1", min.cells = 3, min.genes = 300)
# 15251 genes across 2310 samples.

# Add samples
svz <- AddSamples(object = y1.obj, new.data = o1.data, add.cell.id = "o1")
svz <- AddSamples(object = svz, new.data = y2.data, add.cell.id = "y2")
svz <- AddSamples(object = svz, new.data = o2.data, add.cell.id = "o2")

# Inspect
svz; head(svz@meta.data); tail(svz@meta.data)
# 27998 genes across 9007 samples.

# Add more labels
age <- substr(svz@meta.data$orig.ident, 1, 1)
replicate <- substr(svz@meta.data$orig.ident, 2, 2)
m <- data.frame("Age" = age, "Replicate" = replicate)
rownames(m) <- rownames(svz@meta.data)
svz <- AddMetaData(object = svz, metadata = m)


# Mitochondria Genes QC
mito.genes <- grep(pattern = "^mt-", x = rownames(x = svz@data), value = TRUE)
percent.mito <- Matrix::colSums(svz@raw.data[mito.genes, ])/Matrix::colSums(svz@raw.data)
svz <- AddMetaData(object = svz, metadata = percent.mito, col.name = "percent.mito")
VlnPlot(object = svz, features.plot = c("nGene", "nUMI", "percent.mito"), group.by = "orig.ident")

# UMI vs Genes
GenePlot(object = svz, gene1 = "nUMI", gene2 = "nGene")
GenePlot(object = svz, gene1 = "nUMI", gene2 = "percent.mito")

# Filter
svz <- FilterCells(object = svz, subset.names = c("nGene", "percent.mito"), low.thresholds = c(200, -Inf), high.thresholds = c(4500, 0.15))
VlnPlot(object = svz, features.plot = c("nGene", "nUMI", "percent.mito"), group.by = "orig.ident")

# Normalize
svz <- NormalizeData(object = svz, normalization.method = "LogNormalize", scale.factor = 10000)

# Find Variable Genes
svz <- FindVariableGenes(object = svz, mean.function = ExpMean, dispersion.function = LogVMR, x.low.cutoff = 0.0125, x.high.cutoff = 3, y.cutoff = 0.75)
length(svz@var.genes) # 6839 genes

# Regress out
svz <- ScaleData(object = svz, vars.to.regress = c())

# PCA
svz <- RunPCA(object = svz, pc.genes = svz@var.genes, do.print = TRUE, pcs.print = 1:5, genes.print = 5)
VizPCA(object = svz, pcs.use = 1:2)
PCHeatmap(object = svz, pc.use = 1:25, cells.use = 200, do.balanced = TRUE, label.columns = FALSE, use.full = FALSE)

PCAPlot(object = svz, dim.1 = 1, dim.2 = 2, group.by = "orig.ident")

save(svz, file = "data/svz.rda")
load(file = "data/svz.rda")


# Explore
# Neighest neighboor graph clustering with different resolutions
# for (i in seq(from = .21, to = .25, by = 0.02)) {
# 	print(i)
# 	svz <- FindClusters(object = svz, reduction.type = "pca", dims.use = 1:20, resolution = i, print.output = 0, force.recalc = TRUE)
# 	svz <- RunTSNE(object = svz, dims.use = 1:10)
# 	pdf(paste0("plots/cluster_res_", i, ".pdf"))
# 	TSNEPlot(object = svz)
# 	dev.off()
# }

# Final cluster & tSNE choice.
svz <- FindClusters(object = svz, reduction.type = "pca", dims.use = 1:20, resolution = .25, print.output = 0, force.recalc = TRUE)
svz <- RunTSNE(object = svz, dims.use = 1:18)

save(svz, file = "data/svz_clustered.rda")

# Find positive cluster marker genes
svz.markers <- FindAllMarkers(object = svz, only.pos = TRUE, min.pct = 0.20, thresh.use = 0.25, type = "MAST")
save(svz.markers, file = "data/svz.markers.mast.rda")


top10 <- svz.markers %>% group_by(cluster) %>% top_n(10, avg_logFC)
# setting slim.col.label to TRUE will print just the cluster IDS instead of every cell name
DoHeatmap(object = svz, genes.use = top10$gene, slim.col.label = TRUE, remove.key = TRUE)

current.cluster.ids <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
new.cluster.ids <- c("Endothelial","Microglia","Oligodendrocytes","Astrocytes_qNSCs",
                     "Neuroblasts","aNSCs_NPCs","Pericytes","T_Cells","OPC", "9", "10")
svz@ident <- plyr::mapvalues(x = svz@ident, from = current.cluster.ids, to = new.cluster.ids)
svz <- AddMetaData(object = svz, metadata = svz@ident, col.name = "Celltype")

TSNEPlot(object = svz, do.label = TRUE, pt.size = 0.5)

#=======================================================================================
# Convert Seurat object into one comphensive dataframe
# Make expression dataframe first,then metadata dataframe, then join.

# Inspect object
print(names(attributes(svz)))

svz_alldata <- cbind(svz@meta.data, t(as.matrix(svz@data)))
svz_alldata <- svz_alldata[, !grepl("res.0*", colnames(svz_alldata))]
svz_alldata[1:9, 1:19]
rownames(svz_alldata) <- NULL

write.csv(svz_alldata, file = "data/svz_data.txt", quote = F)

















