suppressMessages({library(Seurat); library(Matrix)})
f <- "/gpfs/gibbs/pi/gerstein/yscorch_data_share/seurat_combined_v3.RDS"
cat("reading", f, "\n"); t0 <- Sys.time()
obj <- readRDS(f)
cat("read in", format(Sys.time()-t0), "\n")
cat("class:", class(obj), "\n")
print(obj)
cat("\n--- assays:", paste(names(obj@assays), collapse=", "), "\n")
for (a in names(obj@assays)) { A <- obj@assays[[a]]; cat(a, ":", class(A)[1], "features", nrow(A), "cells", ncol(A), "\n"); cat("  first features:", paste(head(rownames(A), 15), collapse=", "), "\n") }
cat("\n--- images / spatial slots:", length(obj@images), paste(names(obj@images), collapse=", "), "\n")
if (length(obj@images) > 0) for (im in names(obj@images)) cat(im, ":", class(obj@images[[im]])[1], "\n")
cat("\n--- reductions:", paste(names(obj@reductions), collapse=", "), "\n")
cat("\n--- meta.data columns:\n"); print(colnames(obj@meta.data))
cat("\n--- meta.data head:\n"); print(head(obj@meta.data, 4))
for (cn in colnames(obj@meta.data)) { v <- obj@meta.data[[cn]]; if (is.character(v) || is.factor(v)) { u <- unique(as.character(v)); if (length(u) <= 60) { cat("\n", cn, "(", length(u), "levels ):\n"); print(sort(table(as.character(v)), decreasing=TRUE)) } else cat("\n", cn, ":", length(u), "unique values, e.g.", paste(head(u,5), collapse=", "), "\n") } }
cat("\n--- numeric meta summary:\n"); num <- sapply(obj@meta.data, is.numeric); print(summary(obj@meta.data[, num, drop=FALSE]))
A <- obj@assays[[names(obj@assays)[1]]]
cnt <- tryCatch(GetAssayData(A, layer="counts"), error=function(e) tryCatch(GetAssayData(A, slot="counts"), error=function(e) NULL))
if (!is.null(cnt)) { cs <- Matrix::colSums(cnt); cat("\n--- counts per cell: "); print(summary(cs)); cat("genes detected per cell: "); print(summary(Matrix::colSums(cnt > 0)))
  gl <- read.csv("data/training_gene_list_v3.csv", header=FALSE)[,1]; cat("\noverlap with training gene list (10,134):", sum(rownames(cnt) %in% gl), "of", nrow(cnt), "features\n") }
