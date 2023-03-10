import anndata as ad
import dynamo as dyn
from scipy.sparse import csr_matrix
adata = ad.read_h5ad("./stviewer/assets/dataset/drosophila_E9_10h/h5ad/E9-10h_cellbin.h5ad")
print(adata)
X_counts = csr_matrix(adata.layers["counts_X"].copy())
adata.X = adata.layers["counts_X"].copy()
dyn.pp.normalize_cell_expr_by_size_factors(adata=adata, layers="X", skip_log=False)
X_log1p = adata.X.copy()

spatial_coords = adata.obsm["3d_align_spatial"]
del adata.uns, adata.layers, adata.obsm
adata.obs = adata.obs[
    ["area", "slices", "anno_cell_type", "anno_tissue", "anno_germ_layer"]
]
adata.obsm["spatial"] = spatial_coords
adata.X = X_counts
adata.layers["X_counts"] = X_counts
adata.layers["X_log1p"] = csr_matrix(X_log1p)

adata.write_h5ad("./stviewer/assets/dataset/drosophila_E9_10h/h5ad/E9-10h_cellbin.h5ad", compression="gzip")
