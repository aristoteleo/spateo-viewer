
## 10 minutes to interactive-viewer

Welcome to interactive-viewer!

interactive-viewer is a web application for interactively clipping 3D models created from spatial transcriptomics data in 3D space.

## How to use

### Installation
You can clone the [**Spateo-Viewer**](https://github.com/aristoteleo/spateo-viewer) with ``git`` and install dependencies with ``pip``:

    git clone https://github.com/aristoteleo/spateo-viewer.git
    cd spateo-viewer
    pip install -r requirements.txt

### Running

    python stv_interactive_app.py --port 1234

###  How to generate the model to upload

```
import spateo as st

# Generate the model
adata = st.read_h5ad("E7_8h_cellbin_v3.h5ad")
model,_ = st.tdr.construct_pc(adata, spatial_key="3d_align_spatial", groupby="Annotation")

# Add gene expression info
obs_index = model.point_data["obs_index"].tolist()
for g in adata.var.index.tolist():
    g_exp = adata[obs_index, g].X.flatten()
    st.tdr.add_model_labels(model=model, labels=g_exp, key_added=g, where="point_data", inplace=True)
    
# Save the model
st.tdr.save_model(model=model, filename="0_Embryo_E7_8h_aligned_pc_model.vtk")
```

You can refer to the data structure we include by default in the [**dataset**](https://github.com/aristoteleo/spateo-viewer/blob/main/stviewer/assets/dataset/drosophila_E7_8h/pc_models/0_Embryo_E7_8h_aligned_pc_model.vtk).

