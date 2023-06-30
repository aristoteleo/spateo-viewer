
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

###  How to generate the anndata object to upload

```
import spateo as st

# Load the anndata object
adata = st.read_h5ad("E7_8h_cellbin_v3.h5ad")

# Make sure adata.obsm contains 'spatial' to save the coordinates
adata.obsm['spatial'] = adata.obsm['3d_align_spatial']

# Interactive-viewer will read all info contained in anndata.obs, so please make sure the info you need has been saved in anndata.obs

# Save the anndata object
 adata.write_h5ad("E7_8h_cellbin_v3_new.h5ad", compression="gzip")
```

You can refer to the data structure we include by default in the [**dataset**](https://github.com/aristoteleo/spateo-viewer/blob/main/stviewer/assets/dataset/drosophila_E7_8h/pc_models/0_Embryo_E7_8h_aligned_pc_model.vtk).

### How to upload data

1. Upload file via the tool included in the toolbar in the web application:

    ![UploadFile](https://github.com/aristoteleo/spateo-viewer/blob/main/stviewer/assets/image/upload_file.png)

2. Upload folder via the ``stv_interactive_app.py``:

    ```
   from stviewer.interactive_app import interactive_server, state

   if __name__ == "__main__":
      **state.upload_anndata = None**
      interactive_server.start()
    ```
   
    Change None in ``state.upload_anndata = None`` to the absolute path of the file you want to upload.(Please give priority to this method when used in remote servers)

