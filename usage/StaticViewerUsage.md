
## 10 minutes to static-viewer

Welcome to static-viewer!

static-viewer is a web application for visualizing various models created from spatial transcriptomics data in 3D space, 
including point cloud models, mesh models, trajectory models, vector field models, etc.

## How to use

### Installation
You can clone the [**Spateo-Viewer**](https://github.com/aristoteleo/spateo-viewer) with ``git`` and install dependencies with ``pip``:

    git clone https://github.com/aristoteleo/spateo-viewer.git
    cd spateo-viewer
    pip install -r requirements.txt

### Running

    python stv_static_app.py --port 1234

### Folder Structure of the upload data

```
├── drosophila_E12_13h                              # The folder name 
    ├── h5ad                                        # (Required) The folder includes an anndata object (.h5ad file)
    │   └── E12_13h_cellbin_v3.h5ad                 # (Required) The only one anndata object (.h5ad file)
    └── mesh_models                                 # (Optional) The folder includes mesh models (.vtk files)
    │   ├── 0_Embryo_E12_13h_aligned_mesh_model.vtk # (Optional) The filename start with an ordinal and end with "_mesh_model.vtk"
    │   └── 1_CNS_E12_13h_aligned_mesh_model.vtk    # (Optional) The filename start with an ordinal and end with "_mesh_model.vtk"
    └── pc_models                                   # (Optional) The folder includes point cloud models (.vtk files)
    │   ├── 0_Embryo_E12_13h_aligned_pc_model.vtk   # (Optional) The filename start with an ordinal and end with "_pc_model.vtk"
    │   └── 1_CNS_E12_13h_aligned_pc_model.vtk      # (Optional) The filename start with an ordinal and end with "_pc_model.vtk"
    └── morphometric_models                         # (Optional) The folder includes morphogenesis models of each tissue (.vtk files)
        └── CNS                                     # (Optional) The tissue which includes morphogenesis models
           ├── CNS_Trajectory_E12_13h_model.vtk     # (Optional) The filename start with tissue name and "Trajectory"
           └── CNS_VectorMesh_E12_13h_model.vtk     # (Optional) The filename start with tissue name and "VectorMesh"
           └── CNS_VectorPC_E12_13h_model.vtk       # (Optional) The filename start with tissue name and "VectorPC"
```

You can refer to the folder structure of the data we include by default in the [**dataset**](https://github.com/aristoteleo/spateo-viewer/blob/main/stviewer/assets/dataset).

### How to upload data

1. Upload folder via the tools included in the toolbar in the web application:

   ![UploadFolder](https://github.com/aristoteleo/spateo-viewer/blob/main/stviewer/assets/image/upload_folder.png)

2. Upload folder via the ``stv_static_app.py``:

   ```
        from stviewer.static_app import static_server, state

        if __name__ == "__main__":
            state.selected_dir = None
            static_server.start()
   ```
   
   Change None in ``state.selected_dir = None`` to the absolute path of the folder you want to upload.(Please give priority to this method when used in remote servers)

