import gc
import os
from pathlib import Path
from typing import Optional, Tuple

import anndata as ad
import matplotlib as mpl
import numpy as np
import pyvista as pv
from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def extract_anndata_info(adata: AnnData):
    # Anndata basic info
    obs_str, var_str, uns_str, obsm_str, layers_str = (
        f"    obs:",
        f"    var:",
        f"    uns:",
        f"    obsm:",
        f"    layers:",
    )

    if len(list(adata.obs.keys())) != 0:
        for key in list(adata.obs.keys()):
            obs_str = obs_str + f" '{key}',"
    if len(list(adata.var.keys())) != 0:
        for key in list(adata.var.keys()):
            var_str = var_str + f" '{key}',"
    if len(list(adata.uns.keys())) != 0:
        for key in list(adata.uns.keys()):
            uns_str = uns_str + f" '{key}',"
    if len(list(adata.obsm.keys())) != 0:
        for key in list(adata.obsm.keys()):
            obsm_str = obsm_str + f" '{key}',"
    if len(list(adata.layers.keys())) != 0:
        for key in list(adata.layers.keys()):
            layers_str = layers_str + f" '{key}',"

    anndata_info = (
        f"AnnData object with n_obs × n_vars = {adata.shape[0]} × {adata.shape[1]}\n"
    )
    for ad_str in [obs_str, var_str, uns_str, obsm_str, layers_str]:
        if ad_str.endswith(","):
            anndata_info = anndata_info + f"{ad_str[:-1]}\n"
    return anndata_info


def abstract_anndata(path: str, X_layer: str = "X") -> Tuple[AnnData, str]:
    adata = ad.read_h5ad(filename=path)
    anndata_info = extract_anndata_info(adata=adata)
    if X_layer != "X":
        assert (
            X_layer in adata.layers.keys()
        ), f"``{X_layer}`` does not exist in `adata.layers`."
        adata.X = adata.layers[X_layer]

    return adata, anndata_info


def abstract_models(path: str, model_ids: Optional[list] = None):
    model_files = os.listdir(path=path)
    model_files.sort()
    assert len(model_files) != 0, "There is no file under this path."

    models = [pv.read(filename=os.path.join(path, f)) for f in model_files]
    if model_ids is None:  # Cannot contain `-` and ` `.
        model_ids = [f"Model{i}" for i in range(len(models))]
    assert len(model_ids) == len(
        models
    ), "The number of model_ids does not equal to that of models."

    return models, model_ids


def sample_dataset(
    path: str,
    X_layer: str = "X",
    pc_model_ids: Optional[list] = None,
    mesh_model_ids: Optional[list] = None,
):
    # Generate anndata object
    anndata_path = os.path.join(path, "h5ad")
    anndata_list = [
        f for f in os.listdir(path=anndata_path) if str(f).endswith(".h5ad")
    ]
    adata, anndata_info = abstract_anndata(
        path=os.path.join(anndata_path, anndata_list[0]),
        X_layer=X_layer,
    )
    anndata_metrices = ["X"] + [i for i in adata.layers.keys()]
    anndata_obs_index = list(adata.obs.index.to_list())
    anndata_var_index = list(adata.var.index.to_list())

    # Check matrices
    matrices_path = os.path.join(path, "matrices")
    if os.path.exists(matrices_path):
        pass
    else:
        Path(matrices_path).mkdir(parents=True, exist_ok=True)
        for matrix_id in anndata_metrices:
            matrix = adata.X if matrix_id == "X" else adata.layers[matrix_id]
            sparse.save_npz(f"{matrices_path}/{matrix_id}_sparse_martrix.npz", matrix)

    # Generate point cloud models
    pc_models_path = os.path.join(path, "pc_models")
    if os.path.exists(pc_models_path):
        pc_model_files = [
            f
            for f in os.listdir(path=pc_models_path)
            if str(f).endswith(".vtk") or str(f).endswith(".vtm")
        ]
        pc_model_files.sort()

        if pc_model_ids is None:
            pc_model_ids = [f"PC_{str(i).split('_')[1]}" for i in pc_model_files]
        _pc_models, pc_model_ids = abstract_models(
            path=pc_models_path, model_ids=pc_model_ids
        )
        pc_models = []
        for pc_model in _pc_models:
            _obs_index = pc_model.point_data["obs_index"]
            if "spatial" in adata.obsm.keys():
                coords = np.asarray(adata[_obs_index, :].obsm["spatial"])
                pc_model.point_data["spatial_X"] = coords[:, 0]
                pc_model.point_data["spatial_Y"] = coords[:, 1]
                pc_model.point_data["spatial_Z"] = (
                    0 if coords.shape[1] == 2 else coords[:, 2]
                )
            if "X_umap" in adata.obsm.keys():
                coords = np.asarray(adata[_obs_index, :].obsm["X_umap"])
                pc_model.point_data["UMAP_X"] = coords[:, 0]
                pc_model.point_data["UMAP_Y"] = coords[:, 1]
                pc_model.point_data["UMAP_Z"] = (
                    0 if coords.shape[1] == 2 else coords[:, 2]
                )

            for obs_key in adata.obs_keys():
                array = np.asarray(adata[_obs_index, :].obs[obs_key])
                if array.dtype == "category":
                    array = np.asarray(array, dtype=str)
                if np.issubdtype(array.dtype, np.number):
                    array = np.asarray(array, dtype=float)
                else:
                    od = {o: i for i, o in enumerate(np.unique(array))}
                    array = np.asarray(list(map(lambda x: od[x], array)), dtype=float)
                pc_model.point_data[obs_key] = array
            pc_models.append(pc_model)
    else:
        pc_models, pc_model_ids = None, None

    # Generate mesh models
    mesh_models_path = os.path.join(path, "mesh_models")
    if os.path.exists(mesh_models_path):
        mesh_model_files = [
            f
            for f in os.listdir(path=mesh_models_path)
            if str(f).endswith(".vtk") or str(f).endswith(".vtm")
        ]
        mesh_model_files.sort()

        if mesh_model_ids is None:
            mesh_model_ids = [f"Mesh_{str(i).split('_')[1]}" for i in mesh_model_files]
        mesh_models, mesh_model_ids = abstract_models(
            path=mesh_models_path, model_ids=mesh_model_ids
        )
    else:
        mesh_models, mesh_model_ids = None, None

    # Custom colors
    custom_colors = []
    for key in adata.uns.keys():
        if str(key).endswith("colors"):
            colors = adata.uns[key]
            if isinstance(colors, dict):
                colors = np.asarray([i for i in colors.values()])
            if isinstance(colors, (np.ndarray, list)):
                custom_colors.append(key)
                nodes = np.linspace(0, 1, num=len(colors))
                if key not in mpl.colormaps():
                    mpl.colormaps.register(
                        LinearSegmentedColormap.from_list(key, list(zip(nodes, colors)))
                    )

    # Delete anndata object
    del adata
    gc.collect()

    return (
        anndata_metrices,
        anndata_info,
        anndata_obs_index,
        anndata_var_index,
        pc_models,
        pc_model_ids,
        mesh_models,
        mesh_model_ids,
        custom_colors,
    )
