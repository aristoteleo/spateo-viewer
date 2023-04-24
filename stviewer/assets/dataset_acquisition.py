import os
from typing import Optional

import anndata as ad
import pyvista as pv

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def abstract_anndata(path: str, X_layer: str = "X"):
    adata = ad.read_h5ad(filename=path)
    if X_layer != "X":
        assert (
            X_layer in adata.layers.keys()
        ), f"``{X_layer}`` does not exist in `adata.layers`."
        adata.X = adata.layers[X_layer]
    return adata


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
    adata = abstract_anndata(
        path=os.path.join(anndata_path, os.listdir(path=anndata_path)[0]),
        X_layer=X_layer,
    )

    # Generate point cloud models
    pc_models_path = os.path.join(path, "pc_models")
    if os.path.exists(pc_models_path):
        pc_model_files = os.listdir(path=pc_models_path)
        pc_model_files.sort()

        if pc_model_ids is None:
            pc_model_ids = [f"PC_{str(i).split('_')[1]}" for i in pc_model_files]
        pc_models, pc_model_ids = abstract_models(
            path=pc_models_path, model_ids=pc_model_ids
        )
    else:
        pc_models, pc_model_ids = None, None

    # Generate mesh models
    mesh_models_path = os.path.join(path, "mesh_models")
    if os.path.exists(mesh_models_path):
        mesh_model_files = os.listdir(path=mesh_models_path)
        mesh_model_files.sort()

        if mesh_model_ids is None:
            mesh_model_ids = [f"Mesh_{str(i).split('_')[1]}" for i in mesh_model_files]
        mesh_models, mesh_model_ids = abstract_models(
            path=mesh_models_path, model_ids=mesh_model_ids
        )
    else:
        mesh_models, mesh_model_ids = None, None

    # Generate morphometric models
    mm_models_path = os.path.join(path, "morphometric_models")
    if os.path.exists(mm_models_path):
        mm_model_folders = os.listdir(path=mm_models_path)
        mm_model_folders.sort()

        mm_models, mm_model_ids = [], []
        for mm_model_folder in mm_model_folders:
            sub_mm_models_path = os.path.join(mm_models_path, mm_model_folder)
            sub_mm_model_files = os.listdir(path=sub_mm_models_path)
            sub_mm_model_files.sort()
            sub_mm_model_ids = [
                f"{mm_model_folder}_{str(i).split('_')[1]}" for i in sub_mm_model_files
            ]
            sub_mm_models, sub_mm_model_ids = abstract_models(
                path=sub_mm_models_path, model_ids=sub_mm_model_ids
            )
            mm_models.extend(sub_mm_models)
            mm_model_ids.extend(sub_mm_model_ids)
    else:
        mm_models, mm_model_ids = None, None

    return (
        adata,
        pc_models,
        pc_model_ids,
        mesh_models,
        mesh_model_ids,
        mm_models,
        mm_model_ids,
    )
